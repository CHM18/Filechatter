"""Collection registry for Filechatter.

Owns data/collections.json (configuration + last_updated timestamp) and the
RagStore instances. This module is the single place that creates, opens,
deletes, and locks collections - both the FastAPI server and the tool
registry go through it, which is what makes the single-writer architecture
work.

Live counts (documents/chunks) are intentionally NOT stored in
collections.json; they are read from each collection's SQLite store on
demand to avoid denormalization drift.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import re
import shutil
import stat
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from rag_store import RagStore

logger = logging.getLogger(__name__)

COLLECTIONS_FILE = "collections.json"
RESERVED_NAMES = {"all", "*"}
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class CollectionLockedError(RuntimeError):
    """Raised when a collection directory cannot be removed because a file is
    still locked - typically because another Filechatter server process is
    running against the same data directory (Windows [WinError 32])."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _force_remove(func, path, _exc) -> None:
    """rmtree onexc handler: clear read-only bit and retry once."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        # Leave it for the retry loop in _robust_rmtree to raise a clean error.
        raise


def _robust_rmtree(path: Path, attempts: int = 5, delay: float = 0.2) -> None:
    """Delete a directory tree, tolerating transient Windows file locks.

    Windows keeps a file handle briefly after a process releases it (and AV
    scanners / the search indexer can hold one too), so a single rmtree can
    fail even when nothing is genuinely using the file. Retry a few times,
    then surface a clear error instead of a raw stack trace.
    """
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            shutil.rmtree(path, onexc=_force_remove)
            return
        except OSError as exc:
            last_error = exc
            # Encourage any lingering handles (e.g. a just-closed sqlite
            # connection or a __del__-pending RagStore) to be released.
            gc.collect()
            time.sleep(delay)
    raise CollectionLockedError(
        f"Could not delete '{path.name}': a file is still in use "
        f"({last_error}). Make sure only one Filechatter server is running "
        f"against this data directory, then try again."
    )


def normalize_extensions(extensions: list[str] | None) -> list[str]:
    """Normalize to lowercase dotted extensions. Empty list means 'no restriction'."""
    if not extensions:
        return []
    normalized: list[str] = []
    for extension in extensions:
        value = str(extension).strip().lower()
        if not value:
            continue
        if not value.startswith("."):
            value = "." + value
        if value not in normalized:
            normalized.append(value)
    return normalized


@dataclass(slots=True)
class CollectionEntry:
    name: str
    description: str = ""
    allowed_extensions: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    last_updated: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "allowed_extensions": list(self.allowed_extensions),
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CollectionEntry":
        return cls(
            name=str(data["name"]),
            description=str(data.get("description") or ""),
            allowed_extensions=normalize_extensions(data.get("allowed_extensions")),
            created_at=str(data.get("created_at") or utc_now_iso()),
            last_updated=data.get("last_updated"),
        )

    def allows_extension(self, extension: str) -> bool:
        """Empty allowed_extensions means every supported type is accepted."""
        if not self.allowed_extensions:
            return True
        value = extension.strip().lower()
        if value and not value.startswith("."):
            value = "." + value
        return value in self.allowed_extensions


class CollectionsManager:
    """Registry of collections plus lifecycle of their RagStore instances."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir or config.DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.data_dir / COLLECTIONS_FILE
        self._lock = threading.RLock()
        self._entries: dict[str, CollectionEntry] = {}
        self._stores: dict[str, RagStore] = {}
        self._write_locks: dict[str, threading.RLock] = {}
        self._load()
        self._discover_existing_dirs()

    # ------------------------------------------------------------------
    # Registry persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.registry_path.exists():
            return
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to read %s: %s", self.registry_path, exc)
            return
        for item in payload.get("collections", []):
            try:
                entry = CollectionEntry.from_dict(item)
            except (KeyError, TypeError) as exc:
                logger.warning("Skipping malformed collection entry %r: %s", item, exc)
                continue
            self._entries[entry.name] = entry

    def _save(self) -> None:
        """Atomic write: temp file in same directory, then os.replace."""
        with self._lock:
            payload = {
                "version": 1,
                "collections": [
                    entry.to_dict() for entry in sorted(self._entries.values(), key=lambda e: e.name)
                ],
            }
            tmp_path = self.registry_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp_path, self.registry_path)

    def _discover_existing_dirs(self) -> None:
        """Register on-disk collection directories that predate collections.json."""
        with self._lock:
            changed = False
            for child in sorted(self.data_dir.iterdir()):
                if not child.is_dir():
                    continue
                if not (child / "rag.sqlite3").exists():
                    continue
                name = child.name.lower()
                if name in RESERVED_NAMES or name in self._entries:
                    continue
                if not NAME_RE.match(name):
                    logger.warning("Ignoring collection directory with invalid name: %s", child)
                    continue
                self._entries[name] = CollectionEntry(
                    name=name,
                    description="Discovered existing collection",
                )
                changed = True
                logger.info("Discovered existing collection: %s", name)
            if changed:
                self._save()

    # ------------------------------------------------------------------
    # Name handling
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_name(name: str | None) -> str:
        """Normalize a collection name; returns 'all' for empty/wildcard values."""
        if not name:
            return "all"
        normalized = name.strip().lower()
        if normalized in RESERVED_NAMES:
            return "all"
        return normalized

    @staticmethod
    def validate_name(name: str) -> None:
        if name in RESERVED_NAMES:
            raise ValueError(f"'{name}' is a reserved name")
        if not NAME_RE.match(name):
            raise ValueError(
                "Collection names must start with a letter or digit and contain only "
                "lowercase letters, digits, '-' and '_' (max 64 characters)"
            )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def list_entries(self) -> list[CollectionEntry]:
        with self._lock:
            return sorted(self._entries.values(), key=lambda entry: entry.name)

    def get_entry(self, name: str) -> CollectionEntry | None:
        with self._lock:
            return self._entries.get(self.normalize_name(name))

    def exists(self, name: str) -> bool:
        return self.get_entry(name) is not None

    def create(
        self,
        name: str,
        description: str = "",
        allowed_extensions: list[str] | None = None,
    ) -> CollectionEntry:
        normalized = name.strip().lower()
        self.validate_name(normalized)
        with self._lock:
            if normalized in self._entries:
                raise ValueError(f"Collection '{normalized}' already exists")
            entry = CollectionEntry(
                name=normalized,
                description=description.strip(),
                allowed_extensions=normalize_extensions(allowed_extensions),
            )
            self._entries[normalized] = entry
            self._save()
        # Materialize the store directory right away so the collection is usable.
        self.get_store(normalized)
        return entry

    def ensure(self, name: str) -> CollectionEntry:
        """Return the entry, implicitly registering unknown names.

        Implicit creation keeps the historic behavior where ingesting into an
        unknown collection name simply creates it.
        """
        normalized = self.normalize_name(name)
        if normalized == "all":
            raise ValueError("'all' does not map to a single collection")
        with self._lock:
            entry = self._entries.get(normalized)
            if entry is None:
                self.validate_name(normalized)
                entry = CollectionEntry(
                    name=normalized,
                    description="Created automatically during ingest",
                )
                self._entries[normalized] = entry
                self._save()
                logger.info("Implicitly created collection: %s", normalized)
            return entry

    def update_entry(
        self,
        name: str,
        description: str | None = None,
        allowed_extensions: list[str] | None = None,
    ) -> CollectionEntry:
        normalized = self.normalize_name(name)
        with self._lock:
            entry = self._entries.get(normalized)
            if entry is None:
                raise KeyError(f"Unknown collection: {normalized}")
            if description is not None:
                entry.description = description.strip()
            if allowed_extensions is not None:
                entry.allowed_extensions = normalize_extensions(allowed_extensions)
            self._save()
            return entry

    def touch(self, name: str) -> None:
        """Record a content change (ingest, clear, delete-source) timestamp."""
        normalized = self.normalize_name(name)
        with self._lock:
            entry = self._entries.get(normalized)
            if entry is None:
                return
            entry.last_updated = utc_now_iso()
            self._save()

    def delete(self, name: str) -> dict[str, Any]:
        normalized = self.normalize_name(name)
        if normalized == "all":
            raise ValueError("Refusing to delete all collections at once")
        # Lock ordering everywhere is write_lock -> _lock (ingest paths hold the
        # write lock and then call touch(), which takes _lock). Respect it here
        # to avoid an AB-BA deadlock with in-flight writes.
        collection_write_lock = self.write_lock(normalized)
        with collection_write_lock:
            with self._lock:
                entry = self._entries.get(normalized)
                if entry is None:
                    raise KeyError(f"Unknown collection: {normalized}")
                store = self._stores.pop(normalized, None)
                if store is not None:
                    store.close()
                collection_dir = self.data_dir / normalized
                if collection_dir.exists():
                    # store.close() released the sqlite handle in this process;
                    # retry to absorb Windows' brief handle-release lag.
                    _robust_rmtree(collection_dir)
                del self._entries[normalized]
                self._write_locks.pop(normalized, None)
                self._save()
        logger.info("Deleted collection: %s", normalized)
        return {"name": normalized, "deleted": True}

    # ------------------------------------------------------------------
    # Stores and locking
    # ------------------------------------------------------------------

    def get_store(self, name: str) -> RagStore:
        normalized = self.normalize_name(name)
        if normalized == "all":
            raise ValueError("'all' does not map to a single store")
        with self._lock:
            self.ensure(normalized)
            store = self._stores.get(normalized)
            if store is None:
                store = RagStore(normalized)
                self._stores[normalized] = store
            return store

    def selected_stores(self, name: "str | list[str] | None") -> list[RagStore]:
        """Resolve a collection selector to stores.

        Accepts 'all', a single name, or a list of names (used by the chat
        agent to scope reads to the selected read-collections). Unknown or
        empty selectors resolve to no stores rather than raising, so a chat
        with no read database simply retrieves nothing.
        """
        if isinstance(name, (list, tuple, set)):
            stores: list[RagStore] = []
            seen: set[str] = set()
            for item in name:
                normalized = self.normalize_name(item)
                if normalized == "all":
                    return [self.get_store(entry.name) for entry in self.list_entries()]
                if normalized in seen or not self.exists(normalized):
                    continue
                seen.add(normalized)
                stores.append(self.get_store(normalized))
            return stores

        normalized = self.normalize_name(name)
        if normalized == "all":
            return [self.get_store(entry.name) for entry in self.list_entries()]
        return [self.get_store(normalized)]

    def write_lock(self, name: str) -> threading.RLock:
        normalized = self.normalize_name(name)
        with self._lock:
            lock = self._write_locks.get(normalized)
            if lock is None:
                lock = threading.RLock()
                self._write_locks[normalized] = lock
            return lock

    def close_all(self) -> None:
        with self._lock:
            for store in self._stores.values():
                store.close()
            self._stores.clear()

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def describe(self, name: str) -> dict[str, Any]:
        """Entry metadata merged with live counts from the store."""
        normalized = self.normalize_name(name)
        entry = self.get_entry(normalized)
        if entry is None:
            raise KeyError(f"Unknown collection: {normalized}")
        store = self.get_store(normalized)
        return {
            **entry.to_dict(),
            "document_count": len(store.list_sources()),
            "chunk_count": store.chunk_count,
        }

    def describe_all(self) -> list[dict[str, Any]]:
        return [self.describe(entry.name) for entry in self.list_entries()]
