"""Runtime-editable settings for Filechatter, persisted in data/settings.json.

.env / config.py stay authoritative for static infrastructure settings
(ports, chunking, embedding model). settings.json holds everything the web
UI can change at runtime: LLM endpoint, global tool permissions, and MCP host behavior.

The permissions structure is kept backward-compatible with the original
read/write group names while new UI changes use three global tool buckets.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)

SETTINGS_FILE = "settings.json"

VALID_PERMISSION_VALUES = {"ask", "allow", "deny"}
VALID_PERMISSION_MODES = {"simple", "advanced"}
VALID_ON_ASK = {"host_confirm", "elicit", "deny"}


def default_settings() -> dict[str, Any]:
    return {
        "version": 1,
        "llm": {
            "provider": "lmstudio",
            "base_url": config.LM_STUDIO_BASE_URL,
            "model": config.LM_STUDIO_MODEL,
            "api_key": None,
            "temperature": config.TEMPERATURE,
            "timeout_seconds": config.LM_STUDIO_TIMEOUT,
        },
        "permissions": {
            "mode": "simple",
            "groups": {"read": "allow", "write": "ask"},
            "tools": {},
        },
        "mcp_hosts": {
            "on_ask": "host_confirm",
        },
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate(settings: dict[str, Any]) -> None:
    permissions = settings.get("permissions", {})
    mode = permissions.get("mode")
    if mode not in VALID_PERMISSION_MODES:
        raise ValueError(f"permissions.mode must be one of {sorted(VALID_PERMISSION_MODES)}")
    for group, value in permissions.get("groups", {}).items():
        if group not in {"read", "write", "read_tools", "ingest_tools", "collection_tools"}:
            raise ValueError(f"Unknown permission group: {group}")
        if value not in VALID_PERMISSION_VALUES:
            raise ValueError(f"permissions.groups.{group} must be one of {sorted(VALID_PERMISSION_VALUES)}")
    for tool_name, value in permissions.get("tools", {}).items():
        if value not in VALID_PERMISSION_VALUES:
            raise ValueError(f"permissions.tools.{tool_name} must be one of {sorted(VALID_PERMISSION_VALUES)}")
    on_ask = settings.get("mcp_hosts", {}).get("on_ask")
    if on_ask not in VALID_ON_ASK:
        raise ValueError(f"mcp_hosts.on_ask must be one of {sorted(VALID_ON_ASK)}")


class SettingsManager:
    """Thread-safe settings store with atomic file writes."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir or config.DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / SETTINGS_FILE
        self._lock = threading.RLock()
        self._settings = default_settings()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._save()
            return
        try:
            stored = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to read %s, using defaults: %s", self.path, exc)
            return
        # Merge on top of defaults so new fields appear automatically after upgrades.
        self._settings = _deep_merge(default_settings(), stored)
        old_groups = stored.get("permissions", {}).get("groups", {})
        groups = self._settings["permissions"]["groups"]
        if "read" in old_groups and "read_tools" not in old_groups:
            groups["read_tools"] = old_groups["read"]
        if "write" in old_groups and "ingest_tools" not in old_groups:
            groups["ingest_tools"] = old_groups["write"]
        if "write" in old_groups and "collection_tools" not in old_groups:
            groups["collection_tools"] = old_groups["write"]

    def _save(self) -> None:
        with self._lock:
            tmp_path = self.path.with_suffix(".json.tmp")
            tmp_path.write_text(
                json.dumps(self._settings, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp_path, self.path)

    def get(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._settings)

    def update(self, changes: dict[str, Any]) -> dict[str, Any]:
        """Deep-merge changes into the current settings, validate, persist."""
        with self._lock:
            candidate = _deep_merge(self._settings, changes)
            candidate["version"] = 1
            _validate(candidate)
            self._settings = candidate
            self._save()
            return copy.deepcopy(self._settings)

    def set_permissions(self, permissions: dict[str, Any]) -> dict[str, Any]:
        """Replace the whole permissions block (not a merge).

        Missing sub-keys fall back to defaults so a partial payload stays valid.
        Legacy per-tool overrides are retained for compatibility but are no
        longer exposed by the web UI.
        """
        with self._lock:
            defaults = default_settings()["permissions"]
            supplied_groups = permissions.get("groups") or {}
            groups = {**defaults["groups"], **supplied_groups}
            if "read" in supplied_groups and "read_tools" not in supplied_groups:
                groups["read_tools"] = supplied_groups["read"]
            if "write" in supplied_groups and "ingest_tools" not in supplied_groups:
                groups["ingest_tools"] = supplied_groups["write"]
            if "write" in supplied_groups and "collection_tools" not in supplied_groups:
                groups["collection_tools"] = supplied_groups["write"]
            new_permissions = {
                "mode": permissions.get("mode", defaults["mode"]),
                "groups": groups,
                "tools": dict(permissions.get("tools") or {}),
            }
            candidate = copy.deepcopy(self._settings)
            candidate["permissions"] = new_permissions
            _validate(candidate)
            self._settings = candidate
            self._save()
            return copy.deepcopy(self._settings)

    def public_view(self) -> dict[str, Any]:
        """Settings with secrets masked, safe to send to the browser."""
        view = self.get()
        llm = view.get("llm", {})
        if llm.get("api_key"):
            llm["api_key"] = "***"
        return view
