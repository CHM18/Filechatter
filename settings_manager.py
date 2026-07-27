"""Runtime-editable settings for Filechatter, persisted in data/settings.json.

.env / config.py stay authoritative for static infrastructure settings
(ports, chunking, embedding model). settings.json holds everything the web
UI can change at runtime: LLM endpoint, permissions, chat defaults.

The permissions structure is stored from M1 on so that the file format is
stable; enforcement arrives with milestone M3.
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
        "chat_defaults": {
            "read_collections": [],
            "write_collections": [],
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
        if group not in {"read", "write"}:
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

        The UI sends its complete desired state, so this can also *remove*
        per-tool overrides - which a deep merge cannot do. Missing sub-keys
        fall back to defaults so a partial payload stays valid.
        """
        with self._lock:
            defaults = default_settings()["permissions"]
            new_permissions = {
                "mode": permissions.get("mode", defaults["mode"]),
                "groups": {**defaults["groups"], **(permissions.get("groups") or {})},
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
