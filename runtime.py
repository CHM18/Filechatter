"""Process-wide singletons for the Filechatter server.

The FastAPI server is the single writer for all collection data, so exactly
one CollectionsManager / IngestJobManager / SettingsManager exists per
process. Everything (API endpoints, tool registry) resolves them through
this module; tests call reset() with a patched config.DATA_DIR to get a
fresh, isolated state.
"""
from __future__ import annotations

import atexit
import threading
from typing import Any

from collections_manager import CollectionsManager
from ingest_jobs import IngestJobManager
from settings_manager import SettingsManager


class ChatSessionStore:
    """In-memory chat sessions (single-user local tool).

    A session holds the OpenAI-format message history plus the read/write
    database scope chosen for it, so the agent can pause for a confirmation
    and resume on the next request without the browser round-tripping the
    whole tool-call state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, dict[str, Any]] = {}

    def create(self, session_id: str, read_cols: list[str], write_cols: list[str]) -> dict[str, Any]:
        with self._lock:
            session = {"messages": [], "read_cols": read_cols, "write_cols": write_cols}
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


_lock = threading.Lock()
_collections: CollectionsManager | None = None
_jobs: IngestJobManager | None = None
_settings: SettingsManager | None = None
_chat_sessions: ChatSessionStore | None = None


def collections() -> CollectionsManager:
    global _collections
    with _lock:
        if _collections is None:
            _collections = CollectionsManager()
            atexit.register(_collections.close_all)
        return _collections


def jobs() -> IngestJobManager:
    global _jobs
    with _lock:
        if _jobs is None:
            _jobs = IngestJobManager(collections_manager_unlocked())
        return _jobs


def collections_manager_unlocked() -> CollectionsManager:
    """collections() without re-acquiring _lock (for use while holding it)."""
    global _collections
    if _collections is None:
        _collections = CollectionsManager()
        atexit.register(_collections.close_all)
    return _collections


def settings() -> SettingsManager:
    global _settings
    with _lock:
        if _settings is None:
            _settings = SettingsManager()
        return _settings


def chat_sessions() -> ChatSessionStore:
    global _chat_sessions
    with _lock:
        if _chat_sessions is None:
            _chat_sessions = ChatSessionStore()
        return _chat_sessions


def reset() -> None:
    """Close and drop all singletons (used by tests and shutdown)."""
    global _collections, _jobs, _settings, _chat_sessions
    with _lock:
        if _collections is not None:
            _collections.close_all()
        _collections = None
        _jobs = None
        _settings = None
        _chat_sessions = None
