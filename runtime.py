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
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from collections_manager import CollectionsManager
from ingest_jobs import IngestJobManager
from memory_store import MemoryStore
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


class LocalModelQueue:
    """Serialize local LLM work, prioritizing queued extraction before a new chat turn."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._tasks: deque[Callable[[], None]] = deque()
        self._active = False
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="memory-extraction", daemon=True)
        self._worker.start()

    def enqueue_extraction(self, task: Callable[[], None]) -> None:
        with self._condition:
            if self._closed:
                return
            self._tasks.append(task)
            self._condition.notify_all()

    def has_pending_work(self) -> bool:
        with self._condition:
            return self._active or bool(self._tasks)

    @contextmanager
    def chat_turn(self) -> Iterator[None]:
        with self._condition:
            while not self._closed and (self._active or self._tasks):
                self._condition.wait()
            if self._closed:
                raise RuntimeError("The local model queue is closed.")
            self._active = True
        try:
            yield
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._closed and (not self._tasks or self._active):
                    self._condition.wait()
                if self._closed:
                    return
                task = self._tasks.popleft()
                self._active = True
            try:
                task()
            except Exception:
                # Extraction is opportunistic; a failed task must never stop the queue.
                pass
            finally:
                with self._condition:
                    self._active = False
                    self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._tasks.clear()
            self._condition.notify_all()


_lock = threading.Lock()
_collections: CollectionsManager | None = None
_jobs: IngestJobManager | None = None
_settings: SettingsManager | None = None
_chat_sessions: ChatSessionStore | None = None
_memory_store: MemoryStore | None = None
_model_queue: LocalModelQueue | None = None


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


def memories() -> MemoryStore:
    global _memory_store
    with _lock:
        if _memory_store is None:
            _memory_store = MemoryStore()
        return _memory_store


def model_queue() -> LocalModelQueue:
    global _model_queue
    with _lock:
        if _model_queue is None:
            _model_queue = LocalModelQueue()
        return _model_queue


def reset() -> None:
    """Close and drop all singletons (used by tests and shutdown)."""
    global _collections, _jobs, _settings, _chat_sessions, _memory_store, _model_queue
    with _lock:
        if _collections is not None:
            _collections.close_all()
        if _memory_store is not None:
            _memory_store.close()
        if _model_queue is not None:
            _model_queue.close()
        _collections = None
        _jobs = None
        _settings = None
        _chat_sessions = None
        _memory_store = None
        _model_queue = None
