"""Structured, local long-term memory for the browser chat."""
from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import config


MEMORY_CATEGORIES = frozenset({"preference", "workflow"})
MAX_FACT_LENGTH = 320
MAX_RETRIEVAL_RESULTS = 5
TOKEN_RE = re.compile(r"[\w\-]{2,}", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
STOPWORDS = frozenset({
    "about", "after", "again", "also", "and", "are", "because", "been", "before",
    "between", "both", "but", "can", "could", "does", "for", "from", "have", "how",
    "into", "just", "more", "most", "not", "of", "or", "should", "that", "the", "their",
    "then", "there", "these", "they", "this", "those", "to", "use", "was", "what", "when",
    "which", "who", "will", "with", "would", "you", "your",
})


@dataclass(frozen=True, slots=True)
class Memory:
    id: int
    category: str
    fact: str
    enabled: bool
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_fact(fact: str) -> str:
    return WHITESPACE_RE.sub(" ", fact).strip()


def validate_memory(category: str, fact: str) -> tuple[str, str]:
    normalized_category = str(category or "").strip().lower()
    normalized_fact = normalize_fact(str(fact or ""))
    if normalized_category not in MEMORY_CATEGORIES:
        raise ValueError("Memory category must be 'preference' or 'workflow'.")
    if not normalized_fact:
        raise ValueError("Memory fact must not be empty.")
    if len(normalized_fact) > MAX_FACT_LENGTH:
        raise ValueError(f"Memory fact must be at most {MAX_FACT_LENGTH} characters.")
    return normalized_category, normalized_fact


def _query_terms(text: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for term in TOKEN_RE.findall(text.lower()):
        if term not in STOPWORDS and term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


class MemoryStore:
    """Own the memory database without coupling memories to RAG collections."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        root = Path(data_dir if data_dir is not None else config.DATA_DIR)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "memory.sqlite3"
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL CHECK(category IN ('preference', 'workflow')),
                    fact TEXT NOT NULL,
                    fact_normalized TEXT NOT NULL UNIQUE,
                    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chat_memories_fts
                USING fts5(fact, tokenize='unicode61');
                """
            )
            self._connection.commit()

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Memory:
        return Memory(
            id=int(row["id"]),
            category=str(row["category"]),
            fact=str(row["fact"]),
            enabled=bool(row["enabled"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def list(self, include_disabled: bool = True) -> list[Memory]:
        condition = "" if include_disabled else "WHERE enabled = 1"
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM chat_memories {condition} ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def get(self, memory_id: int) -> Memory | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM chat_memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def add(self, category: str, fact: str) -> Memory:
        category, fact = validate_memory(category, fact)
        fact_normalized = fact.casefold()
        now = _now()
        with self._lock:
            try:
                cursor = self._connection.execute(
                    """
                    INSERT INTO chat_memories(category, fact, fact_normalized, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (category, fact, fact_normalized, now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("An identical memory already exists.") from exc
            memory_id = int(cursor.lastrowid)
            self._connection.execute(
                "INSERT INTO chat_memories_fts(rowid, fact) VALUES (?, ?)",
                (memory_id, fact),
            )
            self._connection.commit()
        memory = self.get(memory_id)
        assert memory is not None
        return memory

    def update(self, memory_id: int, category: str, fact: str, enabled: bool) -> Memory | None:
        category, fact = validate_memory(category, fact)
        fact_normalized = fact.casefold()
        now = _now()
        with self._lock:
            try:
                cursor = self._connection.execute(
                    """
                    UPDATE chat_memories
                    SET category = ?, fact = ?, fact_normalized = ?, enabled = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (category, fact, fact_normalized, int(enabled), now, memory_id),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("An identical memory already exists.") from exc
            if cursor.rowcount == 0:
                self._connection.rollback()
                return None
            self._connection.execute("DELETE FROM chat_memories_fts WHERE rowid = ?", (memory_id,))
            self._connection.execute(
                "INSERT INTO chat_memories_fts(rowid, fact) VALUES (?, ?)",
                (memory_id, fact),
            )
            self._connection.commit()
        return self.get(memory_id)

    def delete(self, memory_id: int) -> bool:
        with self._lock:
            cursor = self._connection.execute("DELETE FROM chat_memories WHERE id = ?", (memory_id,))
            if cursor.rowcount == 0:
                self._connection.rollback()
                return False
            self._connection.execute("DELETE FROM chat_memories_fts WHERE rowid = ?", (memory_id,))
            self._connection.commit()
        return True

    def clear(self) -> int:
        with self._lock:
            count = int(self._connection.execute("SELECT COUNT(*) FROM chat_memories").fetchone()[0])
            self._connection.execute("DELETE FROM chat_memories")
            self._connection.execute("DELETE FROM chat_memories_fts")
            self._connection.commit()
        return count

    def search(self, query: str, limit: int = MAX_RETRIEVAL_RESULTS) -> list[Memory]:
        terms = _query_terms(query)
        if not terms or limit <= 0:
            return []
        match_query = " OR ".join(f'"{term}"' for term in terms)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT memory.*
                FROM chat_memories_fts AS fts
                JOIN chat_memories AS memory ON memory.id = fts.rowid
                WHERE chat_memories_fts MATCH ? AND memory.enabled = 1
                ORDER BY bm25(chat_memories_fts), memory.updated_at DESC
                LIMIT ?
                """,
                (match_query, min(limit, MAX_RETRIEVAL_RESULTS)),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._connection.close()