"""Persistent chunk storage and vector retrieval for Filechatter."""
from __future__ import annotations

import json
import logging
import re
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)


WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[\w\-]{2,}", re.UNICODE)


@dataclass(slots=True)
class SearchResult:
    chunk_id: int
    source: str
    content: str
    metadata: dict[str, Any]
    score: float
    rank: int
    collection_name: str


def _normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def _extract_terms(text: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for match in TOKEN_RE.findall(text.lower()):
        if match not in seen:
            seen.add(match)
            terms.append(match)
    return terms


def _parse_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return float(candidate)
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None
    return None


def _to_utc_iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


class RagStore:
    """Manage persisted chunk metadata and a FAISS index on disk."""

    _model_lock = threading.Lock()
    _shared_model_name: str | None = None
    _shared_embedding_model: SentenceTransformer | None = None
    _shared_dimension: int | None = None

    @classmethod
    def _get_shared_embedding_model(cls) -> tuple[SentenceTransformer, int]:
        model_name = config.EMBEDDING_MODEL
        with cls._model_lock:
            if cls._shared_embedding_model is None or cls._shared_model_name != model_name:
                model = SentenceTransformer(model_name)
                cls._shared_embedding_model = model
                cls._shared_dimension = model.get_sentence_embedding_dimension()
                cls._shared_model_name = model_name
            return cls._shared_embedding_model, int(cls._shared_dimension or 0)

    def __init__(self, collection_name: str = "documentation") -> None:
        self.collection_name = collection_name
        self.data_dir = Path(config.DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.collection_dir = self.data_dir / self.collection_name
        self.collection_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = self.collection_dir / "rag.sqlite3"
        self.index_path = self.collection_dir / "chunks.faiss"
        self.info_path = self.collection_dir / "info.xml"
        self._migrate_legacy_store()
        self.embedding_model, self.dimension = self._get_shared_embedding_model()
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        # Defense in depth: the server process is the single writer, but WAL +
        # busy_timeout keep SQLite safe if a stray process opens the same file.
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._ensure_schema()
        self.index = self._load_index()
        self._rebuild_rowid_cache()
        self._write_info_file()
        self._closed = False

    def _migrate_legacy_store(self) -> None:
        if self.collection_name != "documentation":
            return

        legacy_sqlite = self.data_dir / "rag.sqlite3"
        legacy_index = self.data_dir / "chunks.faiss"
        if legacy_sqlite.exists() and not self.sqlite_path.exists():
            shutil.copy2(legacy_sqlite, self.sqlite_path)
        if legacy_index.exists() and not self.index_path.exists():
            shutil.copy2(legacy_index, self.index_path)

    def _write_info_file(self) -> None:
        with self._lock:
            document_count = self._connection.execute(
                "SELECT COUNT(DISTINCT source) FROM chunks"
            ).fetchone()[0]
            chunk_count = self._connection.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]
            last_updated = self._connection.execute(
                "SELECT MAX(created_at) FROM chunks"
            ).fetchone()[0]

        xml = [
            "<collection>",
            f"  <name>{self.collection_name}</name>",
            f"  <sqlite>{self.sqlite_path.name}</sqlite>",
            f"  <index>{self.index_path.name}</index>",
            f"  <documents>{document_count}</documents>",
            f"  <chunks>{chunk_count}</chunks>",
            f"  <lastUpdate>{last_updated or ''}</lastUpdate>",
            "</collection>",
        ]
        self.info_path.write_text("\n".join(xml), encoding="utf-8")

    def _ensure_schema(self) -> None:
        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_source
                ON chunks(source)
                """
            )
            self._connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(source, content, tokenize='unicode61')
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    source TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    file_changed_at REAL,
                    file_changed_at_iso TEXT,
                    last_ingested_at TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_source_name
                ON documents(source_name)
                """
            )
            self._connection.commit()
            self._sync_fts_index()

    def _sync_fts_index(self) -> None:
        with self._lock:
            chunk_count = self._connection.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]
            fts_count = self._connection.execute(
                "SELECT COUNT(*) FROM chunks_fts"
            ).fetchone()[0]

            if chunk_count == fts_count:
                return

            self._connection.execute("DELETE FROM chunks_fts")
            self._connection.execute(
                """
                INSERT INTO chunks_fts(rowid, source, content)
                SELECT id, source, content FROM chunks
                ORDER BY id ASC
                """
            )
            self._connection.commit()

    def _load_index(self) -> faiss.Index:
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        return faiss.IndexFlatIP(self.dimension)

    def _save_index(self) -> None:
        faiss.write_index(self.index, str(self.index_path))

    def _rebuild_rowid_cache(self) -> None:
        with self._lock:
            rows = self._connection.execute(
                "SELECT id FROM chunks ORDER BY id ASC"
            ).fetchall()
            self._row_ids = [row["id"] for row in rows]
            self._row_id_to_faiss_index = {
                row_id: index for index, row_id in enumerate(self._row_ids)
            }

    def _chunk_count_for_source(self, source: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS count FROM chunks WHERE source = ?",
            (source,),
        ).fetchone()
        return int(row["count"] if row else 0)

    def _get_document_record(self, source: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM documents WHERE source = ?",
            (source,),
        ).fetchone()

    def _extract_file_changed_timestamp(self, metadata: dict[str, Any]) -> float | None:
        for key in (
            "file_changed_epoch",
            "file_changed_ts",
            "file_changed_at",
            "modified_at",
            "last_modified",
        ):
            parsed = _parse_timestamp(metadata.get(key))
            if parsed is not None:
                return parsed
        return None

    def _upsert_document_record(
        self,
        source: str,
        metadata: dict[str, Any],
        file_changed_timestamp: float | None,
        ingested_at: str,
    ) -> None:
        source_path = str(metadata.get("path") or source)
        source_name = str(metadata.get("filename") or Path(source_path).name or Path(source).name)
        file_changed_at_iso = _to_utc_iso(file_changed_timestamp)
        self._connection.execute(
            """
            INSERT INTO documents(source, source_path, source_name, file_changed_at, file_changed_at_iso, last_ingested_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET
                source_path=excluded.source_path,
                source_name=excluded.source_name,
                file_changed_at=excluded.file_changed_at,
                file_changed_at_iso=excluded.file_changed_at_iso,
                last_ingested_at=excluded.last_ingested_at
            """,
            (
                source,
                source_path,
                source_name,
                file_changed_timestamp,
                file_changed_at_iso,
                ingested_at,
            ),
        )

    def _delete_chunks_for_source(self, source: str) -> int:
        removed_count = self._chunk_count_for_source(source)
        if removed_count == 0:
            return 0

        self._connection.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE source = ?)",
            (source,),
        )
        self._connection.execute(
            "DELETE FROM chunks WHERE source = ?",
            (source,),
        )
        return removed_count

    def _rebuild_index_from_chunks(self) -> None:
        rows = self._connection.execute(
            "SELECT content FROM chunks ORDER BY id ASC"
        ).fetchall()
        if not rows:
            self.index = faiss.IndexFlatIP(self.dimension)
            self._save_index()
            self._rebuild_rowid_cache()
            return

        texts = [row["content"] for row in rows]
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        embeddings_array = np.asarray(embeddings, dtype=np.float32)
        rebuilt = faiss.IndexFlatIP(self.dimension)
        rebuilt.add(embeddings_array)
        self.index = rebuilt
        self._save_index()
        self._rebuild_rowid_cache()

    def reload(self) -> None:
        with self._lock:
            self._sync_fts_index()
            self.index = self._load_index()
            self._rebuild_rowid_cache()

    def _row_to_result(self, row: sqlite3.Row, score: float, rank: int) -> SearchResult:
        metadata = json.loads(row["metadata_json"])
        return SearchResult(
            chunk_id=row["id"],
            source=row["source"],
            content=row["content"],
            metadata=metadata,
            score=score,
            rank=rank,
            collection_name=metadata.get("collection_name", self.collection_name),
        )

    def _semantic_candidates(self, question: str) -> list[int]:
        query_embedding = self.embedding_model.encode(
            [question], normalize_embeddings=True
        )
        candidate_count = min(config.RETRIEVAL_K, self.chunk_count)
        scores, indices = self.index.search(
            np.asarray(query_embedding, dtype=np.float32), candidate_count
        )

        candidates: list[int] = []
        for faiss_index in indices[0]:
            if faiss_index < 0 or faiss_index >= len(self._row_ids):
                continue
            candidates.append(self._row_ids[faiss_index])
        return candidates

    def _keyword_candidates(self, question: str) -> tuple[list[int], list[int]]:
        """Return body-text and source-path candidates separately.

        The source path contains filename and directory metadata, which is a
        stronger retrieval signal than an incidental body-text match.
        """
        terms = _extract_terms(question)
        content_candidates: list[int] = []
        source_candidates: list[int] = []

        if terms:
            match_query = " OR ".join(f'"{term}"' for term in terms)
            content_rows = self._connection.execute(
                """
                SELECT rowid
                FROM chunks_fts
                WHERE chunks_fts MATCH 'content : (' || ? || ')'
                ORDER BY bm25(chunks_fts)
                LIMIT ?
                """,
                (match_query, config.KEYWORD_RETRIEVAL_K),
            ).fetchall()
            content_candidates.extend(row["rowid"] for row in content_rows)
            source_rows = self._connection.execute(
                """
                SELECT rowid
                FROM chunks_fts
                WHERE chunks_fts MATCH 'source : (' || ? || ')'
                ORDER BY bm25(chunks_fts)
                LIMIT ?
                """,
                (match_query, config.KEYWORD_RETRIEVAL_K),
            ).fetchall()
            source_candidates.extend(row["rowid"] for row in source_rows)

        phrase = question.strip()
        if phrase:
            content_rows = self._connection.execute(
                """
                SELECT id
                FROM chunks
                WHERE content LIKE ? COLLATE NOCASE
                ORDER BY source ASC, chunk_index ASC
                LIMIT ?
                """,
                (f"%{phrase}%", config.KEYWORD_RETRIEVAL_K),
            ).fetchall()
            content_candidates.extend(row["id"] for row in content_rows)
            source_rows = self._connection.execute(
                """
                SELECT id
                FROM chunks
                WHERE source LIKE ? COLLATE NOCASE
                ORDER BY source ASC, chunk_index ASC
                LIMIT ?
                """,
                (f"%{phrase}%", config.KEYWORD_RETRIEVAL_K),
            ).fetchall()
            source_candidates.extend(row["id"] for row in source_rows)

        def dedupe(candidates: list[int]) -> list[int]:
            deduped: list[int] = []
            seen: set[int] = set()
            for chunk_id in candidates:
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    deduped.append(chunk_id)
            return deduped

        return dedupe(content_candidates), dedupe(source_candidates)

    def _chunk_document(self, text: str) -> list[tuple[str, int, int]]:
        normalized_text = text.replace("\r\n", "\n")
        length = len(normalized_text)
        if not normalized_text.strip():
            return []

        chunk_size = max(config.CHUNK_SIZE_CHARS, 1)
        overlap = max(min(config.CHUNK_OVERLAP_CHARS, chunk_size - 1), 0)
        start = 0
        chunks: list[tuple[str, int, int]] = []

        while start < length:
            target_end = min(start + chunk_size, length)
            split_window_start = start + max(chunk_size // 2, 1)
            split_window_end = min(target_end + config.CHUNK_SPLIT_MARGIN, length)
            split_section = normalized_text[split_window_start:split_window_end]
            split_match = None
            for pattern in ("\n\n", ". ", "\n", " "):
                index = split_section.rfind(pattern)
                if index != -1:
                    split_match = split_window_start + index + len(pattern)
                    break

            end = split_match or target_end
            if end <= start:
                end = target_end

            content = normalized_text[start:end].strip()
            if content:
                chunks.append((content, start, end))

            if end >= length:
                break
            start = max(end - overlap, start + 1)

        return chunks

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> dict[str, int]:
        with self._lock:
            metadata_list = metadatas or [{} for _ in documents]
            if len(metadata_list) != len(documents):
                raise ValueError("metadata length must match documents length")

            pending_texts: list[str] = []
            pending_rows: list[tuple[str, str, str, int, int, int]] = []
            uploaded_sources: set[str] = set()
            replaced_sources: set[str] = set()
            skipped_outdated_sources: set[str] = set()
            source_updates: dict[str, tuple[dict[str, Any], float | None, str]] = {}
            deleted_chunks = 0

            for doc_index, document in enumerate(documents):
                metadata = dict(metadata_list[doc_index] or {})
                source = str(metadata.get("source") or f"document_{doc_index + 1}")
                metadata.setdefault("collection_name", self.collection_name)
                metadata.setdefault("content_type", self.collection_name)
                file_changed_ts = self._extract_file_changed_timestamp(metadata)
                file_changed_at = _to_utc_iso(file_changed_ts)
                if file_changed_ts is not None:
                    metadata["file_changed_epoch"] = file_changed_ts
                if file_changed_at is not None:
                    metadata["file_changed_at"] = file_changed_at

                existing_chunk_count = self._chunk_count_for_source(source)
                if existing_chunk_count > 0:
                    existing_record = self._get_document_record(source)
                    existing_changed_ts = None
                    if existing_record is not None:
                        existing_changed_ts = _parse_timestamp(existing_record["file_changed_at"])

                    if file_changed_ts is None:
                        skipped_outdated_sources.add(source)
                        continue
                    if existing_changed_ts is not None and file_changed_ts <= existing_changed_ts:
                        skipped_outdated_sources.add(source)
                        continue

                    deleted_chunks += self._delete_chunks_for_source(source)
                    replaced_sources.add(source)

                document_chunks = self._chunk_document(document)
                ingested_at = datetime.now(timezone.utc).isoformat()
                source_updates[source] = (metadata, file_changed_ts, ingested_at)
                uploaded_sources.add(source)

                if not document_chunks:
                    continue

                for chunk_index, (chunk_text, start_char, end_char) in enumerate(document_chunks):
                    chunk_metadata = dict(metadata)
                    chunk_metadata.update(
                        {
                            "source": source,
                            "chunk_index": chunk_index,
                            "start_char": start_char,
                            "end_char": end_char,
                            "collection_name": self.collection_name,
                            "ingested_at": ingested_at,
                            "source_path": str(metadata.get("path") or source),
                            "source_name": str(metadata.get("filename") or Path(source).name),
                        }
                    )
                    pending_texts.append(chunk_text)
                    pending_rows.append(
                        (
                            source,
                            chunk_text,
                            json.dumps(chunk_metadata),
                            chunk_index,
                            start_char,
                            end_char,
                        )
                    )

            if pending_rows:
                embeddings = self.embedding_model.encode(pending_texts, normalize_embeddings=True)
                embeddings_array = np.asarray(embeddings, dtype=np.float32)

                previous_max_id = self._connection.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM chunks"
                ).fetchone()[0]
                self._connection.executemany(
                    """
                    INSERT INTO chunks(source, content, metadata_json, chunk_index, start_char, end_char)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    pending_rows,
                )

                inserted_ids = list(
                    range(previous_max_id + 1, previous_max_id + len(pending_rows) + 1)
                )
                self._connection.executemany(
                    "INSERT INTO chunks_fts(rowid, source, content) VALUES (?, ?, ?)",
                    [
                        (inserted_id, row[0], row[1])
                        for inserted_id, row in zip(inserted_ids, pending_rows)
                    ],
                )
            else:
                embeddings_array = None

            for source, (metadata, changed_ts, ingested_at) in source_updates.items():
                self._upsert_document_record(source, metadata, changed_ts, ingested_at)

            self._connection.commit()

            if replaced_sources:
                self._rebuild_index_from_chunks()
            elif embeddings_array is not None and len(pending_rows) > 0:
                self.index.add(embeddings_array)
                self._save_index()
                self._rebuild_rowid_cache()

            self._write_info_file()

            return {
                "documents_uploaded": len(uploaded_sources),
                "chunks_uploaded": len(pending_rows),
                "total_chunks": self.chunk_count,
                "replaced_documents": len(replaced_sources),
                "skipped_outdated_documents": len(skipped_outdated_sources),
                "deleted_chunks": deleted_chunks,
            }

    def search(self, question: str) -> list[SearchResult]:
        with self._lock:
            normalized_question = _normalize_text(question)
            if not normalized_question or self.chunk_count == 0:
                return []

            semantic_candidates = self._semantic_candidates(normalized_question)
            keyword_candidates, metadata_candidates = self._keyword_candidates(normalized_question)

            reciprocal_rank_offset = 60
            combined_scores: dict[int, float] = {}

            for rank, chunk_id in enumerate(semantic_candidates, start=1):
                combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + (
                    config.SEMANTIC_WEIGHT / (reciprocal_rank_offset + rank)
                )

            for rank, chunk_id in enumerate(keyword_candidates, start=1):
                combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + (
                    config.KEYWORD_WEIGHT / (reciprocal_rank_offset + rank)
                )

            for rank, chunk_id in enumerate(metadata_candidates, start=1):
                combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + (
                    config.METADATA_KEYWORD_WEIGHT / (reciprocal_rank_offset + rank)
                )

            ordered_chunk_ids = [
                chunk_id
                for chunk_id, _ in sorted(
                    combined_scores.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ]

            results: list[SearchResult] = []
            budget_chars = 0
            seen_sources: dict[str, int] = {}

            for rank, chunk_id in enumerate(ordered_chunk_ids, start=1):
                row = self._connection.execute(
                    "SELECT * FROM chunks WHERE id = ?",
                    (chunk_id,),
                ).fetchone()
                if row is None:
                    continue

                source = row["source"]
                source_hits = seen_sources.get(source, 0)
                if source_hits >= config.MAX_CHUNKS_PER_SOURCE:
                    continue

                content = row["content"]
                prospective_budget = budget_chars + len(content)
                if results and prospective_budget > config.MAX_CONTEXT_CHARS:
                    continue
                if not results and len(content) > config.MAX_CONTEXT_CHARS:
                    content = content[: config.MAX_CONTEXT_CHARS].rstrip()
                    prospective_budget = len(content)

                results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        source=source,
                        content=content,
                        metadata=json.loads(row["metadata_json"]),
                        score=float(combined_scores[chunk_id]),
                        rank=rank,
                        collection_name=self.collection_name,
                    )
                )
                seen_sources[source] = source_hits + 1
                budget_chars = prospective_budget

                if budget_chars >= config.MAX_CONTEXT_CHARS:
                    break

            return results

    def list_sources(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT c.source,
                       COUNT(*) AS chunk_count,
                       MIN(c.created_at) AS first_indexed_at,
                       MAX(c.created_at) AS last_indexed_at,
                       d.source_path,
                       d.source_name,
                       d.file_changed_at_iso AS file_changed_at,
                       d.last_ingested_at AS last_ingested_at
                FROM chunks c
                LEFT JOIN documents d ON d.source = c.source
                GROUP BY c.source, d.source_path, d.source_name, d.file_changed_at_iso, d.last_ingested_at
                ORDER BY c.source ASC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def clear_source(self, source: str) -> dict[str, Any]:
        with self._lock:
            removed_chunks = self._delete_chunks_for_source(source)
            self._connection.execute(
                "DELETE FROM documents WHERE source = ?",
                (source,),
            )
            self._connection.commit()
            if removed_chunks > 0:
                self._rebuild_index_from_chunks()
            self._write_info_file()
            return {
                "source": source,
                "deleted_chunks": removed_chunks,
                "total_chunks": self.chunk_count,
            }

    def get_chunks(
        self,
        source: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        with self._lock:
            query = (
                "SELECT id, source, content, metadata_json, chunk_index, start_char, end_char, created_at "
                "FROM chunks"
            )
            params: list[Any] = []

            if source:
                query += " WHERE source = ?"
                params.append(source)

            query += " ORDER BY source ASC, chunk_index ASC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = self._connection.execute(query, params).fetchall()
            chunks: list[dict[str, Any]] = []
            for row in rows:
                chunk = dict(row)
                chunk["metadata"] = json.loads(chunk.pop("metadata_json"))
                chunk["collection_name"] = self.collection_name
                chunks.append(chunk)
            return chunks

    def clear(self) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM chunks")
            self._connection.execute("DELETE FROM chunks_fts")
            self._connection.execute("DELETE FROM documents")
            self._connection.commit()
            self.index = faiss.IndexFlatIP(self.dimension)
            self._save_index()
            self._rebuild_rowid_cache()
            self._write_info_file()

    def chunk_count_for_source(self, source: str) -> int:
        with self._lock:
            return self._chunk_count_for_source(source)

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return len(self._row_ids)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            logger.debug("Failed to close RagStore during cleanup", exc_info=True)
