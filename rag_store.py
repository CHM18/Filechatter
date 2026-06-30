"""Persistent chunk storage and vector retrieval for Filechatter."""
from __future__ import annotations

import json
import logging
import re
import shutil
import sqlite3
import threading
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


class RagStore:
    """Manage persisted chunk metadata and a FAISS index on disk."""

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
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
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

    def _keyword_candidates(self, question: str) -> list[int]:
        terms = _extract_terms(question)
        candidates: list[int] = []

        if terms:
            match_query = " OR ".join(f'"{term}"' for term in terms)
            rows = self._connection.execute(
                """
                SELECT rowid
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
                """,
                (match_query, config.KEYWORD_RETRIEVAL_K),
            ).fetchall()
            candidates.extend(row["rowid"] for row in rows)

        phrase = question.strip()
        if phrase:
            like_rows = self._connection.execute(
                """
                SELECT id
                FROM chunks
                WHERE content LIKE ? COLLATE NOCASE OR source LIKE ? COLLATE NOCASE
                ORDER BY source ASC, chunk_index ASC
                LIMIT ?
                """,
                (f"%{phrase}%", f"%{phrase}%", config.KEYWORD_RETRIEVAL_K),
            ).fetchall()
            candidates.extend(row["id"] for row in like_rows)

        deduped: list[int] = []
        seen: set[int] = set()
        for chunk_id in candidates:
            if chunk_id not in seen:
                seen.add(chunk_id)
                deduped.append(chunk_id)
        return deduped

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

            for doc_index, document in enumerate(documents):
                metadata = dict(metadata_list[doc_index] or {})
                source = metadata.get("source") or f"document_{doc_index + 1}"
                metadata.setdefault("collection_name", self.collection_name)
                metadata.setdefault("content_type", self.collection_name)
                document_chunks = self._chunk_document(document)
                uploaded_sources.add(source)

                for chunk_index, (chunk_text, start_char, end_char) in enumerate(document_chunks):
                    chunk_metadata = dict(metadata)
                    chunk_metadata.update(
                        {
                            "source": source,
                            "chunk_index": chunk_index,
                            "start_char": start_char,
                            "end_char": end_char,
                            "collection_name": self.collection_name,
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

            if not pending_rows:
                return {"documents_uploaded": 0, "chunks_uploaded": 0, "total_chunks": self.chunk_count}

            embeddings = self.embedding_model.encode(pending_texts, normalize_embeddings=True)
            embeddings_array = np.asarray(embeddings, dtype=np.float32)

            previous_max_id = self._connection.execute(
                "SELECT COALESCE(MAX(id), 0) FROM chunks"
            ).fetchone()[0]
            cursor = self._connection.cursor()
            cursor.executemany(
                """
                INSERT INTO chunks(source, content, metadata_json, chunk_index, start_char, end_char)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                pending_rows,
            )
            self._connection.commit()

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
            self._connection.commit()

            self.index.add(embeddings_array)
            self._save_index()
            self._rebuild_rowid_cache()
            self._write_info_file()

            return {
                "documents_uploaded": len(uploaded_sources),
                "chunks_uploaded": len(pending_rows),
                "total_chunks": self.chunk_count,
            }

    def search(self, question: str) -> list[SearchResult]:
        with self._lock:
            normalized_question = _normalize_text(question)
            if not normalized_question or self.chunk_count == 0:
                return []

            semantic_candidates = self._semantic_candidates(normalized_question)
            keyword_candidates = self._keyword_candidates(normalized_question)

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
                SELECT source,
                       COUNT(*) AS chunk_count,
                       MIN(created_at) AS first_indexed_at,
                       MAX(created_at) AS last_indexed_at
                FROM chunks
                GROUP BY source
                ORDER BY source ASC
                """
            ).fetchall()
            return [dict(row) for row in rows]

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
            self._connection.commit()
            self.index = faiss.IndexFlatIP(self.dimension)
            self._save_index()
            self._rebuild_rowid_cache()
            self._write_info_file()

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
