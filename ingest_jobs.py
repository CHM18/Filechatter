"""Background ingest jobs for Filechatter.

Extracted from rag_mcp_server.py when the MCP server became a thin proxy.
Jobs now always run inside the single long-lived FastAPI server process, so
the old MCP-timeout batching (pause after 45s, resume on next call) is gone:
a job simply runs to completion in its worker thread. Concurrent job
requests are queued FIFO instead of being rejected.
"""
from __future__ import annotations

import logging
import threading
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from collections_manager import CollectionsManager
from document_loader import discover_documents, load_document

logger = logging.getLogger(__name__)

JOB_LOG_DIR = Path(__file__).resolve().parent / "logs"


@dataclass(slots=True)
class IngestJob:
    job_id: str
    path: str
    recursive: bool
    total_files: int
    collection_name: str = "all"
    job_type: str = "documents"
    log_stamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    status: str = "queued"
    processed_files: int = 0
    succeeded_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    skipped_by_type: int = 0
    documents_uploaded: int = 0
    chunks_uploaded: int = 0
    current_file: str | None = None
    error: str | None = None
    recent_errors: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time)
    finished_at: float | None = None


class IngestJobManager:
    """Runs ingest jobs one at a time; additional jobs wait in a FIFO queue."""

    def __init__(self, collections: CollectionsManager) -> None:
        self.collections = collections
        self._lock = threading.Lock()
        self._jobs: dict[str, IngestJob] = {}
        self._active_job_id: str | None = None
        self._queue: deque[str] = deque()
        JOB_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_job(
        self,
        path: str,
        recursive: bool,
        collection_name: str,
    ) -> dict[str, Any]:
        job = IngestJob(
            job_id=str(uuid4()),
            path=path,
            recursive=recursive,
            total_files=0,
            collection_name=collection_name,
        )
        with self._lock:
            self._jobs[job.job_id] = job
            if self._active_job_id is None:
                self._active_job_id = job.job_id
                started = True
            else:
                self._queue.append(job.job_id)
                started = False

        if started:
            self._spawn(job.job_id)
            message = "Ingest job started. Poll get_ingest_status(job_id) for progress."
        else:
            message = "Another ingest job is running; this job was queued and starts automatically."
        # Explicit status last so the live job status in the snapshot (which may
        # already be 'running') does not overwrite the started/queued signal.
        return {**self.snapshot(job), "message": message, "status": "started" if started else "queued"}

    def status(self, job_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            target_job_id = job_id or self._active_job_id
            if target_job_id is None:
                known_jobs = sorted(self._jobs.values(), key=lambda job: job.started_at, reverse=True)
                if not known_jobs:
                    return {"status": "idle", "message": "No ingest job has been started yet."}
                return {"status": "idle", "latest_job": self.snapshot(known_jobs[0])}
            job = self._jobs.get(target_job_id)
            if job is None:
                raise ValueError(f"Unknown ingest job: {target_job_id}")
            return self.snapshot(job)

    def get_job(self, job_id: str) -> IngestJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job: IngestJob) -> dict[str, Any]:
        if job.total_files == 0:
            progress = 100.0 if job.status in {"completed", "failed"} else 0.0
        else:
            progress = round((job.processed_files / job.total_files) * 100, 1)
        return {
            "job_id": job.job_id,
            "status": job.status,
            "path": job.path,
            "recursive": job.recursive,
            "collection_name": job.collection_name,
            "job_type": job.job_type,
            "total_files": job.total_files,
            "processed_files": job.processed_files,
            "succeeded_files": job.succeeded_files,
            "failed_files": job.failed_files,
            "skipped_files": job.skipped_files,
            "skipped_by_type": job.skipped_by_type,
            "documents_uploaded": job.documents_uploaded,
            "chunks_uploaded": job.chunks_uploaded,
            "current_file": job.current_file,
            "progress_percent": progress,
            "error": job.error,
            "recent_errors": job.recent_errors[-10:],
            "log_stamp": job.log_stamp,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        }

    def trace_log_path(self, job_id: str) -> Path:
        return JOB_LOG_DIR / f"ingest-{job_id}.log"

    def collection_log_path(self, collection_name: str, stamp: str) -> Path:
        store = self.collections.get_store(collection_name)
        return store.collection_dir / f"ingest_{stamp}.log"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spawn(self, job_id: str) -> None:
        thread = threading.Thread(
            target=self._run,
            args=(job_id,),
            daemon=True,
            name=f"filechatter-ingest-{job_id[:8]}",
        )
        thread.start()

    def _start_next_queued(self) -> None:
        with self._lock:
            if not self._queue:
                self._active_job_id = None
                return
            next_job_id = self._queue.popleft()
            self._active_job_id = next_job_id
        self._spawn(next_job_id)

    def _append_collection_log(self, collection_name: str, stamp: str, message: str) -> None:
        try:
            path = self.collection_log_path(collection_name, stamp)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(message.rstrip() + "\n")
        except OSError:
            logger.debug("Could not write collection ingest log", exc_info=True)

    def _run(self, job_id: str) -> None:
        job = self._jobs[job_id]
        with self._lock:
            job.status = "running"

        log_path = self.trace_log_path(job_id)
        start_time = time()
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(f"JOB STARTED: {job_id} | discovering files... | collection={job.collection_name} | ts={start_time}\n")

        try:
            files = discover_documents(Path(job.path), recursive=job.recursive)
            with self._lock:
                job.total_files = len(files)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"DISCOVERED: {len(files)} files\n")

            explicit_entry = None
            if job.collection_name != "all":
                explicit_entry = self.collections.ensure(job.collection_name)

            for index, file_path in enumerate(files):
                with self._lock:
                    job.current_file = str(file_path)

                try:
                    # Extension filter for an explicit target collection.
                    if explicit_entry is not None and not explicit_entry.allows_extension(file_path.suffix):
                        with self._lock:
                            job.skipped_by_type += 1
                            job.processed_files += 1
                        with open(log_path, "a", encoding="utf-8") as fh:
                            fh.write(f"SKIP (type not allowed in '{explicit_entry.name}'): {file_path}\n")
                        continue

                    loaded_document = load_document(file_path)
                    target_collection = job.collection_name
                    if target_collection == "all":
                        target_collection = str(
                            loaded_document.metadata.get("collection_name", "documentation")
                        ).strip().lower()
                    else:
                        loaded_document.metadata["collection_name"] = target_collection
                        loaded_document.metadata["content_type"] = target_collection

                    if not loaded_document.content.strip():
                        with self._lock:
                            job.skipped_files += 1
                            job.processed_files += 1
                        with open(log_path, "a", encoding="utf-8") as fh:
                            fh.write(f"SKIP (empty): {file_path}\n")
                        continue

                    target_store = self.collections.get_store(target_collection)
                    file_start = time()
                    with self.collections.write_lock(target_collection):
                        result = target_store.add_documents(
                            [loaded_document.content], [loaded_document.metadata]
                        )
                        self.collections.touch(target_collection)
                    elapsed = time() - file_start
                    with self._lock:
                        job.succeeded_files += 1
                        job.processed_files += 1
                        job.documents_uploaded += result["documents_uploaded"]
                        job.chunks_uploaded += result["chunks_uploaded"]
                    message = (
                        f"OK: {file_path} -> {result['documents_uploaded']} docs, "
                        f"{result['chunks_uploaded']} chunks ({elapsed:.2f}s)"
                    )
                    with open(log_path, "a", encoding="utf-8") as fh:
                        fh.write(f"[{index + 1}/{len(files)}] {message}\n")
                    self._append_collection_log(target_collection, job.log_stamp, message)

                except Exception as exc:
                    error_message = f"{file_path}: {exc}\n{traceback.format_exc()}"
                    with self._lock:
                        job.failed_files += 1
                        job.processed_files += 1
                        job.recent_errors.append(error_message)
                    with open(log_path, "a", encoding="utf-8") as fh:
                        fh.write(f"ERROR: {error_message}\n")

            with self._lock:
                job.status = "completed"
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(
                    f"JOB COMPLETED: {job_id} | ok={job.succeeded_files} failed={job.failed_files} "
                    f"skipped={job.skipped_files} skipped_by_type={job.skipped_by_type} "
                    f"| elapsed={time() - start_time:.1f}s\n"
                )

        except Exception as job_exc:
            with self._lock:
                job.status = "failed"
                job.error = str(job_exc)
                job.recent_errors.append(f"Job crashed: {job_exc}\n{traceback.format_exc()}")
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"JOB CRASHED: {job_exc}\n{traceback.format_exc()}\n")

        finally:
            with self._lock:
                job.current_file = None
                job.finished_at = time()
            self._start_next_queued()

