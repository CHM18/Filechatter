"""MCP server exposing Filechatter retrieval tools to LM Studio."""
from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from pathlib import Path
import threading
from time import time
from typing import Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

from document_loader import discover_documents, get_document_support_status, load_document
from rag_store import RagStore


def _build_instructions() -> str:
    instructions = (
        "Use these tools to search a local document index and manage indexed sources. "
        "Prefer search_documents for grounded retrieval before answering questions. "
        "For large folders, prefer start_ingest_directory and then poll get_ingest_status instead of using the synchronous ingest_directory tool."
    )

    unavailable = [
        f"{extension}: {details['message']}"
        for extension, details in get_document_support_status().items()
        if not details["available"]
    ]
    if unavailable:
        instructions += " Reader readiness check: some formats are currently unavailable. " + " ".join(unavailable)

    return instructions


mcp = FastMCP(
    "Filechatter",
    instructions=_build_instructions(),
    json_response=True,
)

store = RagStore()
atexit.register(store.close)


@dataclass(slots=True)
class IngestJob:
    job_id: str
    path: str
    recursive: bool
    total_files: int
    status: str = "queued"
    processed_files: int = 0
    succeeded_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    documents_uploaded: int = 0
    chunks_uploaded: int = 0
    current_file: str | None = None
    error: str | None = None
    recent_errors: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time)
    finished_at: float | None = None


job_lock = threading.Lock()
jobs: dict[str, IngestJob] = {}
active_job_id: str | None = None


def _job_snapshot(job: IngestJob) -> dict[str, Any]:
    total_files = max(job.total_files, 1)
    progress = round((job.processed_files / total_files) * 100, 1)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "path": job.path,
        "recursive": job.recursive,
        "total_files": job.total_files,
        "processed_files": job.processed_files,
        "succeeded_files": job.succeeded_files,
        "failed_files": job.failed_files,
        "skipped_files": job.skipped_files,
        "documents_uploaded": job.documents_uploaded,
        "chunks_uploaded": job.chunks_uploaded,
        "current_file": job.current_file,
        "progress_percent": 100.0 if job.total_files == 0 else progress,
        "error": job.error,
        "recent_errors": job.recent_errors[-10:],
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


def _set_active_job(job_id: str | None) -> None:
    global active_job_id
    active_job_id = job_id


def _run_ingest_job(job_id: str, paths: list[Path]) -> None:
    job = jobs[job_id]
    with job_lock:
        job.status = "running"

    try:
        for file_path in paths:
            with job_lock:
                job.current_file = str(file_path)

            try:
                loaded_document = load_document(file_path)
                content = loaded_document.content
                if not content.strip():
                    with job_lock:
                        job.skipped_files += 1
                else:
                    result = store.add_documents([content], [loaded_document.metadata])
                    with job_lock:
                        job.succeeded_files += result["documents_uploaded"]
                        job.documents_uploaded += result["documents_uploaded"]
                        job.chunks_uploaded += result["chunks_uploaded"]
            except Exception as exc:
                with job_lock:
                    job.failed_files += 1
                    job.recent_errors.append(f"{file_path}: {exc}")
            finally:
                with job_lock:
                    job.processed_files += 1

        with job_lock:
            job.status = "completed"
            job.current_file = None
            job.finished_at = time()
            _set_active_job(None)
    except Exception as exc:
        with job_lock:
            job.status = "failed"
            job.error = str(exc)
            job.current_file = None
            job.finished_at = time()
            _set_active_job(None)


def _ingest_paths(paths: list[Path]) -> dict[str, int]:
    documents: list[str] = []
    metadata: list[dict[str, Any]] = []

    for file_path in paths:
        loaded_document = load_document(file_path)
        content = loaded_document.content
        if not content.strip():
            continue
        documents.append(content)
        metadata.append(loaded_document.metadata)

    if not documents:
        return {"documents_uploaded": 0, "chunks_uploaded": 0, "total_chunks": store.chunk_count}
    return store.add_documents(documents, metadata)


@mcp.tool()
def search_documents(query: str) -> list[dict[str, Any]]:
    """Search the local RAG index and return the most relevant chunks."""
    return [
        {
            "chunk_id": result.chunk_id,
            "source": result.source,
            "content": result.content,
            "metadata": result.metadata,
            "score": result.score,
            "rank": result.rank,
        }
        for result in store.search(query)
    ]


@mcp.tool()
def list_sources() -> dict[str, Any]:
    """List indexed sources and chunk counts in the local store."""
    return {
        "total_documents": len(store.list_sources()),
        "total_chunks": store.chunk_count,
        "sources": store.list_sources(),
    }


@mcp.tool()
def get_document_support() -> dict[str, Any]:
    """Report which document formats are currently ingest-ready and which optional dependencies are missing."""
    support = get_document_support_status()
    unavailable = {
        extension: details
        for extension, details in support.items()
        if not details["available"]
    }
    return {
        "supported_extensions": sorted(support.keys()),
        "formats": support,
        "all_formats_ready": not unavailable,
        "unavailable_formats": unavailable,
    }


@mcp.tool()
def ingest_file(path: str) -> dict[str, int]:
    """Index a single supported file into the local store."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise ValueError(f"File not found: {file_path}")
    return _ingest_paths([file_path])


@mcp.tool()
def ingest_directory(path: str, recursive: bool = True) -> dict[str, int]:
    """Index all supported files from a directory into the local store synchronously."""
    dir_path = Path(path).expanduser().resolve()
    if not dir_path.is_dir():
        raise ValueError(f"Directory not found: {dir_path}")

    files = discover_documents(dir_path, recursive=recursive)
    return _ingest_paths(files)


@mcp.tool()
def start_ingest_directory(path: str, recursive: bool = True) -> dict[str, Any]:
    """Start indexing a large directory in the background and return a job id for progress polling."""
    dir_path = Path(path).expanduser().resolve()
    if not dir_path.is_dir():
        raise ValueError(f"Directory not found: {dir_path}")

    files = discover_documents(dir_path, recursive=recursive)
    job_id = str(uuid4())

    with job_lock:
        if active_job_id is not None:
            active_job = jobs.get(active_job_id)
            if active_job and active_job.status in {"queued", "running"}:
                return {
                    "status": "busy",
                    "message": "Another ingest job is already running.",
                    "active_job": _job_snapshot(active_job),
                }

        job = IngestJob(
            job_id=job_id,
            path=str(dir_path),
            recursive=recursive,
            total_files=len(files),
        )
        jobs[job_id] = job
        _set_active_job(job_id)

    thread = threading.Thread(
        target=_run_ingest_job,
        args=(job_id, files),
        daemon=True,
        name=f"filechatter-ingest-{job_id[:8]}",
    )
    thread.start()
    return {
        "status": "started",
        "message": "Ingest job started. Poll get_ingest_status(job_id) for progress.",
        **_job_snapshot(job),
    }


@mcp.tool()
def get_ingest_status(job_id: str = "") -> dict[str, Any]:
    """Return the progress of a background ingest job. Omit job_id to inspect the active job."""
    with job_lock:
        target_job_id = job_id if job_id else active_job_id
        if target_job_id is None:
            completed_jobs = sorted(
                jobs.values(),
                key=lambda job: job.started_at,
                reverse=True,
            )
            if not completed_jobs:
                return {"status": "idle", "message": "No ingest job has been started yet."}
            return {"status": "idle", "latest_job": _job_snapshot(completed_jobs[0])}

        job = jobs.get(target_job_id)
        if job is None:
            raise ValueError(f"Unknown ingest job: {target_job_id}")
        return _job_snapshot(job)


@mcp.tool()
def clear_index(confirm: bool = False) -> dict[str, Any]:
    """Clear the local index. Set confirm=true to actually delete all indexed chunks."""
    if not confirm:
        return {
            "status": "cancelled",
            "message": "Pass confirm=true to clear the index.",
            "total_chunks": store.chunk_count,
        }

    store.clear()
    return {"status": "success", "message": "Index cleared", "total_chunks": 0}


if __name__ == "__main__":
    mcp.run()
