"""MCP server exposing Filechatter retrieval tools to LM Studio."""
from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from pathlib import Path
import logging
import traceback
import threading
from time import time
from typing import Any
from uuid import uuid4
import signal
import sys

from mcp.server.fastmcp import FastMCP

from document_loader import discover_documents, get_document_support_status, load_document
from rag_store import RagStore

# Optional memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


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

# Per-job ingest logs
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Global reference to active job log for signal handlers
_active_log_path: Path | None = None
_active_log_lock = threading.Lock()

def _signal_handler(signum, frame):
    """Handle termination signals and log them"""
    global _active_log_path
    with _active_log_lock:
        if _active_log_path and _active_log_path.exists():
            try:
                with open(_active_log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"SIGNAL RECEIVED: {signal.Signals(signum).name} (code {signum}) at {time()}\n")
                    fh.flush()
            except Exception:
                pass

# Register signal handlers
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, _signal_handler)
if hasattr(signal, 'SIGINT'):
    signal.signal(signal.SIGINT, _signal_handler)

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# MCP tool timeout safety: process files in time-boxed batches
# LM Studio likely has 60-90s timeout on tool calls, so we stop at 45s to be safe
MAX_PROCESSING_TIME_PER_BATCH = 45  # seconds

def _job_log_path(job_id: str) -> Path:
    return LOG_DIR / f"ingest-{job_id}.log"


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
    remaining_file_count: int = 0  # Files not yet processed (for resume)


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
        # Full tracebacks and errors for detailed debugging
        "recent_errors_full": job.recent_errors,
        "error_full": job.error,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


def _set_active_job(job_id: str | None) -> None:
    global active_job_id
    active_job_id = job_id


def _run_ingest_job(job_id: str, paths: list[Path]) -> None:
    global _active_log_path
    
    job = jobs[job_id]
    with job_lock:
        job.status = "running"

    log_path = _job_log_path(job_id)
    
    # Register this log file with signal handler
    with _active_log_lock:
        _active_log_path = log_path
    
    start_time = time()
    
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"JOB STARTED: {job_id} | {len(paths)} files to ingest | timestamp={start_time}\n")
        fh.flush()
    
    try:
        for idx, file_path in enumerate(paths):
            # Check if we're running out of time (MCP timeout safety)
            elapsed_total = time() - start_time
            if elapsed_total > MAX_PROCESSING_TIME_PER_BATCH:
                remaining_count = len(paths) - idx
                with job_lock:
                    job.remaining_file_count = remaining_count
                    job.status = "paused"
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"[TIMEOUT] Batch time limit ({MAX_PROCESSING_TIME_PER_BATCH}s) reached after {elapsed_total:.1f}s\n")
                    fh.write(f"[RESUME] {remaining_count} files remaining. Call start_ingest_directory() again to continue.\n")
                    fh.flush()
                break
            
            with job_lock:
                job.current_file = str(file_path)
            
            # Log progress checkpoint
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"[CHECKPOINT] File {idx+1}/{len(paths)} | elapsed={elapsed_total:.1f}s\n")
                fh.flush()
            
            # Log memory usage periodically
            if HAS_PSUTIL:
                try:
                    process = psutil.Process()
                    mem_info = process.memory_info()
                    with open(log_path, "a", encoding="utf-8") as fh:
                        fh.write(f"[MEM] RSS: {mem_info.rss / 1024 / 1024:.1f}MB | Processing: {file_path.name}\n")
                except Exception:
                    pass

            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"[LOAD_START] Loading: {file_path.name}\n")
                    fh.flush()
                
                loaded_document = load_document(file_path)
                
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"[LOAD_OK] Loaded {len(loaded_document.content)} chars\n")
                    fh.flush()
                
                content = loaded_document.content
                if not content.strip():
                    with job_lock:
                        job.skipped_files += 1
                    with open(log_path, "a", encoding="utf-8") as fh:
                        fh.write(f"SKIP: {file_path}\n")
                else:
                    try:
                        file_start_time = time()
                        result = store.add_documents([content], [loaded_document.metadata])
                        elapsed = time() - file_start_time
                        with job_lock:
                            job.succeeded_files += result["documents_uploaded"]
                            job.documents_uploaded += result["documents_uploaded"]
                            job.chunks_uploaded += result["chunks_uploaded"]
                        with open(log_path, "a", encoding="utf-8") as fh:
                            fh.write(f"OK: {file_path} -> {result['documents_uploaded']} docs, {result['chunks_uploaded']} chunks ({elapsed:.2f}s)\n")
                            fh.flush()
                    except Exception as store_exc:
                        tb = traceback.format_exc()
                        err_msg = f"{file_path}: store.add_documents() failed: {store_exc}\n{tb}"
                        with job_lock:
                            job.failed_files += 1
                            job.recent_errors.append(err_msg)
                        with open(log_path, "a", encoding="utf-8") as fh:
                            fh.write(f"ERROR (store): {err_msg}\n")
                        # Try to recover by reloading the index
                        try:
                            store.reload()
                            with open(log_path, "a", encoding="utf-8") as fh:
                                fh.write(f"INFO: Reloaded store index after error\n")
                        except Exception as reload_exc:
                            with open(log_path, "a", encoding="utf-8") as fh:
                                fh.write(f"ERROR: Failed to reload store: {reload_exc}\n")
            except Exception as exc:
                tb = traceback.format_exc()
                err_msg = f"{file_path}: {exc}\n{tb}"
                with job_lock:
                    job.failed_files += 1
                    job.recent_errors.append(err_msg)
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"ERROR: {err_msg}\n")
        
        # Job completed or paused
        elapsed_total = time() - start_time
        with job_lock:
            if job.status != "paused":  # Only set to completed if not already paused due to timeout
                job.status = "completed"
        
        if job.status == "completed":
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"JOB COMPLETED: {job_id} | Processed {job.succeeded_files} succeeded, {job.failed_files} failed, {job.skipped_files} skipped | elapsed={elapsed_total:.1f}s\n")
        else:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"JOB PAUSED: {job_id} | Progress: {job.succeeded_files} succeeded, {job.failed_files} failed, {job.skipped_files} skipped, {job.remaining_file_count} remaining | elapsed={elapsed_total:.1f}s\n")
    
    except Exception as job_exc:
        tb = traceback.format_exc()
        err_msg = f"Job {job_id} crashed: {job_exc}\n{tb}"
        with job_lock:
            job.status = "failed"
            job.error = str(job_exc)
            job.recent_errors.append(err_msg)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"JOB CRASHED: {err_msg}\n")
    
    finally:
        # Always write a final heartbeat to detect if the thread is still running
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"JOB ENDED: {job_id} | final_status={job.status} | timestamp={time()}\n")
            fh.flush()
        
        # Clear active log path
        with _active_log_lock:
            _active_log_path = None
        
        with job_lock:
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
    """Start indexing a large directory in the background and return a job id for progress polling.
    
    This tool uses batch processing to work around MCP timeout limits. If a job returns status "paused",
    call this tool again to resume processing the remaining files."""
    dir_path = Path(path).expanduser().resolve()
    if not dir_path.is_dir():
        raise ValueError(f"Directory not found: {dir_path}")

    files = discover_documents(dir_path, recursive=recursive)
    job_id = str(uuid4())

    with job_lock:
        # Check if we have an active paused job we can resume
        if active_job_id is not None:
            active_job = jobs.get(active_job_id)
            if active_job and active_job.status == "paused":
                # Resume the paused job with the same directory
                active_job.status = "running"
                active_job.remaining_file_count = 0
                thread = threading.Thread(
                    target=_run_ingest_job,
                    args=(active_job_id, files),
                    daemon=True,
                    name=f"filechatter-ingest-{active_job_id[:8]}-resume",
                )
                thread.start()
                return {
                    "status": "resumed",
                    "message": "Resuming paused ingest job. Poll get_ingest_status() for progress.",
                    **_job_snapshot(active_job),
                }
            elif active_job and active_job.status in {"queued", "running"}:
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
        "message": "Ingest job started. Poll get_ingest_status(job_id) for progress. If job status is 'paused', call this tool again to resume.",
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
def get_ingest_log(job_id: str, tail_lines: int = 200) -> dict[str, Any]:
    """Return the tail of the ingest log file for a given job id."""
    path = _job_log_path(job_id)
    if not path.exists():
        raise ValueError(f"No log found for job: {job_id}")
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    tail = lines[-tail_lines:] if tail_lines and len(lines) > tail_lines else lines
    return {"job_id": job_id, "log": "".join(tail)}


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
