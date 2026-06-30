"""MCP server exposing Filechatter retrieval tools to LM Studio."""
from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from pathlib import Path
import logging
import traceback
import threading
from datetime import datetime
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
        "For large folders, prefer start_ingest_directory and then poll get_ingest_status instead of using a long-running ingest call."
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

COLLECTION_NAMES = ("documentation", "code")
stores = {collection_name: RagStore(collection_name) for collection_name in COLLECTION_NAMES}
for collection_store in stores.values():
    atexit.register(collection_store.close)

_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",  # Hyphen
        "\u2011": "-",  # Non-breaking hyphen
        "\u2012": "-",  # Figure dash
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
        "\u2015": "-",  # Horizontal bar
        "\u2212": "-",  # Minus sign
    }
)


def _normalize_input_path(path: str) -> Path:
    # Normalize common Unicode dash variants copied from UIs/export tools.
    return Path(path.translate(_DASH_TRANSLATION)).expanduser().resolve()


def _normalize_collection_name(collection_name: str | None) -> str:
    if not collection_name:
        return "all"
    normalized = collection_name.strip().lower()
    if normalized in {"all", "*"}:
        return "all"
    return normalized


def _get_or_create_store(collection_name: str) -> RagStore:
    normalized = _normalize_collection_name(collection_name)
    if normalized == "all":
        raise ValueError("'all' does not map to a single store")

    store = stores.get(normalized)
    if store is None:
        store = RagStore(normalized)
        stores[normalized] = store
        atexit.register(store.close)
    return store


def _selected_stores(collection_name: str) -> list[RagStore]:
    normalized = _normalize_collection_name(collection_name)
    if normalized == "all":
        return list(stores.values())
    return [_get_or_create_store(normalized)]


def _store_for_metadata(metadata: dict[str, Any]) -> RagStore:
    collection_name = metadata.get("collection_name") or metadata.get("content_type") or metadata.get("database_name")
    if isinstance(collection_name, str) and collection_name.strip():
        return _get_or_create_store(collection_name.strip().lower())
    return _get_or_create_store("code") if str(metadata.get("extension", "")).lower() in {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".sh", ".ps1", ".json", ".yaml", ".yml", ".toml"} else _get_or_create_store("documentation")


def _job_log_path(collection_name: str, stamp: str) -> Path:
    return _get_or_create_store(collection_name).collection_dir / f"ingest_{stamp}.log"


def _append_collection_log(collection_name: str, stamp: str, message: str) -> None:
    path = _job_log_path(collection_name, stamp)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


JOB_LOG_DIR = Path(__file__).resolve().parent / "logs"
JOB_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _job_trace_log_path(job_id: str) -> Path:
    return JOB_LOG_DIR / f"ingest-{job_id}.log"

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

# MCP tool timeout safety: process files in time-boxed batches
# LM Studio likely has 60-90s timeout on tool calls, so we stop at 45s to be safe
MAX_PROCESSING_TIME_PER_BATCH = 45  # seconds


@dataclass(slots=True)
class IngestJob:
    job_id: str
    path: str
    recursive: bool
    total_files: int
    collection_name: str = "all"
    log_stamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
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
        "collection_name": job.collection_name,
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
        "log_stamp": job.log_stamp,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


def _set_active_job(job_id: str | None) -> None:
    global active_job_id
    active_job_id = job_id


def _run_ingest_job(job_id: str, paths: list[Path], collection_name: str) -> None:
    global _active_log_path
    
    job = jobs[job_id]
    with job_lock:
        job.status = "running"

    log_path = _job_trace_log_path(job_id)
    
    # Register this log file with signal handler
    with _active_log_lock:
        _active_log_path = log_path
    
    start_time = time()
    
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"JOB STARTED: {job_id} | {len(paths)} files to ingest | collection={collection_name} | timestamp={start_time}\n")
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
                target_collection = job.collection_name
                if target_collection == "all":
                    target_collection = str(loaded_document.metadata.get("collection_name", "documentation")).strip().lower()
                else:
                    loaded_document.metadata["collection_name"] = target_collection
                    loaded_document.metadata["content_type"] = target_collection
                target_store = _get_or_create_store(target_collection)
                
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
                        result = target_store.add_documents([content], [loaded_document.metadata])
                        elapsed = time() - file_start_time
                        with job_lock:
                            job.succeeded_files += result["documents_uploaded"]
                            job.documents_uploaded += result["documents_uploaded"]
                            job.chunks_uploaded += result["chunks_uploaded"]
                        with open(log_path, "a", encoding="utf-8") as fh:
                            fh.write(f"OK: {file_path} -> {result['documents_uploaded']} docs, {result['chunks_uploaded']} chunks ({elapsed:.2f}s)\n")
                            fh.flush()
                        _append_collection_log(target_collection, job.log_stamp, f"OK: {file_path} -> {result['documents_uploaded']} docs, {result['chunks_uploaded']} chunks ({elapsed:.2f}s)")
                    except Exception as store_exc:
                        tb = traceback.format_exc()
                        err_msg = f"{file_path}: store.add_documents() failed: {store_exc}\n{tb}"
                        with job_lock:
                            job.failed_files += 1
                            job.recent_errors.append(err_msg)
                        with open(log_path, "a", encoding="utf-8") as fh:
                            fh.write(f"ERROR (store): {err_msg}\n")
                        _append_collection_log(target_collection, job.log_stamp, f"ERROR (store): {err_msg}")
                        # Try to recover by reloading the index
                        try:
                            target_store.reload()
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
    documents_by_collection: dict[str, list[str]] = {}
    metadata_by_collection: dict[str, list[dict[str, Any]]] = {}

    for file_path in paths:
        loaded_document = load_document(file_path)
        content = loaded_document.content
        if not content.strip():
            continue

        collection_name = str(loaded_document.metadata.get("collection_name", "documentation")).lower()
        _get_or_create_store(collection_name)
        documents_by_collection.setdefault(collection_name, []).append(content)
        metadata_by_collection.setdefault(collection_name, []).append(loaded_document.metadata)

    aggregated = {"documents_uploaded": 0, "chunks_uploaded": 0, "total_chunks": 0, "per_collection": {}}
    for collection_name, documents in documents_by_collection.items():
        if not documents:
            continue
        result = stores[collection_name].add_documents(documents, metadata_by_collection[collection_name])
        aggregated["documents_uploaded"] += result["documents_uploaded"]
        aggregated["chunks_uploaded"] += result["chunks_uploaded"]
        aggregated["per_collection"][collection_name] = result

    aggregated["total_chunks"] = sum(store.chunk_count for store in stores.values())
    return aggregated


@mcp.tool()
def search_documents(query: str, collection_name: str = "all") -> list[dict[str, Any]]:
    """Search the local RAG index and return the most relevant chunks."""
    normalized_collection = _normalize_collection_name(collection_name)
    results = [
        result
        for selected_store in _selected_stores(normalized_collection)
        for result in selected_store.search(query)
    ]
    results.sort(key=lambda result: (-result.score, result.collection_name, result.source, result.chunk_id))
    return [
        {
            "chunk_id": result.chunk_id,
            "source": result.source,
            "content": result.content,
            "metadata": result.metadata,
            "score": result.score,
            "rank": result.rank,
            "collection_name": result.collection_name,
        }
        for result in results
    ]


@mcp.tool()
def list_sources(collection_name: str = "all") -> dict[str, Any]:
    """List indexed sources and chunk counts in the local store."""
    normalized_collection = _normalize_collection_name(collection_name)
    selected_stores = _selected_stores(normalized_collection)
    sources = []
    for selected_store in selected_stores:
        sources.extend(selected_store.list_sources())
    return {
        "collection_name": normalized_collection,
        "total_documents": len(sources),
        "total_chunks": sum(selected_store.chunk_count for selected_store in selected_stores),
        "sources": sources,
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
def ingest_file(path: str, collection_name: str = "all") -> dict[str, int]:
    """Index a single supported file into the local store."""
    file_path = _normalize_input_path(path)
    if not file_path.is_file():
        raise ValueError(f"File not found: {file_path}")
    normalized_collection = _normalize_collection_name(collection_name)
    if normalized_collection == "all":
        return _ingest_paths([file_path])

    loaded_document = load_document(file_path)
    loaded_document.metadata["collection_name"] = normalized_collection
    loaded_document.metadata["content_type"] = normalized_collection
    if loaded_document.content.strip():
        return _get_or_create_store(normalized_collection).add_documents([loaded_document.content], [loaded_document.metadata])
    return {"documents_uploaded": 0, "chunks_uploaded": 0, "total_chunks": _get_or_create_store(normalized_collection).chunk_count}


@mcp.tool()
def start_ingest_directory(path: str, recursive: bool = True, collection_name: str = "all") -> dict[str, Any]:
    """Start indexing a large directory in the background and return a job id for progress polling.
    
    This tool uses batch processing to work around MCP timeout limits. If a job returns status "paused",
    call this tool again to resume processing the remaining files."""
    dir_path = _normalize_input_path(path)
    if not dir_path.is_dir():
        raise ValueError(f"Directory not found: {dir_path}")

    files = discover_documents(dir_path, recursive=recursive)
    job_id = str(uuid4())
    normalized_collection = _normalize_collection_name(collection_name)
    if normalized_collection != "all":
        _get_or_create_store(normalized_collection)

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
                    args=(active_job_id, files, active_job.collection_name),
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
            collection_name=normalized_collection,
        )
        jobs[job_id] = job
        _set_active_job(job_id)

    thread = threading.Thread(
        target=_run_ingest_job,
        args=(job_id, files, normalized_collection),
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
def get_ingest_log(job_id: str, tail_lines: int = 200, collection_name: str = "all") -> dict[str, Any]:
    """Return the tail of the ingest log file for a given job id."""
    normalized_collection = _normalize_collection_name(collection_name)
    if normalized_collection == "all":
        path = _job_trace_log_path(job_id)
    else:
        job = jobs.get(job_id)
        if job is None:
            raise ValueError(f"Unknown ingest job: {job_id}")
        path = _job_log_path(normalized_collection, job.log_stamp)
    if not path.exists():
        raise ValueError(f"No log found for job: {job_id}")
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    tail = lines[-tail_lines:] if tail_lines and len(lines) > tail_lines else lines
    return {"job_id": job_id, "log": "".join(tail)}


@mcp.tool()
def clear_index(confirm: bool = False, collection_name: str = "all") -> dict[str, Any]:
    """Clear the local index. Set confirm=true to actually delete all indexed chunks."""
    normalized_collection = _normalize_collection_name(collection_name)
    if not confirm:
        return {
            "status": "cancelled",
            "message": "Pass confirm=true to clear the index.",
            "collection_name": normalized_collection,
            "total_chunks": sum(store.chunk_count for store in _selected_stores(normalized_collection)),
        }

    for selected_store in _selected_stores(normalized_collection):
        selected_store.clear()
    return {
        "status": "success",
        "message": "Index cleared",
        "collection_name": normalized_collection,
        "total_chunks": sum(store.chunk_count for store in _selected_stores(normalized_collection)),
    }


if __name__ == "__main__":
    mcp.run()
