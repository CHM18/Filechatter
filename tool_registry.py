"""Shared tool registry for Filechatter.

Single source of truth for the tools an LLM may call. Both the MCP proxy
(rag_mcp_server.py, via HTTP) and the future web-chat agent loop execute
tools through this registry, so read/write classification and (from M3 on)
permission enforcement live in exactly one place.

Every tool is described by a ToolSpec with a JSON-schema parameter
definition, so tool lists for MCP hosts and OpenAI/Anthropic-style
tool-calling can be generated from the same data.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import runtime
from document_loader import get_document_support_status, load_document

logger = logging.getLogger(__name__)

READ = "read"
WRITE = "write"

_DASH_TRANSLATION = str.maketrans(
    {
        "‐": "-",  # Hyphen
        "‑": "-",  # Non-breaking hyphen
        "‒": "-",  # Figure dash
        "–": "-",  # En dash
        "—": "-",  # Em dash
        "―": "-",  # Horizontal bar
        "−": "-",  # Minus sign
    }
)


def _normalize_input_path(path: str) -> Path:
    # Normalize common Unicode dash variants copied from UIs/export tools.
    return Path(path.translate(_DASH_TRANSLATION)).expanduser().resolve()


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    access: str  # READ or WRITE
    handler: Callable[..., Any]
    params_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "access": self.access,
            "params_schema": self.params_schema,
        }


TOOLS: dict[str, ToolSpec] = {}


def _register(spec: ToolSpec) -> None:
    TOOLS[spec.name] = spec


def list_specs() -> list[ToolSpec]:
    return list(TOOLS.values())


def get_spec(name: str) -> ToolSpec:
    spec = TOOLS.get(name)
    if spec is None:
        raise KeyError(f"Unknown tool: {name}")
    return spec


def execute(name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Run a registered tool. Raises KeyError for unknown tools and
    TypeError/ValueError for bad arguments - callers map these to HTTP errors."""
    spec = get_spec(name)
    return spec.handler(**(arguments or {}))


# ----------------------------------------------------------------------
# Read tools
# ----------------------------------------------------------------------


def search_documents(query: str, collection_name: str = "all") -> list[dict[str, Any]]:
    manager = runtime.collections()
    results = [
        result
        for store in manager.selected_stores(collection_name)
        for result in store.search(query)
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


def _compact_source(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source.get("source"),
        "source_name": source.get("source_name"),
        "chunk_count": source.get("chunk_count", 0),
        "first_indexed_at": source.get("first_indexed_at"),
        "last_indexed_at": source.get("last_indexed_at"),
        "file_changed_at": source.get("file_changed_at"),
        "last_ingested_at": source.get("last_ingested_at"),
    }


def _summarize_collection(store, max_sources: int) -> dict[str, Any]:
    manager = runtime.collections()
    sources = store.list_sources()
    entry = manager.get_entry(store.collection_name)
    preview = sorted(
        sources,
        key=lambda item: (-int(item.get("chunk_count") or 0), str(item.get("source_name") or item.get("source") or "")),
    )[:max_sources]
    last_indexed_at = max((item.get("last_indexed_at") or "" for item in sources), default="") or None
    last_ingested_at = max((item.get("last_ingested_at") or "" for item in sources), default="") or None
    document_count = len(sources)
    chunk_count = sum(int(item.get("chunk_count") or 0) for item in sources)
    summary_bits = [f"{document_count} document(s)", f"{chunk_count} chunk(s)"]
    if entry and entry.last_updated:
        summary_bits.append(f"last updated {entry.last_updated}")
    return {
        "collection_name": store.collection_name,
        "description": entry.description if entry else "",
        "document_count": document_count,
        "chunk_count": chunk_count,
        "last_updated": entry.last_updated if entry else None,
        "last_indexed_at": last_indexed_at,
        "last_ingested_at": last_ingested_at,
        "source_count": document_count,
        "sources": [_compact_source(item) for item in preview],
        "sources_omitted": max(0, document_count - len(preview)),
        "summary": "; ".join(summary_bits),
    }


def list_sources(collection_name: str = "all", max_sources: int = 10) -> dict[str, Any]:
    manager = runtime.collections()
    stores = manager.selected_stores(collection_name)
    limit = max(1, min(int(max_sources), 50))
    summaries = [_summarize_collection(store, limit) for store in stores]

    if isinstance(collection_name, (list, tuple, set)) or manager.normalize_name(collection_name) == "all" or len(summaries) > 1:
        display_name = ",".join(str(item) for item in collection_name) if isinstance(collection_name, (list, tuple, set)) else manager.normalize_name(collection_name)
        total_documents = sum(item["document_count"] for item in summaries)
        total_chunks = sum(item["chunk_count"] for item in summaries)
        return {
            "collection_name": display_name or "all",
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "collections": summaries,
            "summary": f"{total_documents} document(s) across {len(summaries)} collection(s); {total_chunks} chunk(s)",
        }

    summary = summaries[0] if summaries else {
        "collection_name": manager.normalize_name(collection_name),
        "description": "",
        "document_count": 0,
        "chunk_count": 0,
        "last_updated": None,
        "last_indexed_at": None,
        "last_ingested_at": None,
        "source_count": 0,
        "sources": [],
        "sources_omitted": 0,
        "summary": "0 document(s); 0 chunk(s)",
    }
    return {
        **summary,
        "total_documents": summary["document_count"],
        "total_chunks": summary["chunk_count"],
    }


def get_document_support() -> dict[str, Any]:
    support = get_document_support_status()
    unavailable = {
        extension: details for extension, details in support.items() if not details["available"]
    }
    return {
        "supported_extensions": sorted(support.keys()),
        "formats": support,
        "all_formats_ready": not unavailable,
        "unavailable_formats": unavailable,
    }


def get_ingest_status(job_id: str = "") -> dict[str, Any]:
    return runtime.jobs().status(job_id or None)


def get_ingest_log(job_id: str, tail_lines: int = 200, collection_name: str = "") -> dict[str, Any]:
    """Return ingest logs for a job.

    The job id is globally unique and is sufficient to resolve the canonical
    trace log. collection_name is treated only as an optional hint for
    compatibility with older clients.
    """
    job_manager = runtime.jobs()
    safe_tail = max(1, min(int(tail_lines), 5000))

    candidates: list[Path] = [job_manager.trace_log_path(job_id)]
    job = job_manager.get_job(job_id)

    if job is not None and job.collection_name != "all":
        candidates.append(job_manager.collection_log_path(job.collection_name, job.log_stamp))

    hint = (collection_name or "").strip()
    if hint and hint.lower() != "all" and job is not None:
        normalized_hint = runtime.collections().normalize_name(hint)
        hinted_path = job_manager.collection_log_path(normalized_hint, job.log_stamp)
        if hinted_path not in candidates:
            candidates.append(hinted_path)

    for path in candidates:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as fh:
            tail = deque(fh, maxlen=safe_tail)
        return {
            "job_id": job_id,
            "log": "".join(tail),
            "tail_lines": safe_tail,
            "resolved_log": str(path),
        }

    raise ValueError(f"No log found for job: {job_id}")


# ----------------------------------------------------------------------
# Write tools
# ----------------------------------------------------------------------


def ingest_file(path: str, collection_name: str = "all") -> dict[str, Any]:
    manager = runtime.collections()
    file_path = _normalize_input_path(path)
    if not file_path.is_file():
        raise ValueError(f"File not found: {file_path}")

    normalized = manager.normalize_name(collection_name)
    loaded_document = load_document(file_path)

    if normalized == "all":
        # Auto-route via metadata inferred by the loader (documentation/code).
        target_collection = str(loaded_document.metadata.get("collection_name", "documentation")).lower()
    else:
        entry = manager.ensure(normalized)
        if not entry.allows_extension(file_path.suffix):
            allowed = ", ".join(entry.allowed_extensions)
            raise ValueError(
                f"Collection '{entry.name}' does not accept '{file_path.suffix}' files "
                f"(allowed: {allowed})"
            )
        target_collection = normalized
        loaded_document.metadata["collection_name"] = target_collection
        loaded_document.metadata["content_type"] = target_collection

    store = manager.get_store(target_collection)
    if not loaded_document.content.strip():
        return {
            "documents_uploaded": 0,
            "chunks_uploaded": 0,
            "total_chunks": store.chunk_count,
            "collection_name": target_collection,
        }

    with manager.write_lock(target_collection):
        result = store.add_documents([loaded_document.content], [loaded_document.metadata])
        manager.touch(target_collection)
    return {**result, "collection_name": target_collection}


def start_ingest_directory(path: str, recursive: bool = True, collection_name: str = "all") -> dict[str, Any]:
    manager = runtime.collections()
    dir_path = _normalize_input_path(path)
    if not dir_path.is_dir():
        raise ValueError(f"Directory not found: {dir_path}")

    normalized = manager.normalize_name(collection_name)
    if normalized != "all":
        manager.ensure(normalized)

    return runtime.jobs().start_job(
        path=str(dir_path),
        recursive=recursive,
        collection_name=normalized,
    )


def clear_index(confirm: bool = False, collection_name: str = "all", source: str = "") -> dict[str, Any]:
    manager = runtime.collections()
    normalized = manager.normalize_name(collection_name)
    source_value = source.strip()

    if not confirm:
        stores = manager.selected_stores(normalized)
        total_chunks = sum(store.chunk_count for store in stores)
        target_chunks = total_chunks
        if source_value:
            source_candidates = {source_value}
            try:
                source_candidates.add(str(_normalize_input_path(source_value)))
            except OSError:
                pass
            target_chunks = 0
            for store in stores:
                for candidate in source_candidates:
                    target_chunks += store.chunk_count_for_source(candidate)
        return {
            "status": "cancelled",
            "message": "Pass confirm=true to clear the index.",
            "collection_name": normalized,
            "source": source_value or None,
            "target_chunks": target_chunks,
            "total_chunks": total_chunks,
        }

    stores = manager.selected_stores(normalized)
    deleted_chunks = 0
    cleared_sources: list[dict[str, Any]] = []

    if source_value:
        source_candidates = [source_value]
        try:
            normalized_source = str(_normalize_input_path(source_value))
            if normalized_source not in source_candidates:
                source_candidates.append(normalized_source)
        except OSError:
            pass

        for store in stores:
            for candidate in source_candidates:
                count = store.chunk_count_for_source(candidate)
                if count <= 0:
                    continue
                with manager.write_lock(store.collection_name):
                    result = store.clear_source(candidate)
                    manager.touch(store.collection_name)
                deleted_chunks += result["deleted_chunks"]
                cleared_sources.append(
                    {
                        "collection_name": store.collection_name,
                        "source": candidate,
                        "deleted_chunks": result["deleted_chunks"],
                    }
                )
                break
        message = "Document chunks cleared" if deleted_chunks > 0 else "No matching document chunks found"
    else:
        for store in stores:
            with manager.write_lock(store.collection_name):
                deleted_chunks += store.chunk_count
                store.clear()
                manager.touch(store.collection_name)
        message = "Index cleared"

    return {
        "status": "success",
        "message": message,
        "collection_name": normalized,
        "source": source_value or None,
        "deleted_chunks": deleted_chunks,
        "cleared_sources": cleared_sources,
        "total_chunks": sum(store.chunk_count for store in stores),
    }


# ----------------------------------------------------------------------
# Registry definitions
# ----------------------------------------------------------------------

_COLLECTION_PARAM = {
    "type": "string",
    "description": "Collection to target, or 'all' for every collection.",
    "default": "all",
}

_register(
    ToolSpec(
        name="search_documents",
        description="Search the local RAG index and return the most relevant chunks.",
        access=READ,
        handler=search_documents,
        params_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "collection_name": _COLLECTION_PARAM,
            },
            "required": ["query"],
        },
    )
)

_register(
    ToolSpec(
        name="list_sources",
        description="List indexed sources and chunk counts in the local store.",
        access=READ,
        handler=list_sources,
        params_schema={
            "type": "object",
            "properties": {
                "collection_name": _COLLECTION_PARAM,
                "max_sources": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": [],
        },
    )
)

_register(
    ToolSpec(
        name="get_document_support",
        description="Report which document formats are ingest-ready and which optional dependencies are missing.",
        access=READ,
        handler=get_document_support,
        params_schema={"type": "object", "properties": {}, "required": []},
    )
)

_register(
    ToolSpec(
        name="get_ingest_status",
        description="Return the progress of a background ingest job. Omit job_id to inspect the active job.",
        access=READ,
        handler=get_ingest_status,
        params_schema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Ingest job id.", "default": ""},
            },
            "required": [],
        },
    )
)

_register(
    ToolSpec(
        name="get_ingest_log",
        description="Return the tail of the ingest log file for a given job id.",
        access=READ,
        handler=get_ingest_log,
        params_schema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Ingest job id."},
                "tail_lines": {"type": "integer", "default": 200},
                "collection_name": {
                    "type": "string",
                    "description": "Optional compatibility hint. job_id is sufficient.",
                    "default": "",
                },
            },
            "required": ["job_id"],
        },
    )
)

_register(
    ToolSpec(
        name="ingest_file",
        description="Index a single supported file into the local store.",
        access=WRITE,
        handler=ingest_file,
        params_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to index."},
                "collection_name": _COLLECTION_PARAM,
            },
            "required": ["path"],
        },
    )
)

_register(
    ToolSpec(
        name="start_ingest_directory",
        description=(
            "Start indexing a directory in the background and return a job id for progress polling "
            "with get_ingest_status."
        ),
        access=WRITE,
        handler=start_ingest_directory,
        params_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to index."},
                "recursive": {"type": "boolean", "default": True},
                "collection_name": _COLLECTION_PARAM,
            },
            "required": ["path"],
        },
    )
)

_register(
    ToolSpec(
        name="clear_index",
        description=(
            "Clear the local index. Set confirm=true to delete. Provide source to clear one document; "
            "omit source to clear an entire collection (or all collections)."
        ),
        access=WRITE,
        handler=clear_index,
        params_schema={
            "type": "object",
            "properties": {
                "confirm": {"type": "boolean", "default": False},
                "collection_name": _COLLECTION_PARAM,
                "source": {"type": "string", "default": ""},
            },
            "required": [],
        },
    )
)
