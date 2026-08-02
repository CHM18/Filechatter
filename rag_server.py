"""Filechatter server: RAG API, tool execution endpoint, and web frontend.

This process is the single owner of all collection data (SQLite + FAISS).
MCP hosts reach it through the thin proxy in rag_mcp_server.py, the browser
frontend through the REST API, so no other process ever opens the data files.
"""
import json
import uuid
import threading
import time
import re
from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import requests
import logging
from langchain.schema import Document

import chat_agent
import config
import permissions
import runtime
import tool_registry
from collections_manager import CollectionLockedError
from llm_providers import ProviderError, build_provider
from memory_store import MEMORY_CATEGORIES, validate_memory
from rag_store import SearchResult
from langchain_support import LMStudioChatClient, build_prompt, format_context, parse_model_output

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
_MODEL_CACHE_TTL_SECONDS = 60
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str], tuple[float, list[str]]] = {}
MAX_MEMORY_FACTS_PER_TURN = 3
MAX_MEMORY_CONTEXT_CHARS = 800
SENSITIVE_MEMORY_RE = re.compile(
    r"\b(?:password|passphrase|api[ _-]?key|access[ _-]?token|secret|ssn|social security|credit card)\b",
    re.IGNORECASE,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        runtime.collections().close_all()


# Initialize FastAPI app
app = FastAPI(title="Filechatter RAG Server", version="3.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_ui_assets(request, call_next):
    """Serve UI assets with revalidation so a browser never runs a stale mix of
    HTML/JS/CSS after an update (which can leave buttons wired to elements that
    no longer exist). ETag/Last-Modified still yield cheap 304s when unchanged."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/ui"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _search_results(question: str, collection_name: str) -> list[SearchResult]:
    manager = runtime.collections()
    results = [
        result
        for store in manager.selected_stores(collection_name)
        for result in store.search(question)
    ]
    return sorted(results, key=lambda result: (-result.score, result.collection_name, result.source, result.chunk_id))


def _list_sources(collection_name: str) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for store in runtime.collections().selected_stores(collection_name):
        sources.extend(store.list_sources())
    return sources


def _total_chunks(collection_name: str) -> int:
    return sum(store.chunk_count for store in runtime.collections().selected_stores(collection_name))


def _cached_model_list(llm_settings: dict[str, Any], force: bool = False) -> list[str]:
    provider = build_provider(llm_settings)
    cache_key = (
        str(getattr(provider, "base_url", llm_settings.get("base_url", ""))),
        str(getattr(provider, "api_key", llm_settings.get("api_key", "")) or ""),
    )
    now = time.monotonic()
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if not force and cached and (now - cached[0]) < _MODEL_CACHE_TTL_SECONDS:
            return list(cached[1])

    models = provider.list_models()
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[cache_key] = (now, list(models))
    return models


lm_client = LMStudioChatClient(
    base_url=config.LM_STUDIO_BASE_URL,
    model=config.LM_STUDIO_MODEL,
    temperature=config.TEMPERATURE,
    timeout=config.LM_STUDIO_TIMEOUT,
)

# Configuration
LM_STUDIO_BASE_URL = config.LM_STUDIO_BASE_URL
LM_STUDIO_MODEL = config.LM_STUDIO_MODEL


class QueryRequest(BaseModel):
    question: str
    include_context: bool = True
    collection_name: str = "all"


class DocumentUpload(BaseModel):
    documents: list[str]
    metadata: list[dict[str, Any]] | None = None
    collection_name: str = "all"


class RetrievedChunk(BaseModel):
    chunk_id: int
    source: str
    content: str
    metadata: dict[str, Any]
    score: float
    rank: int
    collection_name: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    context: list[RetrievedChunk] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)


class ChatLangChainResponse(BaseModel):
    question: str
    prompt: str
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    context: list[RetrievedChunk] = Field(default_factory=list)


class SearchResponse(BaseModel):
    question: str
    results: list[RetrievedChunk]


class CollectionCreateRequest(BaseModel):
    name: str
    description: str = ""
    allowed_extensions: list[str] = Field(default_factory=list)


class CollectionUpdateRequest(BaseModel):
    description: str | None = None
    allowed_extensions: list[str] | None = None


class ToolCallRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
    client: str = permissions.CLIENT_API


class SettingsUpdateRequest(BaseModel):
    changes: dict[str, Any]


class PermissionsUpdateRequest(BaseModel):
    permissions: dict[str, Any]


class ChatStreamRequest(BaseModel):
    session_id: str | None = None
    message: str | None = None
    read_collections: list[str] = Field(default_factory=list)
    write_collections: list[str] = Field(default_factory=list)
    decisions: dict[str, Any] = Field(default_factory=dict)


class ChatCancelRequest(BaseModel):
    session_id: str


class MemoryCreateRequest(BaseModel):
    category: str
    fact: str


class MemoryUpdateRequest(BaseModel):
    category: str | None = None
    fact: str | None = None
    enabled: bool | None = None


def _serialize_result(result: SearchResult) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=result.chunk_id,
        source=result.source,
        content=result.content,
        metadata=result.metadata,
        score=result.score,
        rank=result.rank,
        collection_name=result.collection_name,
    )


@app.get("/health")
async def health_check():
    """Check server health and connectivity"""
    try:
        _cached_model_list(runtime.settings().get()["llm"])
        lm_studio_ok = True
    except ProviderError as e:
        lm_studio_ok = False
        logger.warning(f"LM Studio connection issue: {e}")

    manager = runtime.collections()
    return {
        "status": "ok",
        "lm_studio_connected": lm_studio_ok,
        "documents_count": len(_list_sources("all")),
        "chunks_count": _total_chunks("all"),
        "collections_count": len(manager.list_entries()),
        "data_dir": config.DATA_DIR,
    }


# ----------------------------------------------------------------------
# Collections API (M2)
# ----------------------------------------------------------------------


@app.get("/file-types")
async def file_types():
    """File-type categories with per-extension availability (for the UI selector)."""
    from document_loader import file_type_catalog

    return {"categories": file_type_catalog()}


@app.get("/collections")
async def list_collections():
    """Collection metadata from collections.json merged with live counts."""
    try:
        return {"collections": runtime.collections().describe_all()}
    except Exception as e:
        logger.error(f"List collections error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections", status_code=201)
async def create_collection(request: CollectionCreateRequest):
    manager = runtime.collections()
    try:
        entry = manager.create(
            name=request.name,
            description=request.description,
            allowed_extensions=request.allowed_extensions,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "success", "collection": manager.describe(entry.name)}


@app.patch("/collections/{name}")
async def update_collection(name: str, request: CollectionUpdateRequest):
    manager = runtime.collections()
    try:
        manager.update_entry(
            name,
            description=request.description,
            allowed_extensions=request.allowed_extensions,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown collection: {name}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "success", "collection": manager.describe(name)}


@app.delete("/collections/{name}")
async def delete_collection(name: str, confirm: bool = False):
    manager = runtime.collections()
    if not confirm:
        try:
            described = manager.describe(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown collection: {name}")
        return {
            "status": "cancelled",
            "message": "Pass confirm=true to delete this collection and all its data.",
            "collection": described,
        }
    try:
        result = manager.delete(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown collection: {name}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except CollectionLockedError as e:
        # 409 Conflict: the collection exists but its files are locked.
        raise HTTPException(status_code=409, detail=str(e))
    return {"status": "success", **result}


# ----------------------------------------------------------------------
# Tool execution API (M1) - used by the MCP proxy and (later) web chat
# ----------------------------------------------------------------------


@app.get("/tools")
async def list_tools(visible: bool = False):
    """List tools with access classification, parameter schemas, and the
    currently resolved permission. Pass visible=true to exclude denied tools
    (the MCP proxy uses this to decide which tools to advertise)."""
    perms = runtime.settings().get()["permissions"]
    tools = []
    for spec in tool_registry.list_specs():
        decision = permissions.resolve(spec.name, perms)
        if visible and decision == permissions.DENY:
            continue
        tools.append({**spec.to_dict(), "permission": decision})
    return {"tools": tools}


@app.post("/tools/{tool_name}")
async def execute_tool(tool_name: str, request: ToolCallRequest):
    # Resolve permissions before running. Unknown tools -> 404.
    try:
        tool_registry.get_spec(tool_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")
    try:
        permissions.enforce(tool_name, client=request.client)
    except permissions.PermissionDenied as e:
        raise HTTPException(status_code=403, detail=str(e))

    try:
        result = tool_registry.execute(tool_name, request.arguments)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Tool '{tool_name}' failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "tool": tool_name, "result": result}


# ----------------------------------------------------------------------
# Settings API
# ----------------------------------------------------------------------


@app.get("/settings")
async def get_settings():
    return runtime.settings().public_view()


@app.put("/settings")
async def update_settings(request: SettingsUpdateRequest):
    try:
        runtime.settings().update(request.changes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return runtime.settings().public_view()


@app.put("/settings/permissions")
async def set_permissions(request: PermissionsUpdateRequest):
    """Replace the permissions block wholesale (lets the UI clear per-tool
    overrides, which a deep-merge update cannot do)."""
    try:
        runtime.settings().set_permissions(request.permissions)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return runtime.settings().public_view()


# ----------------------------------------------------------------------
# Local long-term memory API
# ----------------------------------------------------------------------


@app.get("/memories")
async def list_memories():
    return {"memories": [memory.to_dict() for memory in runtime.memories().list()]}


@app.post("/memories", status_code=201)
async def create_memory(request: MemoryCreateRequest):
    try:
        memory = runtime.memories().add(request.category, request.fact)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"memory": memory.to_dict()}


@app.patch("/memories/{memory_id}")
async def update_memory(memory_id: int, request: MemoryUpdateRequest):
    current = runtime.memories().get(memory_id)
    if current is None:
        raise HTTPException(status_code=404, detail="Memory not found.")
    try:
        memory = runtime.memories().update(
            memory_id,
            request.category if request.category is not None else current.category,
            request.fact if request.fact is not None else current.fact,
            request.enabled if request.enabled is not None else current.enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    assert memory is not None
    return {"memory": memory.to_dict()}


@app.delete("/memories/{memory_id}", status_code=204)
async def delete_memory(memory_id: int):
    if not runtime.memories().delete(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found.")


@app.delete("/memories")
async def clear_memories(confirm: bool = False):
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass confirm=true to clear all memories.")
    return {"deleted": runtime.memories().clear()}


# ----------------------------------------------------------------------
# LLM endpoint helpers (browser chat, OpenAI-compatible only)
# ----------------------------------------------------------------------


@app.get("/llm/models")
async def llm_models(force: bool = False):
    """List models advertised by the configured endpoint (for the UI picker)."""
    try:
        return {"models": _cached_model_list(runtime.settings().get()["llm"], force=force)}
    except ProviderError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/llm/test")
async def llm_test():
    """Test connectivity to the configured endpoint without raising."""
    provider = build_provider(runtime.settings().get()["llm"])
    try:
        models = provider.list_models()
        return {"ok": True, "model_count": len(models), "models": models[:50]}
    except ProviderError as e:
        return {"ok": False, "error": str(e)}


# ----------------------------------------------------------------------
# Built-in chat (SSE)
# ----------------------------------------------------------------------


def _memory_context(message: str | None) -> list[str]:
    if not message:
        return []
    facts: list[str] = []
    remaining = MAX_MEMORY_CONTEXT_CHARS
    for memory in runtime.memories().search(message):
        if len(memory.fact) > remaining:
            continue
        facts.append(memory.fact)
        remaining -= len(memory.fact)
    return facts


def _parse_extracted_memories(content: str) -> list[tuple[str, str]]:
    value = content.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    candidates = payload.get("memories", []) if isinstance(payload, dict) else []
    facts: list[tuple[str, str]] = []
    for candidate in candidates[:MAX_MEMORY_FACTS_PER_TURN]:
        if not isinstance(candidate, dict):
            continue
        category = candidate.get("category")
        fact = candidate.get("fact")
        if not isinstance(category, str) or not isinstance(fact, str):
            continue
        if category not in MEMORY_CATEGORIES or SENSITIVE_MEMORY_RE.search(fact):
            continue
        try:
            facts.append(validate_memory(category, fact))
        except ValueError:
            continue
    return facts


def _extract_memory_facts(provider: Any, user_message: str, assistant_message: str) -> None:
    """Extract only structured preference/workflow facts from a completed turn."""
    prompt = (
        "Extract durable user preferences or workflow facts stated directly by the user. "
        "Ignore tool activity, document content, factual claims from the assistant, and anything "
        "sensitive or personal. Return JSON only: {\"memories\":[{\"category\":\"preference\" "
        "or \"workflow\",\"fact\":\"concise fact\"}]}. Return an empty array when no fact qualifies.\n\n"
        f"User message:\n{user_message[:2000]}\n\n"
        f"Assistant response:\n{assistant_message[:4000]}"
    )
    response = ""
    try:
        for kind, value in provider.stream(
            [
                {"role": "system", "content": "You return valid JSON and nothing else."},
                {"role": "user", "content": prompt},
            ]
        ):
            if kind == "message" and isinstance(value, dict):
                response = str(value.get("content") or "")
    except ProviderError as exc:
        logger.info("Memory extraction skipped: %s", exc)
        return

    for category, fact in _parse_extracted_memories(response):
        try:
            runtime.memories().add(category, fact)
        except ValueError:
            # Duplicate and validation failures are expected for automatic capture.
            continue


@app.post("/chat/stream")
async def chat_stream(request: ChatStreamRequest):
    """Run or resume the agent loop, streaming events as Server-Sent Events.

    A new session is created when session_id is missing/unknown; the returned
    'session' event carries the id for follow-up requests (message turns and
    confirmation decisions)."""
    store = runtime.chat_sessions()
    session_id = request.session_id
    session = store.get(session_id) if session_id else None
    if session is None:
        session_id = uuid.uuid4().hex
        session = store.create(session_id, request.read_collections, request.write_collections)

    # Scope can change between turns (the user may re-pick databases).
    session["read_cols"] = request.read_collections
    session["write_cols"] = request.write_collections
    if request.message:
        session["messages"].append({"role": "user", "content": request.message})

    provider = build_provider(runtime.settings().get()["llm"])
    messages = session["messages"]
    read_cols = session["read_cols"]
    write_cols = session["write_cols"]
    decisions = request.decisions
    memory_facts = _memory_context(request.message)
    cancel_event = store.begin_stream(session_id)
    if cancel_event is None:
        raise HTTPException(status_code=404, detail="Unknown chat session.")

    def event_stream():
        try:
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id}, ensure_ascii=False)}\n\n"
            model_queue = runtime.model_queue()
            if model_queue.has_pending_work():
                yield (
                    f"data: {json.dumps({'type': 'queued', 'message': 'Waiting for local memory update.'}, ensure_ascii=False)}\n\n"
                )
            with model_queue.chat_turn():
                for event in chat_agent.run(
                    provider,
                    messages,
                    read_cols,
                    write_cols,
                    decisions,
                    memory_facts,
                    cancel_event=cancel_event,
                ):
                    if event.get("type") == "done" and request.message and event.get("content"):
                        model_queue.enqueue_extraction(
                            lambda user_message=request.message, assistant_message=event["content"]: _extract_memory_facts(
                                provider, user_message, assistant_message
                            )
                        )
                    yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        finally:
            store.end_stream(session_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat/cancel")
async def chat_cancel(request: ChatCancelRequest):
    """Request cancellation of an active streamed chat turn."""
    cancelled = runtime.chat_sessions().cancel(request.session_id)
    if cancelled:
        return {"status": "success", "session_id": request.session_id}
    return {
        "status": "idle",
        "session_id": request.session_id,
        "message": "No active generation for this session.",
    }


# ----------------------------------------------------------------------
# Retrieval and compatibility chat endpoints
# ----------------------------------------------------------------------


@app.post("/search", response_model=SearchResponse)
async def search(request: QueryRequest):
    """Retrieve relevant chunks without invoking the LLM."""
    results = _search_results(request.question, request.collection_name) if request.include_context else []
    return SearchResponse(
        question=request.question,
        results=[_serialize_result(result) for result in results],
    )


def _run_grounded_chat(request: QueryRequest) -> tuple[str, str, dict[str, Any], list[SearchResult]]:
    retrieved_results = _search_results(request.question, request.collection_name) if request.include_context else []
    retrieved_documents = [
        Document(
            page_content=result.content,
            metadata={
                **result.metadata,
                "source": result.source,
                "chunk_index": result.metadata.get("chunk_index", result.rank),
                "collection_name": result.collection_name,
            },
        )
        for result in retrieved_results
    ]
    context = format_context(retrieved_documents)
    prompt_text = build_prompt(request.question, context)

    answer_text = lm_client.request(prompt_text)
    try:
        parsed = parse_model_output(answer_text)
    except Exception as parse_exc:
        logger.warning("Failed to parse LM Studio output, falling back to raw text: %s", parse_exc)
        parsed = {"answer": answer_text.strip(), "sources": []}
    return prompt_text, answer_text, parsed, retrieved_results


@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest):
    """Compatibility chat endpoint using chunked retrieval plus LM Studio."""
    try:
        _, answer_text, parsed, retrieved_results = _run_grounded_chat(request)
        return ChatResponse(
            question=request.question,
            answer=parsed.get("answer", answer_text).strip(),
            sources=parsed.get("sources", []),
            context=[_serialize_result(result) for result in retrieved_results] if request.include_context else [],
        )

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to LM Studio. Ensure it's running on " + LM_STUDIO_BASE_URL
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat_langchain", response_model=ChatLangChainResponse)
async def chat_langchain(request: QueryRequest):
    """Chat endpoint that returns the prompt, structured sources, and grounded answer."""
    try:
        prompt_text, answer_text, parsed, retrieved_results = _run_grounded_chat(request)
        return ChatLangChainResponse(
            question=request.question,
            prompt=prompt_text,
            answer=parsed.get("answer", answer_text).strip(),
            sources=parsed.get("sources", []),
            context=[_serialize_result(result) for result in retrieved_results] if request.include_context else [],
        )

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to LM Studio. Ensure it's running on " + LM_STUDIO_BASE_URL
        )
    except Exception as e:
        logger.error(f"Chat langchain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_documents(request: DocumentUpload):
    """Upload documents to the persistent chunked store."""
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        manager = runtime.collections()
        collection_name = manager.normalize_name(request.collection_name)
        metadata = request.metadata or [{} for _ in request.documents]

        # Route each document into its target collection.
        routed_documents: dict[str, list[str]] = {}
        routed_metadata: dict[str, list[dict[str, Any]]] = {}
        for index, document in enumerate(request.documents):
            entry = dict(metadata[index] or {})
            if collection_name != "all":
                target_collection = collection_name
            else:
                target_collection = manager.normalize_name(
                    str(entry.get("collection_name") or entry.get("content_type") or "documentation")
                )
                if target_collection == "all":
                    target_collection = "documentation"
            entry["collection_name"] = target_collection
            entry["content_type"] = target_collection
            routed_documents.setdefault(target_collection, []).append(document)
            routed_metadata.setdefault(target_collection, []).append(entry)

        result: dict[str, Any] = {"documents_uploaded": 0, "chunks_uploaded": 0, "per_collection": {}}
        for target_collection, documents in routed_documents.items():
            store = manager.get_store(target_collection)
            with manager.write_lock(target_collection):
                collection_result = store.add_documents(documents, routed_metadata[target_collection])
                manager.touch(target_collection)
            result["documents_uploaded"] += collection_result["documents_uploaded"]
            result["chunks_uploaded"] += collection_result["chunks_uploaded"]
            result["per_collection"][target_collection] = collection_result
        result["total_chunks"] = _total_chunks("all")

        logger.info(
            "Uploaded %s documents as %s chunks",
            result["documents_uploaded"],
            result["chunks_uploaded"],
        )
        return {"status": "success", **result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List indexed sources and storage summary."""
    try:
        return {
            "total_documents": len(_list_sources("all")),
            "total_chunks": _total_chunks("all"),
            "collection_name": "all",
            "sources": _list_sources("all"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chunks")
async def list_chunks(
    source: str | None = None,
    limit: int = 20,
    offset: int = 0,
    collection_name: str = "all",
):
    """List stored chunks for debugging indexed content."""
    try:
        safe_limit = max(1, min(limit, 200))
        safe_offset = max(0, offset)
        manager = runtime.collections()
        normalized_collection = manager.normalize_name(collection_name)
        chunks: list[dict[str, Any]] = []
        for store in manager.selected_stores(normalized_collection):
            chunks.extend(store.get_chunks(source=source, limit=safe_limit, offset=0))
        chunks.sort(key=lambda item: (item["collection_name"], item["source"], item["chunk_index"], item["id"]))
        chunks = chunks[safe_offset : safe_offset + safe_limit]
        return {
            "source": source,
            "collection_name": normalized_collection,
            "count": len(chunks),
            "limit": safe_limit,
            "offset": safe_offset,
            "chunks": chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
async def clear_documents(collection_name: str = "all"):
    """Clear all documents from the database"""
    try:
        manager = runtime.collections()
        normalized_collection = manager.normalize_name(collection_name)
        for store in manager.selected_stores(normalized_collection):
            with manager.write_lock(store.collection_name):
                store.clear()
                manager.touch(store.collection_name)
        logger.info("Database cleared")
        return {
            "status": "success",
            "collection_name": normalized_collection,
            "message": "Documents cleared"
        }

    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# Web frontend
# ----------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/ui/")


if STATIC_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn
    from config import RAG_SERVER_HOST, RAG_SERVER_PORT

    if RAG_SERVER_HOST not in {"localhost", "127.0.0.1", "::1"}:
        logger.warning(
            "RAG_HOST=%s exposes the server beyond this machine. "
            "The web UI can ingest and delete files - do NOT expose it on a network without protection.",
            RAG_SERVER_HOST,
        )

    logger.info(f"Starting RAG server on {RAG_SERVER_HOST}:{RAG_SERVER_PORT}")
    logger.info(f"LM Studio URL: {LM_STUDIO_BASE_URL}")
    logger.info(f"Web UI: http://{RAG_SERVER_HOST}:{RAG_SERVER_PORT}/ui/")
    uvicorn.run(
        app,
        host=RAG_SERVER_HOST,
        port=RAG_SERVER_PORT
    )
