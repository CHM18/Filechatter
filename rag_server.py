"""RAG server for Filechatter with persistent chunked retrieval."""
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
import logging
from langchain.schema import Document
import atexit

import config
from rag_store import RagStore, SearchResult
from langchain_support import LMStudioChatClient, build_prompt, format_context, parse_model_output

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION_NAMES = ("documentation", "code")
stores = {collection_name: RagStore(collection_name) for collection_name in COLLECTION_NAMES}


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        for selected_store in stores.values():
            selected_store.close()


# Initialize FastAPI app
app = FastAPI(title="Filechatter RAG Server", version="2.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        raise HTTPException(status_code=400, detail="'all' does not map to a single collection")

    store = stores.get(normalized)
    if store is None:
        store = RagStore(normalized)
        stores[normalized] = store
        atexit.register(store.close)
    return store


def _selected_stores(collection_name: str) -> list[RagStore]:
    if collection_name == "all":
        return list(stores.values())
    return [_get_or_create_store(collection_name)]


def _search_results(question: str, collection_name: str) -> list[SearchResult]:
    selected_stores = _selected_stores(collection_name)
    results = [result for selected_store in selected_stores for result in selected_store.search(question)]
    return sorted(results, key=lambda result: (-result.score, result.collection_name, result.source, result.chunk_id))


def _list_sources(collection_name: str) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for selected_store in _selected_stores(collection_name):
        sources.extend(selected_store.list_sources())
    return sources


def _total_chunks(collection_name: str) -> int:
    return sum(selected_store.chunk_count for selected_store in _selected_stores(collection_name))


for store in stores.values():
    atexit.register(store.close)
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


def _format_context(results: list[SearchResult]) -> str:
    return format_context(
        [
            Document(page_content=result.content, metadata={
                **result.metadata,
                "source": result.source,
                "chunk_index": result.metadata.get("chunk_index", result.rank),
            })
            for result in results
        ]
    )


@app.get("/health")
async def health_check():
    """Check server health and connectivity"""
    try:
        response = requests.get(
            f"{LM_STUDIO_BASE_URL}/v1/models",
            timeout=5
        )
        lm_studio_ok = response.status_code == 200
    except Exception as e:
        lm_studio_ok = False
        logger.warning(f"LM Studio connection issue: {e}")
    
    return {
        "status": "ok",
        "lm_studio_connected": lm_studio_ok,
        "documents_count": len(_list_sources("all")),
        "chunks_count": _total_chunks("all"),
        "data_dir": config.DATA_DIR,
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: QueryRequest):
    """Retrieve relevant chunks without invoking the LLM."""
    collection_name = _normalize_collection_name(request.collection_name)
    results = _search_results(request.question, collection_name) if request.include_context else []
    return SearchResponse(
        question=request.question,
        results=[_serialize_result(result) for result in results],
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest):
    """Compatibility chat endpoint using chunked retrieval plus LM Studio."""
    try:
        collection_name = _normalize_collection_name(request.collection_name)
        retrieved_results = _search_results(request.question, collection_name) if request.include_context else []
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
        parsed = {}
        try:
            parsed = parse_model_output(answer_text)
        except Exception as parse_exc:
            logger.warning("Failed to parse LM Studio output, falling back to raw text: %s", parse_exc)
            parsed = {"answer": answer_text.strip(), "sources": []}

        chat_context = [
            _serialize_result(result) for result in retrieved_results
        ] if request.include_context else []

        return ChatResponse(
            question=request.question,
            answer=parsed.get("answer", answer_text).strip(),
            sources=parsed.get("sources", []),
            context=chat_context,
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
        collection_name = _normalize_collection_name(request.collection_name)
        retrieved_results = _search_results(request.question, collection_name) if request.include_context else []
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
        parsed = {}
        try:
            parsed = parse_model_output(answer_text)
        except Exception as parse_exc:
            logger.warning("Failed to parse LM Studio output, falling back to raw text: %s", parse_exc)
            parsed = {"answer": answer_text.strip(), "sources": []}

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
        collection_name = _normalize_collection_name(request.collection_name)
        metadata = request.metadata or [{} for _ in request.documents]
        if collection_name != "all":
            for entry in metadata:
                entry["collection_name"] = collection_name
                entry["content_type"] = collection_name
            result = _get_or_create_store(collection_name).add_documents(request.documents, metadata)
        else:
            routed_documents: dict[str, list[str]] = {name: [] for name in stores}
            routed_metadata: dict[str, list[dict[str, Any]]] = {name: [] for name in stores}
            for index, document in enumerate(request.documents):
                entry = dict(metadata[index] or {})
                target_collection = str(entry.get("collection_name") or entry.get("content_type") or "documentation").lower()
                if target_collection not in stores:
                    target_store = _get_or_create_store(target_collection)
                    stores[target_collection] = target_store
                    routed_documents[target_collection] = []
                    routed_metadata[target_collection] = []
                entry["collection_name"] = target_collection
                entry["content_type"] = target_collection
                routed_documents[target_collection].append(document)
                routed_metadata[target_collection].append(entry)

            result = {"documents_uploaded": 0, "chunks_uploaded": 0, "total_chunks": _total_chunks("all"), "per_collection": {}}
            for target_collection, documents in routed_documents.items():
                if not documents:
                    continue
                collection_result = _get_or_create_store(target_collection).add_documents(documents, routed_metadata[target_collection])
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
        normalized_collection = _normalize_collection_name(collection_name)
        chunks: list[dict[str, Any]] = []
        for selected_store in _selected_stores(normalized_collection):
            chunks.extend(selected_store.get_chunks(source=source, limit=safe_limit, offset=0))
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
        normalized_collection = _normalize_collection_name(collection_name)
        for selected_store in _selected_stores(normalized_collection):
            selected_store.clear()
        logger.info("Database cleared")
        return {
            "status": "success",
            "collection_name": normalized_collection,
            "message": "Documents cleared"
        }

    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    from config import RAG_SERVER_HOST, RAG_SERVER_PORT
    
    logger.info(f"Starting RAG server on {RAG_SERVER_HOST}:{RAG_SERVER_PORT}")
    logger.info(f"LM Studio URL: {LM_STUDIO_BASE_URL}")
    uvicorn.run(
        app,
        host=RAG_SERVER_HOST,
        port=RAG_SERVER_PORT
    )
