"""RAG server for Filechatter with persistent chunked retrieval."""
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import logging

import config
from rag_store import RagStore, SearchResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Filechatter RAG Server", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = RagStore()

# Configuration
LM_STUDIO_BASE_URL = config.LM_STUDIO_BASE_URL
LM_STUDIO_MODEL = config.LM_STUDIO_MODEL


class QueryRequest(BaseModel):
    question: str
    include_context: bool = True


class DocumentUpload(BaseModel):
    documents: list[str]
    metadata: list[dict[str, Any]] | None = None


class RetrievedChunk(BaseModel):
    chunk_id: int
    source: str
    content: str
    metadata: dict[str, Any]
    score: float
    rank: int


class ChatResponse(BaseModel):
    question: str
    answer: str
    context: list[RetrievedChunk] = []


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
    )


def _format_context(results: list[SearchResult]) -> str:
    blocks = []
    for result in results:
        blocks.append(
            "\n".join(
                [
                    f"Source: {result.source}",
                    f"Chunk: {result.metadata.get('chunk_index', result.rank)}",
                    result.content,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


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
        "documents_count": len(store.list_sources()),
        "chunks_count": store.chunk_count,
        "data_dir": config.DATA_DIR,
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: QueryRequest):
    """Retrieve relevant chunks without invoking the LLM."""
    results = store.search(request.question) if request.include_context else []
    return SearchResponse(
        question=request.question,
        results=[_serialize_result(result) for result in results],
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest):
    """Compatibility chat endpoint using chunked retrieval plus LM Studio."""
    try:
        search_results = store.search(request.question) if request.include_context else []
        context = _format_context(search_results)

        # Prepare messages for LM Studio
        system_message = (
            "You are a helpful assistant. Answer from the retrieved context when it is relevant. "
            "If the context is insufficient, say that directly instead of inventing details."
        )
        if context:
            system_message += (
                "\n\nUse the following retrieved context. Cite the source name when you rely on it:\n"
                f"{context}"
            )

        # Call LM Studio API
        response = requests.post(
            f"{LM_STUDIO_BASE_URL}/v1/chat/completions",
            json={
                "model": LM_STUDIO_MODEL,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": request.question}
                ],
                "temperature": config.TEMPERATURE,
                "top_p": 0.9,
            },
            timeout=config.LM_STUDIO_TIMEOUT
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"LM Studio error: {response.text}"
            )
        
        answer = response.json()["choices"][0]["message"]["content"]
        
        return ChatResponse(
            question=request.question,
            answer=answer,
            context=[_serialize_result(result) for result in search_results]
        )

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to LM Studio. Ensure it's running on " + LM_STUDIO_BASE_URL
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_documents(request: DocumentUpload):
    """Upload documents to the persistent chunked store."""
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="No documents provided")

        result = store.add_documents(request.documents, request.metadata)
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
            "total_documents": len(store.list_sources()),
            "total_chunks": store.chunk_count,
            "collection_name": "faiss_sqlite_store",
            "sources": store.list_sources(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chunks")
async def list_chunks(
    source: str | None = None,
    limit: int = 20,
    offset: int = 0,
):
    """List stored chunks for debugging indexed content."""
    try:
        safe_limit = max(1, min(limit, 200))
        safe_offset = max(0, offset)
        chunks = store.get_chunks(source=source, limit=safe_limit, offset=safe_offset)
        return {
            "source": source,
            "count": len(chunks),
            "limit": safe_limit,
            "offset": safe_offset,
            "chunks": chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the database"""
    try:
        store.clear()
        logger.info("Database cleared")
        return {
            "status": "success",
            "message": "All documents cleared"
        }

    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    """Close local resources on server shutdown."""
    store.close()


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
