"""RAG server for Filechatter with persistent chunked retrieval."""
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
import logging
from langchain.schema import Document

import config
from rag_store import RagStore, SearchResult
from langchain_support import (
    LMStudioChatClient,
    RagRetriever,
    build_prompt,
    format_context,
    parse_model_output,
)

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
retriever = RagRetriever(store)
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
        retrieved_documents = retriever.get_relevant_documents(request.question) if request.include_context else []
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
            _serialize_result(result) for result in store.search(request.question)
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
        retrieved_documents = retriever.get_relevant_documents(request.question) if request.include_context else []
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
            context=[_serialize_result(result) for result in store.search(request.question)] if request.include_context else [],
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
