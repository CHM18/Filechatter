"""RAG Server for Filechatter - Integrates with LM Studio"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import numpy as np
import requests
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Filechatter RAG Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize FAISS and sentence transformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search

# Store documents and their embeddings
documents = []
doc_ids = []
id_counter = 0

# Configuration
LM_STUDIO_BASE_URL = config.LM_STUDIO_BASE_URL
LM_STUDIO_MODEL = config.LM_STUDIO_MODEL
CONTEXT_SIZE = config.CONTEXT_SIZE
class QueryRequest(BaseModel):
    question: str
    include_context: bool = True


class DocumentUpload(BaseModel):
    documents: list[str]
    metadata: list[dict] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    context: list[str] = []


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
        "documents_count": len(documents)
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest):
    """Chat endpoint with RAG - queries documents and LM Studio"""
    try:
        context = ""
        context_docs = []
        
        # Retrieve relevant documents if there are any
        if len(documents) > 0 and request.include_context:
            # Encode the query
            query_embedding = embedding_model.encode([request.question])
            
            # Search for similar documents
            k = min(CONTEXT_SIZE, len(documents))
            distances, indices = index.search(query_embedding.astype(np.float32), k)
            
            # Get the retrieved documents
            context_docs = [documents[i] for i in indices[0]]
            context = "\n---\n".join(context_docs)
        
        # Prepare messages for LM Studio
        system_message = "You are a helpful assistant."
        if context:
            system_message += f"\n\nUse the following context to answer the question:\n{context}"
        
        # Call LM Studio API
        response = requests.post(
            f"{LM_STUDIO_BASE_URL}/v1/chat/completions",
            json={
                "model": LM_STUDIO_MODEL,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": request.question}
                ],
                "temperature": 0.7,
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
            context=context_docs
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
    """Upload documents to vector database"""
    global id_counter
    
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        # Generate embeddings for the documents
        embeddings = embedding_model.encode(request.documents)
        
        # Add to FAISS index
        index.add(embeddings.astype(np.float32))
        
        # Store documents and IDs
        for doc in request.documents:
            documents.append(doc)
            doc_ids.append(f"doc_{id_counter}")
            id_counter += 1
        
        logger.info(f"Uploaded {len(request.documents)} documents")
        return {
            "status": "success",
            "documents_uploaded": len(request.documents),
            "total_documents": len(documents)
        }
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List all documents in the database"""
    try:
        return {
            "total_documents": len(documents),
            "collection_name": "faiss_index"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the database"""
    global documents, doc_ids, id_counter
    
    try:
        # Clear FAISS index and document lists
        index.reset()
        documents.clear()
        doc_ids.clear()
        id_counter = 0
        
        logger.info("Database cleared")
        return {
            "status": "success",
            "message": "All documents cleared"
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
