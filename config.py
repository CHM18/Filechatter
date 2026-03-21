"""Configuration for Filechatter RAG Server"""
import os
from dotenv import load_dotenv

load_dotenv()

# LM Studio API Configuration
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "local-model")

# RAG Server Configuration
RAG_SERVER_HOST = os.getenv("RAG_HOST", "localhost")
RAG_SERVER_PORT = int(os.getenv("RAG_PORT", "8000"))

# Vector DB Configuration
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# RAG Parameters
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "3"))  # Number of documents to retrieve
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
LM_STUDIO_TIMEOUT = int(os.getenv("LM_STUDIO_TIMEOUT", "60"))  # Timeout for LM Studio requests in seconds
