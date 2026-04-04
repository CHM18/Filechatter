"""Configuration for Filechatter RAG Server"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent


def _resolve_path(value: str) -> str:
	path = Path(value).expanduser()
	if not path.is_absolute():
		path = BASE_DIR / path
	return str(path.resolve())


load_dotenv(BASE_DIR / ".env")

# LM Studio API Configuration
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "local-model")

# RAG Server Configuration
RAG_SERVER_HOST = os.getenv("RAG_HOST", "localhost")
RAG_SERVER_PORT = int(os.getenv("RAG_PORT", "8000"))

# Vector DB Configuration
CHROMA_DB_PATH = _resolve_path(os.getenv("CHROMA_DB_PATH", "./data/chroma"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DATA_DIR = _resolve_path(os.getenv("DATA_DIR", "./data"))

# RAG Parameters
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "3"))  # Number of documents to retrieve
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
LM_STUDIO_TIMEOUT = int(os.getenv("LM_STUDIO_TIMEOUT", "60"))  # Timeout for LM Studio requests in seconds
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1800"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "250"))
CHUNK_SPLIT_MARGIN = int(os.getenv("CHUNK_SPLIT_MARGIN", "400"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "10"))
KEYWORD_RETRIEVAL_K = int(os.getenv("KEYWORD_RETRIEVAL_K", "15"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
MAX_CHUNKS_PER_SOURCE = int(os.getenv("MAX_CHUNKS_PER_SOURCE", "4"))
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "1.0"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "2.0"))
