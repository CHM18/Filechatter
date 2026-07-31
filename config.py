"""Configuration for Filechatter RAG Server

All settings are read from .env file. See .env.example for available options.
config.py is read-only; to change settings, edit .env
"""
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

# Set Sentence Transformers cache to local llmodels directory
LLMODELS_DIR = BASE_DIR / "llmodels"
LLMODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(LLMODELS_DIR)

# LM Studio API Configuration
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "local-model")

# RAG Server Configuration
RAG_SERVER_HOST = os.getenv("RAG_HOST", "localhost")
RAG_SERVER_PORT = int(os.getenv("RAG_PORT", "8000"))

# Vector Database Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
DATA_DIR = _resolve_path(os.getenv("DATA_DIR", "./data"))

# RAG Parameters
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
LM_STUDIO_TIMEOUT = int(os.getenv("LM_STUDIO_TIMEOUT", "60"))
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1800"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "250"))
CHUNK_SPLIT_MARGIN = int(os.getenv("CHUNK_SPLIT_MARGIN", "400"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "10"))
KEYWORD_RETRIEVAL_K = int(os.getenv("KEYWORD_RETRIEVAL_K", "15"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))
MAX_CHUNKS_PER_SOURCE = int(os.getenv("MAX_CHUNKS_PER_SOURCE", "4"))
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "1.0"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "2.0"))
METADATA_KEYWORD_WEIGHT = float(os.getenv("METADATA_KEYWORD_WEIGHT", "6.0"))
