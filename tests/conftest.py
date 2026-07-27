"""Shared pytest fixtures.

Every test runs against an isolated temp data directory and a fake embedding
model, so no network access, no model download, and no interference with the
real ./data directory.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import ingest_jobs  # noqa: E402
import runtime  # noqa: E402
from rag_store import RagStore  # noqa: E402

EMBEDDING_DIM = 32


class FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer."""

    def encode(self, texts, normalize_embeddings=True, **_kwargs):
        vectors = []
        for text in texts:
            digest = hashlib.md5(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "little")
            rng = np.random.default_rng(seed)
            vector = rng.random(EMBEDDING_DIM, dtype=np.float32)
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.stack(vectors)


@pytest.fixture()
def data_env(tmp_path, monkeypatch):
    """Isolated data dir + fake embedder + fresh runtime singletons."""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(ingest_jobs, "JOB_LOG_DIR", tmp_path / "logs")

    fake = FakeEmbedder()
    monkeypatch.setattr(
        RagStore,
        "_get_shared_embedding_model",
        classmethod(lambda cls: (fake, EMBEDDING_DIM)),
    )

    runtime.reset()
    yield data_dir
    runtime.reset()


@pytest.fixture()
def client(data_env):
    """FastAPI test client against the isolated environment."""
    from fastapi.testclient import TestClient

    import rag_server

    with TestClient(rag_server.app) as test_client:
        yield test_client
