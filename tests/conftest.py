"""Shared test fixtures — no real embedding model or network calls in the test suite."""
import hashlib

import pytest
from langchain_core.embeddings import Embeddings

VECTOR_DIM = 16


class FakeEmbeddings(Embeddings):
    """Deterministic, hash-based fake embeddings — fast and network-free for tests."""

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:VECTOR_DIM]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()
