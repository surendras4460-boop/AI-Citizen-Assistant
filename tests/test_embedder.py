"""Unit tests for the SentenceEmbedder component."""
import pytest
from src.embedder import SentenceEmbedder
from src.config import EMBEDDING_DIMENSION


def test_embedder_initialization():
    embedder = SentenceEmbedder()
    assert embedder.backend in ["sentence_transformers", "chroma_onnx", "hash_vectorizer"]


def test_embed_single_query():
    embedder = SentenceEmbedder()
    vec = embedder.embed_query("Scholarship eligibility")
    assert isinstance(vec, list)
    assert len(vec) == EMBEDDING_DIMENSION
    assert any(v != 0.0 for v in vec)


def test_embed_empty_list():
    embedder = SentenceEmbedder()
    res = embedder.embed_texts([])
    assert res == []


def test_hash_vectorizer_normalization():
    embedder = SentenceEmbedder()
    vec = embedder._hash_embed("Test sample query for vectorization")
    assert len(vec) == EMBEDDING_DIMENSION
    # Check unit norm
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-4
