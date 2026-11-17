"""Unit tests for the VectorIndex and retrieval pipeline."""
import pytest
from src.embedder import SentenceEmbedder
from src.models import Chunk
from src.vector_store import VectorIndex, retrieve


@pytest.fixture
def sample_embedder():
    return SentenceEmbedder()


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            text="PM Kisan Samman Nidhi provides direct income support of 6000 rupees to farmers.",
            meta={"kind": "scheme", "source": "PM-KISAN", "page": "Overview", "score": 0.0}
        ),
        Chunk(
            text="Post Matric Scholarship reimburses complete tuition fees for SC and ST students.",
            meta={"kind": "scheme", "source": "Post Matric", "page": "Benefits", "score": 0.0}
        ),
        Chunk(
            text="Ayushman Bharat provides health insurance coverage up to 5 lakh rupees per family.",
            meta={"kind": "scheme", "source": "AB-PMJAY", "page": "Details", "score": 0.0}
        )
    ]


def test_vector_index_add_and_query(sample_embedder, sample_chunks):
    idx = VectorIndex("test_collection", sample_embedder)
    idx.add(sample_chunks)

    results = idx.query("How much money do farmers get in PM Kisan?", top_k=2)
    assert len(results) > 0
    top_hit = results[0]
    assert "PM-KISAN" in top_hit.source
    assert top_hit.score > 0.0


def test_retrieve_across_indexes(sample_embedder, sample_chunks):
    idx1 = VectorIndex("test_idx1", sample_embedder)
    idx1.add(sample_chunks[:2])

    idx2 = VectorIndex("test_idx2", sample_embedder)
    idx2.add(sample_chunks[2:])

    hits = retrieve([idx1, idx2, None], "scholarship for students", top_k=3)
    assert len(hits) > 0
    assert any("Post Matric" in h.source for h in hits)
