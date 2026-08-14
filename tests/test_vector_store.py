"""
Unit tests for the VectorStore and EmbeddingService.
"""

import os
import tempfile
import pytest
from src.services.embedding import EmbeddingService
from src.database.vector_store import VectorStore


@pytest.fixture
def temp_vector_store():
    temp_dir = tempfile.mkdtemp()
    store_path = os.path.join(temp_dir, "test_vectors.json")
    store = VectorStore(storage_path=store_path)
    yield store
    if os.path.exists(store_path):
        try:
            os.remove(store_path)
        except Exception:
            pass


def test_embedding_and_vector_search(temp_vector_store):
    embedder = EmbeddingService()
    
    text1 = "Employees receive 20 days of paid time off every calendar year."
    text2 = "Kubernetes clusters can be deployed across multiple cloud availability zones."
    
    v1 = embedder.embed_text(text1)
    v2 = embedder.embed_text(text2)

    records = [
        {
            "id": "chunk-1",
            "values": v1,
            "metadata": {
                "document_id": "doc-hr",
                "filename": "handbook.pdf",
                "page_number": 14,
                "text": text1
            }
        },
        {
            "id": "chunk-2",
            "values": v2,
            "metadata": {
                "document_id": "doc-infra",
                "filename": "cloud_guide.pdf",
                "page_number": 3,
                "text": text2
            }
        }
    ]

    temp_vector_store.upsert(records)
    assert temp_vector_store.count() == 2

    # Query matching HR topic
    query = "How many days of paid vacation do employees get?"
    q_vec = embedder.embed_text(query)
    results = temp_vector_store.search(q_vec, top_k=2, threshold=0.1)

    assert len(results) >= 1
    top_result = results[0]
    assert top_result["metadata"]["document_id"] == "doc-hr"
    assert top_result["metadata"]["filename"] == "handbook.pdf"
    assert top_result["metadata"]["page_number"] == 14
    assert top_result["score"] > 0.1


def test_threshold_filtering(temp_vector_store):
    embedder = EmbeddingService()
    text = "The quick brown fox jumps over the lazy dog."
    v = embedder.embed_text(text)
    
    temp_vector_store.upsert([{
        "id": "chunk-fox",
        "values": v,
        "metadata": {"document_id": "doc-animal", "filename": "story.txt", "page_number": 1, "text": text}
    }])

    # Completely unrelated query with high threshold
    unrelated_query = "quantum electrodynamics perturbation theory"
    q_vec = embedder.embed_text(unrelated_query)
    
    results = temp_vector_store.search(q_vec, top_k=5, threshold=0.85)
    assert len(results) == 0
