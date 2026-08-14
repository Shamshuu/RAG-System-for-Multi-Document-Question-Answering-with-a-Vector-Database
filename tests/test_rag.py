"""
Integration and End-to-End Tests for the complete RAG System.
Tests ingestion, semantic retrieval, citations, conversation history, and graceful failure.
"""

import os
import shutil
import tempfile
import pytest
from src.core.rag import RAGSystem


@pytest.fixture
def rag_instance():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_rag.db")
    v_path = os.path.join(temp_dir, "test_vectors.json")
    
    rag = RAGSystem(
        db_path=db_path,
        vector_store_path=v_path,
        similarity_threshold=0.35,
        top_k=3
    )
    yield rag
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


def test_full_rag_lifecycle(rag_instance):
    # 1. Ingest test document
    doc_content = (
        "EMPLOYEE HANDBOOK\n\n"
        "Chapter 1: Paid Time Off (PTO)\n"
        "Full-time regular employees are entitled to 20 days of paid time off per calendar year.\n"
        "Contractors and temporary workers are not eligible for paid time off.\n\n"
        "Chapter 2: Remote Work Policy\n"
        "Employees may work remotely up to 3 days per week upon manager approval."
    )
    
    ingest_res = rag_instance.ingest_document(
        file_bytes=doc_content.encode("utf-8"),
        filename="employee_handbook.txt",
        mime_type="text/plain"
    )

    assert ingest_res["status"] == "success"
    assert ingest_res["total_chunks"] >= 1
    assert ingest_res["filename"] == "employee_handbook.txt"

    # 2. Query factual question (Grounded Answer + Citation)
    query_res = rag_instance.query("How many days of paid time off do full-time employees get?")
    
    assert "20 days" in query_res["answer"] or "paid time off" in query_res["answer"].lower()
    assert len(query_res["citations"]) >= 1
    assert query_res["citations"][0]["document_name"] == "employee_handbook.txt"
    assert query_res["citations"][0]["page_number"] == 1
    session_id = query_res["session_id"]
    assert session_id is not None

    # 3. Follow-up question using the same session_id
    follow_up_res = rag_instance.query(
        "Does that apply to contractors?",
        session_id=session_id
    )
    assert follow_up_res["session_id"] == session_id
    assert "contractor" in follow_up_res["answer"].lower()

    # 4. Out-of-Scope Query (Graceful Failure - Anti-Hallucination)
    unrelated_res = rag_instance.query("What is the capital city of France?", threshold=0.75)
    assert unrelated_res["answer"] == "I could not find an answer in the provided documents."
    assert len(unrelated_res["citations"]) == 0
