"""
Integration tests for FastAPI REST API endpoints (/api/upload, /api/chat, /api/documents).
"""

import io
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_upload_and_chat_api_pipeline():
    # 1. Upload sample document
    doc_content = (
        "COMPANY POLICY 2026\n\n"
        "Section 1: Parental Leave\n"
        "Eligible employees receive 16 weeks of fully paid parental leave following the birth or adoption of a child.\n"
        "Contractors are not eligible for parental leave benefits.\n\n"
        "Section 2: Health Insurance\n"
        "Comprehensive health insurance starts on the first day of employment."
    )
    
    files = [
        ("files", ("company_policy.txt", io.BytesIO(doc_content.encode("utf-8")), "text/plain"))
    ]
    
    upload_res = client.post("/api/upload", files=files)
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    assert upload_data["status"] == "success"
    assert upload_data["total_processed"] == 1

    # 2. Check document is listed
    docs_res = client.get("/api/documents")
    assert docs_res.status_code == 200
    docs = docs_res.json()["documents"]
    assert any(d["filename"] == "company_policy.txt" for d in docs)

    # 3. Query chat endpoint
    chat_payload = {
        "query": "How many weeks of parental leave do eligible employees get?"
    }
    chat_res = client.post("/api/chat", json=chat_payload)
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert "16 weeks" in chat_data["answer"] or "parental leave" in chat_data["answer"].lower()
    assert len(chat_data["citations"]) >= 1
    assert chat_data["citations"][0]["document_name"] == "company_policy.txt"
    assert chat_data["citations"][0]["page_number"] == 1
    session_id = chat_data["session_id"]

    # 4. Multi-turn follow-up with session_id
    followup_payload = {
        "query": "Does that apply to contractors?",
        "session_id": session_id
    }
    followup_res = client.post("/api/chat", json=followup_payload)
    assert followup_res.status_code == 200
    followup_data = followup_res.json()
    assert followup_data["session_id"] == session_id
    assert "contractor" in followup_data["answer"].lower()

    # 5. Out-of-Scope Query (Graceful Failure - Anti-Hallucination)
    unrelated_payload = {
        "query": "What is the boiling point of liquid nitrogen in Kelvin?"
    }
    unrelated_res = client.post("/api/chat", json=unrelated_payload)
    assert unrelated_res.status_code == 200
    unrelated_data = unrelated_res.json()
    assert unrelated_data["answer"] == "I could not find an answer in the provided documents."
    assert len(unrelated_data["citations"]) == 0
