import os
import shutil
import tempfile
import pytest
from src.database.relational import RelationalDB


@pytest.fixture
def temp_db():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_rag.db")
    db = RelationalDB(db_path=db_path)
    yield db
    # Cleanup temp directory
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


def test_document_lifecycle(temp_db):
    doc = temp_db.create_document(
        doc_id="doc-123",
        filename="handbook.pdf",
        file_size=1024,
        mime_type="application/pdf",
        total_chunks=10
    )
    assert doc["id"] == "doc-123"
    assert doc["filename"] == "handbook.pdf"

    retrieved = temp_db.get_document("doc-123")
    assert retrieved is not None
    assert retrieved["total_chunks"] == 10

    docs = temp_db.list_documents()
    assert len(docs) == 1

    deleted = temp_db.delete_document("doc-123")
    assert deleted is True
    assert temp_db.get_document("doc-123") is None


def test_session_and_message_history(temp_db):
    session = temp_db.create_session(session_id="session-1", title="Test Session")
    assert session["id"] == "session-1"

    # Add messages
    temp_db.save_message("session-1", "user", "What is PTO?")
    temp_db.save_message(
        "session-1", 
        "assistant", 
        "PTO is 20 days [handbook.pdf, Page 14]", 
        citations=[{"document_name": "handbook.pdf", "page_number": 14}]
    )
    temp_db.save_message("session-1", "user", "Does that apply to contractors?")
    temp_db.save_message(
        "session-1", 
        "assistant", 
        "Contractors are not eligible for PTO [handbook.pdf, Page 14]", 
        citations=[{"document_name": "handbook.pdf", "page_number": 14}]
    )

    # Fetch last 3 messages (should return chronological order)
    history = temp_db.get_session_history("session-1", limit=3)
    assert len(history) == 3
    assert history[0]["role"] == "assistant"
    assert history[0]["content"] == "PTO is 20 days [handbook.pdf, Page 14]"
    assert history[1]["role"] == "user"
    assert history[1]["content"] == "Does that apply to contractors?"
    assert history[2]["role"] == "assistant"
    assert len(history[2]["citations"]) == 1
    assert history[2]["citations"][0]["document_name"] == "handbook.pdf"
