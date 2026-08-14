"""
FastAPI Route Handlers for Document Ingestion and Chat Retrieval.
Defines endpoints: POST /api/upload, POST /api/chat, GET /api/documents, GET /api/sessions.
"""

from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.core.rag import RAGSystem

router = APIRouter(prefix="/api")

# Instantiate singleton RAG orchestrator
rag_system = RAGSystem()


# --- Pydantic Schemas ---

class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's question to ask the document assistant.")
    session_id: Optional[str] = Field(None, description="Optional conversation session ID for continuity.")


class CitationItem(BaseModel):
    document_name: str
    page_number: int


class ChatResponse(BaseModel):
    answer: str
    citations: List[CitationItem]
    session_id: str


# --- Endpoints ---

@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "total_documents": len(rag_system.list_documents()),
        "total_vectors": rag_system.vector_store.count()
    }


@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Accepts multipart/form-data containing one or more PDF/DOCX/TXT files.
    Extracts text, splits into page-aware overlapping chunks, generates embeddings,
    and stores in vector and relational databases.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results = []
    for file in files:
        filename = file.filename or "unknown_file"
        file_bytes = await file.read()

        if not file_bytes:
            continue

        try:
            res = rag_system.ingest_document(
                file_bytes=file_bytes,
                filename=filename,
                mime_type=file.content_type or "application/octet-stream"
            )
            results.append(res)
        except Exception as e:
            results.append({
                "filename": filename,
                "status": "error",
                "error": str(e)
            })

    return {
        "status": "success",
        "processed_files": results,
        "total_processed": len(results)
    }


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Accepts a user question, performs semantic search, fetches history,
    constructs a grounded prompt, and returns the generated answer + structured citations.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        response = rag_system.query(
            query_text=request.query,
            session_id=request.session_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@router.get("/documents")
def list_documents():
    """Returns all registered documents and their metadata."""
    return {"documents": rag_system.list_documents()}


@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    """Deletes a document from vector index and relational database."""
    deleted = rag_system.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"status": "success", "deleted_id": document_id}


@router.get("/sessions")
def list_sessions():
    """Returns all chat conversation sessions."""
    return {"sessions": rag_system.relational_db.list_sessions()}


@router.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    """Returns all message turns for a given session."""
    messages = rag_system.get_session_history(session_id, limit=50)
    return {"session_id": session_id, "messages": messages}
