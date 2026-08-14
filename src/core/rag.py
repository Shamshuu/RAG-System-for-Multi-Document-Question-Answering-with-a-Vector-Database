"""
RAG Core Orchestrator
Coordinates the Ingestion Pipeline (parsing, chunking, embedding, vector storage)
and the Query Pipeline (semantic retrieval, history injection, LLM generation, citation delivery).
"""

import os
import uuid
from typing import Any, Dict, List, Optional

from src.database.relational import RelationalDB
from src.database.vector_store import VectorStore
from src.services.chunking import chunk_document
from src.services.document import DocumentParser
from src.services.embedding import EmbeddingService
from src.services.llm import LLMService


class RAGSystem:
    """End-to-end Retrieval-Augmented Generation system."""

    def __init__(
        self,
        db_path: str = "./rag_history.db",
        vector_store_path: str = "./vector_store.json",
        similarity_threshold: Optional[float] = None,
        top_k: Optional[int] = None
    ):
        self.relational_db = RelationalDB(db_path=db_path)
        self.vector_store = VectorStore(storage_path=vector_store_path)
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()

        # Load threshold and top_k from environment or defaults
        thresh_env = os.getenv("SIMILARITY_THRESHOLD", "0.10")
        top_k_env = os.getenv("TOP_K", "5")
        
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else float(thresh_env)
        self.top_k = top_k if top_k is not None else int(top_k_env)

    # --- Ingestion Pipeline ---

    def ingest_document(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str = "application/octet-stream"
    ) -> Dict[str, Any]:
        """
        Executes the ingestion pipeline:
        1. Extracts text preserving page numbers from PDF/DOCX.
        2. Splits text into overlapping chunks with source metadata.
        3. Generates dense vector embeddings.
        4. Upserts vectors to the Vector Database.
        5. Persists document record to the Relational Database.
        """
        doc_id = str(uuid.uuid4())
        file_size = len(file_bytes)

        # 1. Parse document pages
        pages = DocumentParser.parse_document(file_bytes, filename, doc_id)
        if not pages:
            raise ValueError(f"No extractable text found in '{filename}'.")

        # 2. Chunk pages with sliding window
        chunks = chunk_document(pages, chunk_size=1000, overlap=200)

        # 3. Generate dense vector embeddings in batches (enrich text with filename context)
        enriched_texts = [f"Document: {filename}\n{c.text}" for c in chunks]
        vectors = self.embedding_service.embed_batch(enriched_texts, batch_size=50)

        # 4. Prepare vector records and upsert into Vector Store
        vector_records = []
        for chunk, vector in zip(chunks, vectors):
            vector_records.append({
                "id": chunk.id,
                "values": vector,
                "metadata": chunk.metadata
            })
        self.vector_store.upsert(vector_records)

        # 5. Log document in Relational DB
        doc_record = self.relational_db.create_document(
            doc_id=doc_id,
            filename=filename,
            file_size=file_size,
            mime_type=mime_type,
            total_chunks=len(chunks)
        )

        return {
            "document_id": doc_id,
            "filename": filename,
            "file_size": file_size,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "status": "success"
        }

    # --- Query Pipeline ---

    def _build_search_query(self, query_text: str, history: List[Dict[str, Any]]) -> str:
        """
        Contextualizes search query with recent user queries to resolve conversational pronouns.
        """
        if not history:
            return query_text

        # Get last user query turn
        prior_user_turns = [m["content"] for m in history if m.get("role") == "user"]
        if prior_user_turns:
            last_turn = prior_user_turns[-1]
            return f"{last_turn} {query_text}"
        return query_text

    def query(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes the query pipeline:
        1. Detects summary / overview intent or factual search.
        2. Contextualizes query with history for pronoun resolution.
        3. Retrieves Top-K vector chunks with confidence threshold filtering.
        4. Injects prior chat history into prompt.
        5. Builds grounded prompt and invokes LLM.
        6. Persists user & assistant turns with structured citations in Relational DB.
        7. Returns structured response.
        """
        k = top_k if top_k is not None else self.top_k
        thresh = threshold if threshold is not None else self.similarity_threshold
        fallback_msg = "I could not find an answer in the provided documents."

        # Ensure active session exists
        session = self.relational_db.get_or_create_session(session_id)
        active_session_id = session["id"]

        # Fetch recent conversation history (last 5 turns)
        history = self.relational_db.get_session_history(active_session_id, limit=5)

        # 1. Check if user is asking for a summary/overview of the files
        is_summary = self.embedding_service.is_summary_intent(query_text)
        
        if is_summary and self.vector_store.count() > 0:
            # For summary queries, retrieve top representative chunks across indexed documents
            all_records = list(self.vector_store.vectors.values())
            retrieved_chunks = [
                {"id": r["id"], "score": 1.0, "metadata": r["metadata"]}
                for r in all_records[:k]
            ]
        else:
            # 2. Contextualize search query for vector retrieval
            search_query_text = self._build_search_query(query_text, history)
            query_vector = self.embedding_service.embed_text(search_query_text)

            # 3. Vector DB semantic retrieval with threshold filtering
            retrieved_chunks = self.vector_store.search(
                query_vector=query_vector,
                top_k=k,
                threshold=thresh
            )

        # 4. Graceful failure if no chunks meet threshold
        if not retrieved_chunks:
            # Persist turns
            self.relational_db.save_message(active_session_id, "user", query_text)
            self.relational_db.save_message(active_session_id, "assistant", fallback_msg, citations=[])
            return {
                "answer": fallback_msg,
                "citations": [],
                "session_id": active_session_id
            }

        # 5. Generate LLM grounded response
        response = self.llm_service.generate_response(
            query=query_text,
            retrieved_chunks=retrieved_chunks,
            conversation_history=history
        )

        answer = response["answer"]
        citations = response["citations"]

        # 6. Persist user and assistant messages
        self.relational_db.save_message(active_session_id, "user", query_text)
        self.relational_db.save_message(active_session_id, "assistant", answer, citations=citations)

        return {
            "answer": answer,
            "citations": citations,
            "session_id": active_session_id
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns all documents registered in the system."""
        return self.relational_db.list_documents()

    def delete_document(self, document_id: str) -> bool:
        """Deletes a document from both vector store and relational database."""
        self.vector_store.delete_by_document_id(document_id)
        return self.relational_db.delete_document(document_id)

    def get_session_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns conversation messages for a session."""
        return self.relational_db.get_session_history(session_id, limit=limit)
