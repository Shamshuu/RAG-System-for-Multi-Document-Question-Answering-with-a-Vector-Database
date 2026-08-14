"""
Vector Database Service
Stores dense vector embeddings with metadata payloads and performs fast cosine similarity searches.
Supports persistence, top-k retrieval, threshold filtering, and document-level deletion.
"""

import json
import os
from typing import Any, Dict, List, Optional
import numpy as np


class VectorStore:
    """In-memory vector store with persistence and Cosine Similarity nearest-neighbor search."""

    def __init__(self, storage_path: str = "./vector_store.json"):
        self.storage_path = storage_path
        # Map: chunk_id -> { "id": str, "values": List[float], "metadata": Dict[str, Any] }
        self.vectors: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Loads persisted vector records from disk if present."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.vectors = json.load(f)
            except Exception:
                self.vectors = {}

    def _save(self) -> None:
        """Persists vector records to disk."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.vectors, f, ensure_ascii=False)
        except Exception:
            pass

    def upsert(self, records: List[Dict[str, Any]]) -> int:
        """
        Upserts vector records into the store.
        Each record must contain: { "id": str, "values": List[float], "metadata": dict }
        """
        for record in records:
            chunk_id = record["id"]
            self.vectors[chunk_id] = {
                "id": chunk_id,
                "values": record["values"],
                "metadata": record["metadata"]
            }
        self._save()
        return len(records)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        threshold: float = 0.45,
        filter_doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs Cosine Similarity search against all stored vectors.
        Returns top-K results with similarity score >= threshold.
        """
        if not self.vectors or not query_vector:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_unit = q_vec / q_norm

        scored_results = []

        for chunk_id, record in self.vectors.items():
            metadata = record.get("metadata", {})
            if filter_doc_id and metadata.get("document_id") != filter_doc_id:
                continue

            v_vec = np.array(record["values"], dtype=np.float32)
            if v_vec.shape != q_vec.shape:
                continue
            v_norm = np.linalg.norm(v_vec)
            if v_norm == 0:
                continue
            v_unit = v_vec / v_norm

            # Cosine similarity is dot product of unit vectors
            score = float(np.dot(q_unit, v_unit))

            if score >= threshold:
                scored_results.append({
                    "id": chunk_id,
                    "score": round(score, 4),
                    "metadata": metadata
                })

        # Sort descending by similarity score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    def delete_by_document_id(self, document_id: str) -> int:
        """Removes all vectors belonging to a specific document ID."""
        to_delete = [
            cid for cid, rec in self.vectors.items() 
            if rec.get("metadata", {}).get("document_id") == document_id
        ]
        for cid in to_delete:
            del self.vectors[cid]
        if to_delete:
            self._save()
        return len(to_delete)

    def clear(self) -> None:
        """Clears all vectors in the store."""
        self.vectors.clear()
        self._save()

    def count(self) -> int:
        """Returns the total number of stored vector chunks."""
        return len(self.vectors)
