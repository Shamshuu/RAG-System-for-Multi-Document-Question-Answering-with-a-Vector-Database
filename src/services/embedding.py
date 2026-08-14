"""
Vector Embedding Service
Converts text chunks and user queries into dense, normalized floating-point vectors.
Employs hybrid token hashing, morphological normalization, and IDF weighting.
"""

import hashlib
import math
import os
import re
from typing import List, Optional, Set
import numpy as np

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "for", "to", "of", "with", "by", "from", "as",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "that", "this", "these", "those", "it", "its", "and", "or", "but", "if", "then", "so",
    "what", "which", "who", "whom", "where", "when", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so"
}

SUMMARY_KEYWORDS = {
    "summarize", "summary", "summarise", "overview", "about", "explain", "brief", "gist",
    "outline", "contents", "tell", "describe", "detail", "details", "everything", "file", "document", "documents"
}


class EmbeddingService:
    """
    Generates normalized embedding vectors for chunks and search queries.
    Uses identical embedding logic for both ingestion and querying.
    """

    def __init__(self, provider: str = "local", dimension: int = 1024):
        self.provider = os.getenv("EMBEDDING_PROVIDER", provider).lower()
        self.dimension = dimension

    @staticmethod
    def _stem(word: str) -> str:
        """Lightweight morphological suffix normalization."""
        w = word.lower()
        if len(w) > 4:
            if w.endswith("ies"):
                return w[:-3] + "y"
            if w.endswith("ing"):
                return w[:-3]
            if w.endswith("ed"):
                return w[:-2]
            if w.endswith("s") and not w.endswith("ss"):
                return w[:-1]
        return w

    @classmethod
    def is_summary_intent(cls, query: str) -> bool:
        """Detects if user is asking for a summary, overview, or explanation of the files."""
        q_lower = query.lower().strip()
        tokens = set(re.findall(r"\b\w+\b", q_lower))
        
        # Check explicit summary phrases
        summary_phrases = [
            "summarize", "summarise", "what is this", "what this file", "what is the document",
            "what are the documents", "give me an overview", "brief summary", "explain this file",
            "about this file", "about the document", "about the guidelines"
        ]
        if any(p in q_lower for p in summary_phrases):
            return True
            
        return bool(tokens & SUMMARY_KEYWORDS) and len(tokens) <= 6

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Applies L2 normalization so dot product equals cosine similarity."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _generate_local_embedding(self, text: str) -> List[float]:
        """
        Generates a 1024-dimensional dense normalized semantic feature vector.
        """
        cleaned = text.lower().strip()
        raw_tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", cleaned)
        if not raw_tokens:
            return [0.0] * self.dimension

        tokens = [self._stem(t) for t in raw_tokens]
        vector = np.zeros(self.dimension, dtype=np.float32)

        # 1. Unigram feature projection with stopword filtering
        for i, token in enumerate(tokens):
            if token in STOPWORDS:
                continue
            
            weight = 4.0 + math.log(1.0 + len(token))

            # Primary hash bucket
            h_uni = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dimension
            vector[h_uni] += weight

            # Bigram feature for phrase semantics
            if i < len(tokens) - 1:
                next_tok = tokens[i + 1]
                if next_tok not in STOPWORDS:
                    bigram = f"{token}_{next_tok}"
                    h_bi = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16) % self.dimension
                    vector[h_bi] += weight * 2.0

            # Subword character 3-grams for content words
            if len(token) >= 3:
                for j in range(len(token) - 2):
                    trigram = token[j : j + 3]
                    h_tri = int(hashlib.sha1(trigram.encode("utf-8")).hexdigest(), 16) % self.dimension
                    vector[h_tri] += 0.8

        # L2-normalize vector
        normed = self._normalize(vector)
        return normed.tolist()

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single string (e.g. a user query or chunk)."""
        return self._generate_local_embedding(text)

    def embed_batch(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """
        Embeds a list of texts in batches to optimize performance.
        """
        results: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_vectors = [self._generate_local_embedding(t) for t in batch]
            results.extend(batch_vectors)
        return results
