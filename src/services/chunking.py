"""
Text Chunking Service
Splits extracted document pages into overlapping chunks while preserving page and source metadata.
"""

import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List
from src.services.document import ExtractedPage


@dataclass
class TextChunk:
    """Represents a discrete chunk of text ready for vector embedding and storage."""
    id: str
    text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata
        }


class TextChunker:
    """Splits extracted document pages into configurable overlapping chunks."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        if overlap >= chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_pages(self, pages: List[ExtractedPage]) -> List[TextChunk]:
        """
        Takes a list of ExtractedPage objects and returns a list of TextChunk objects.
        Every chunk retains its original filename, document_id, and exact page_number.
        """
        all_chunks: List[TextChunk] = []
        global_chunk_idx = 0

        for page in pages:
            text = page.text.strip()
            if not text:
                continue

            # If page text fits in a single chunk
            if len(text) <= self.chunk_size:
                chunk_id = f"{page.document_id}-p{page.page_number}-c{global_chunk_idx}"
                metadata = {
                    "document_id": page.document_id,
                    "filename": page.filename,
                    "page_number": page.page_number,
                    "chunk_index": global_chunk_idx,
                    "start_char": 0,
                    "end_char": len(text),
                    "text": text
                }
                all_chunks.append(TextChunk(id=chunk_id, text=text, metadata=metadata))
                global_chunk_idx += 1
                continue

            # Sliding window for text longer than chunk_size
            start = 0
            text_length = len(text)
            step = self.chunk_size - self.overlap

            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunk_id = f"{page.document_id}-p{page.page_number}-c{global_chunk_idx}"
                    metadata = {
                        "document_id": page.document_id,
                        "filename": page.filename,
                        "page_number": page.page_number,
                        "chunk_index": global_chunk_idx,
                        "start_char": start,
                        "end_char": end,
                        "text": chunk_text
                    }
                    all_chunks.append(TextChunk(id=chunk_id, text=chunk_text, metadata=metadata))
                    global_chunk_idx += 1

                start += step
                if end == text_length:
                    break

        return all_chunks


def chunk_document(
    pages: List[ExtractedPage], 
    chunk_size: int = 1000, 
    overlap: int = 200
) -> List[TextChunk]:
    """Convenience helper function to chunk extracted document pages."""
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
    return chunker.chunk_pages(pages)
