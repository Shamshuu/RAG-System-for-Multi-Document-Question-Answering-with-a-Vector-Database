"""
Unit tests for the Document Parser and Text Chunking service.
"""

from src.services.document import ExtractedPage, DocumentParser
from src.services.chunking import TextChunker, chunk_document


def test_chunking_preserves_page_numbers():
    pages = [
        ExtractedPage(
            document_id="doc-1",
            filename="policy.pdf",
            page_number=1,
            text="Page one contains company vision and mission statements."
        ),
        ExtractedPage(
            document_id="doc-1",
            filename="policy.pdf",
            page_number=2,
            text="Page two contains health benefits and 20 days PTO policy for full-time employees."
        )
    ]

    chunks = chunk_document(pages, chunk_size=500, overlap=100)
    assert len(chunks) == 2
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[0].metadata["filename"] == "policy.pdf"
    assert chunks[1].metadata["page_number"] == 2
    assert "PTO policy" in chunks[1].text


def test_sliding_window_overlap():
    # Long text exceeding chunk size
    long_text = "Word " * 300 # ~1500 chars
    pages = [
        ExtractedPage(
            document_id="doc-2",
            filename="manual.pdf",
            page_number=5,
            text=long_text
        )
    ]

    chunker = TextChunker(chunk_size=600, overlap=150)
    chunks = chunker.chunk_pages(pages)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["page_number"] == 5
        assert chunk.metadata["filename"] == "manual.pdf"
        assert len(chunk.text) <= 600
