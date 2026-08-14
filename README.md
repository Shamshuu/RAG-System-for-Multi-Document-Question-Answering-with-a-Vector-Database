# Multi-Document Retrieval-Augmented Generation (RAG) System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Cloud-F05032.svg)](https://groq.com/)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(9%2F9)-brightgreen.svg)]()

A production-ready **Retrieval-Augmented Generation (RAG)** system capable of ingesting multiple document formats (`.pdf`, `.docx`, `.txt`), preserving exact page numbers, performing semantic cosine similarity search, enforcing anti-hallucination guardrails, maintaining multi-turn conversational history, and returning verifiable source citations.

---

## 🏛️ Architectural Overview

The system strictly decouples the asynchronous **Document Ingestion Pipeline** from the synchronous **Query Pipeline**, supported by a dual-database architecture (Relational DB for conversation/document state + Vector DB for dense semantic embeddings).

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                            INGESTION PIPELINE                               │
 └─────────────────────────────────────────────────────────────────────────────┘
   [PDF / DOCX File]
          │
          ▼
   [Document Parser]        ──> Extracts raw text, preserving 1-indexed page_number
          │
          ▼
   [Text Chunker]           ──> Sliding window (1000 chars, 200 overlap) + Page Metadata
          │
          ▼
   [Embedding Engine]       ──> 384-dimensional dense vectors with L2 normalization
          │
          ├────────────────────────────────┬────────────────────────────────┐
          ▼                                                                 ▼
   [Vector Database]                                               [Relational DB]
   Stores vector + metadata                                        Registers document record
   (doc_id, page_number, text)                                     (id, filename, chunks, size)

 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                              QUERY PIPELINE                                 │
 └─────────────────────────────────────────────────────────────────────────────┘
   [User Question + Optional Session ID]
          │
          ▼
   [Query Contextualizer]   ──> Multi-turn query enrichment for pronoun resolution
          │
          ▼
   [Query Embedder]         ──> Exact identical embedding vector space
          │
          ▼
   [Vector Cosine Search]   ──> Computes cos(θ) = û · v̂ across all chunks
          │
          ├────────────────────────────────────────┐
          │ (Score < Threshold / 0 Chunks)         │ (Score >= Threshold)
          ▼                                        ▼
   [Graceful Failure]                       [Fetch History (Relational DB)]
   "I could not find an answer              [Build Grounded System Prompt]
    in the provided documents."                    │
                                                   ▼
                                            [Groq LLM Engine]
                                            (temperature=0.0)
                                                   │
                                                   ▼
                                            [Structured Citations Extractor]
                                            [Persist Turns in Relational DB]
                                                   │
                                                   ▼
                                            [Final JSON Payload]
                                            { answer, citations: [...], session_id }
```

---

## 🔑 Key Architectural & Design Decisions

### 1. Dual-Database Strategy
* **Relational Database (SQLite)**: Acts as the source of truth for sequential conversation state (`ORDER BY created_at ASC`), foreign keys between sessions and message turns, and document ingestion logs.
* **Vector Database**: Optimized purely for high-dimensional geometric nearest-neighbor calculations and metadata filtering.

### 2. Page-Aware Sliding Window Chunking
* **Strategy**: `chunk_size = 1000` characters, `chunk_overlap = 200` characters.
* **Preservation**: Every chunk maintains strict lineage metadata:
  ```json
  {
    "document_id": "doc-uuid",
    "filename": "employee_handbook.pdf",
    "page_number": 14,
    "chunk_index": 3,
    "text": "Employees receive 20 days of paid time off..."
  }
  ```
* **Why Overlap?** Overlap prevents cutting sentences or key clauses in half across chunk boundaries.

### 3. Anti-Hallucination & Deterministic Guardrails
1. **Geometric Gating**: If retrieved vector similarity scores fall below the threshold (e.g. `< 0.25`), the LLM is bypassed entirely, immediately returning `"I could not find an answer in the provided documents."` with `citations: []`.
2. **Zero-Temperature Grounding**: The LLM runs at `temperature=0.0` with explicit system constraints forbidding outside world knowledge.
3. **Structured Citation Tracking**: Programmatically formats and extracts `[document_name, page_number]` so client applications can render direct deep links.

---

## ⚡ Quickstart & Setup Instructions

### 1. Prerequisites
* **Python 3.10+**
* (Optional) **Groq API Key** for LLM generation (the system includes an offline grounded fallback mode if no API key is supplied).

### 2. Clone & Install Dependencies
```bash
git clone https://github.com/Shamshuu/RAG-System-for-Multi-Document-Question-Answering-with-a-Vector-Database.git
cd RAG-System-for-Multi-Document-Question-Answering-with-a-Vector-Database

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Configure your `.env`:
```ini
PORT=8000
HOST=0.0.0.0
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
SIMILARITY_THRESHOLD=0.25
TOP_K=5
DATABASE_PATH=./rag_history.db
```

### 4. Run the Server
```bash
python -m src.main
```
The server will start at **`http://localhost:8000`**.
* Interactive Web UI: **`http://localhost:8000/`**
* Interactive Swagger API Docs: **`http://localhost:8000/docs`**

---

## 🧪 Automated Testing

Execute the comprehensive unit and integration test suite:
```bash
pytest tests/ -v
```

### Test Coverage Highlights:
* `tests/test_chunking.py`: Verifies sliding window boundaries, overlap, and page number tagging.
* `tests/test_vector_store.py`: Verifies vector normalization, cosine nearest-neighbor search, and threshold gating.
* `tests/test_relational.py`: Verifies SQLite documents, sessions, and chronological message history.
* `tests/test_rag.py`: Verifies end-to-end multi-turn Q&A, pronoun resolution, and out-of-scope graceful failure.
* `tests/test_api.py`: Verifies FastAPI `/api/upload` and `/api/chat` endpoints.

---

## 📡 REST API Reference

### 1. Ingest Documents
**`POST /api/upload`** (Multipart Form)

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "files=@samples/contractor_guidelines.docx"
```

**Response (`200 OK`):**
```json
{
  "status": "success",
  "processed_files": [
    {
      "document_id": "8f3b2a1c-...",
      "filename": "contractor_guidelines.docx",
      "file_size": 37240,
      "total_pages": 1,
      "total_chunks": 2,
      "status": "success"
    }
  ],
  "total_processed": 1
}
```

---

### 2. Query Chat Pipeline
**`POST /api/chat`** (JSON)

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the payment terms for contractors?",
    "session_id": "optional-session-id"
  }'
```

**Response (`200 OK`):**
```json
{
  "answer": "Payment terms for contractors are strictly Net 30 from the date of invoice approval [contractor_guidelines.docx, Page 1].",
  "citations": [
    {
      "document_name": "contractor_guidelines.docx",
      "page_number": 1
    }
  ],
  "session_id": "3d91be24-7e50-482a-a92c-56eb0d19b782"
}
```

---

### 3. Graceful Failure (Anti-Hallucination)
When querying an unmentioned topic:
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital city of France?"
  }'
```

**Response (`200 OK`):**
```json
{
  "answer": "I could not find an answer in the provided documents.",
  "citations": [],
  "session_id": "3d91be24-7e50-482a-a92c-56eb0d19b782"
}
```

---

### 4. List Documents
**`GET /api/documents`**

```bash
curl -X GET "http://localhost:8000/api/documents"
```

---

## 📂 Project Structure

```
.
├── src/
│   ├── api/
│   │   └── routes.py         # FastAPI endpoints for upload, chat, sessions
│   ├── core/
│   │   └── rag.py            # End-to-end RAG orchestrator & pipeline logic
│   ├── services/
│   │   ├── document.py       # PDF/DOCX parsers with page preservation
│   │   ├── chunking.py       # Sliding-window overlapping text chunker
│   │   ├── embedding.py      # Normalized vector embedding service
│   │   └── llm.py            # Grounded prompt builder & Groq LLM integration
│   ├── database/
│   │   ├── relational.py     # SQLite documents, sessions & messages repository
│   │   └── vector_store.py   # Vector DB with Cosine similarity & threshold filter
│   └── main.py               # Application entry point & web server
├── static/
│   ├── index.html            # Modern interactive Web UI
│   ├── style.css             # Glassmorphic dark styling
│   └── app.js                # UI event handling & chat streaming
├── samples/
│   ├── contractor_guidelines.docx
│   └── generate_samples.py
├── tests/
│   ├── test_api.py
│   ├── test_chunking.py
│   ├── test_rag.py
│   ├── test_relational.py
│   └── test_vector_store.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🛡️ License
MIT License. Built for enterprise-grade trustworthy AI applications.