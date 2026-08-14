"""
Relational Database Module using SQLite for session history, messages, and document registry.
Handles conversation turns, message sequential ordering, and document metadata.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class RelationalDB:
    """Manages SQLite storage for chat history, sessions, and uploaded documents."""

    def __init__(self, db_path: str = "./rag_history.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a configured SQLite connection with foreign keys and row factory enabled."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self) -> None:
        """Initializes tables for documents, sessions, and messages."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Documents Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    mime_type TEXT NOT NULL,
                    total_chunks INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Sessions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT DEFAULT 'New Conversation',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Messages Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    citations_json TEXT DEFAULT '[]',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
            """)

            # Index on session_id and created_at for fast sequential lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_created 
                ON messages(session_id, created_at ASC);
            """)
            conn.commit()
        finally:
            conn.close()

    # --- Document Management ---

    def create_document(
        self,
        doc_id: str,
        filename: str,
        file_size: int,
        mime_type: str,
        total_chunks: int = 0
    ) -> Dict[str, Any]:
        """Registers an ingested document in the database."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute("""
                INSERT OR REPLACE INTO documents (id, filename, file_size, mime_type, total_chunks, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (doc_id, filename, file_size, mime_type, total_chunks, now))
            conn.commit()
            return {
                "id": doc_id,
                "filename": filename,
                "file_size": file_size,
                "mime_type": mime_type,
                "total_chunks": total_chunks,
                "created_at": now
            }
        finally:
            conn.close()

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns all registered documents."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY created_at DESC;")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific document by its unique ID."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?;", (doc_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document record from SQLite."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE id = ?;", (doc_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    # --- Session Management ---

    def create_session(self, session_id: Optional[str] = None, title: str = "New Conversation") -> Dict[str, Any]:
        """Creates a new chat session."""
        session_id = session_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, title, now, now))
            conn.commit()
            return {
                "id": session_id,
                "title": title,
                "created_at": now,
                "updated_at": now
            }
        finally:
            conn.close()

    def get_or_create_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieves an existing session or creates a new one if not found."""
        if session_id:
            existing = self.get_session(session_id)
            if existing:
                return existing
            return self.create_session(session_id=session_id)
        return self.create_session()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a session by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?;", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists all chat sessions sorted by updated_at descending."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions ORDER BY updated_at DESC;")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    # --- Message Management ---

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Saves a conversation turn (user or assistant) and updates the session timestamp."""
        # Ensure session exists
        self.get_or_create_session(session_id)
        
        msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        citations_json = json.dumps(citations or [])

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages (id, session_id, role, content, citations_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (msg_id, session_id, role, content, citations_json, now))
            
            cursor.execute("""
                UPDATE sessions SET updated_at = ? WHERE id = ?
            """, (now, session_id))
            conn.commit()

            return {
                "id": msg_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "citations": citations or [],
                "created_at": now
            }
        finally:
            conn.close()

    def get_session_history(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the last `limit` message turns for a given session.
        Returns messages in chronological order (oldest to newest) for prompt context assembly.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM (
                    SELECT * FROM messages 
                    WHERE session_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ) ORDER BY created_at ASC;
            """, (session_id, limit))
            rows = cursor.fetchall()
            
            history = []
            for row in rows:
                item = dict(row)
                try:
                    item["citations"] = json.loads(item.get("citations_json", "[]"))
                except Exception:
                    item["citations"] = []
                history.append(item)
            return history
        finally:
            conn.close()
