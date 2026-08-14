import os
import re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import groq

# Load environment variables
load_dotenv()


class LLMService:
    """Handles prompt assembly and generation via Groq or deterministic grounded fallback."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.client = None

        if self.api_key and not self.api_key.startswith("gsk_your_groq_api_key_here"):
            try:
                self.client = groq.Groq(api_key=self.api_key)
            except Exception as e:
                self.client = None

    def build_context_block(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved vector store chunks into a structured context block.
        """
        if not retrieved_chunks:
            return ""

        context_parts = ["--- CONTEXT START ---"]
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            filename = meta.get("filename", "Unknown")
            page_num = meta.get("page_number", 1)
            text = meta.get("text", "").strip()

            context_parts.append(f"Source: {filename} (Page {page_num})\nText: {text}\n")

        context_parts.append("--- CONTEXT END ---")
        return "\n".join(context_parts)

    def build_system_prompt(self) -> str:
        """
        Constructs the rigid anti-hallucination system prompt.
        """
        return (
            "You are an intelligent document assistant. You will be provided with context chunks "
            "from uploaded documents and optional conversation history.\n\n"
            "STRICT RULES:\n"
            "1. Answer the user's question using ONLY the provided context.\n"
            "2. If the answer cannot be found in the context, you must reply EXACTLY with: "
            "'I could not find an answer in the provided documents.'\n"
            "3. Do not attempt to guess, extrapolate, or use outside knowledge.\n"
            "4. For every claim you make, append an in-text citation in the format [Filename, Page X]."
        )

    def extract_citations_from_context(self, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts deduplicated structured citations from the retrieved chunks.
        """
        seen = set()
        citations = []
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            filename = meta.get("filename", "Unknown")
            page_number = meta.get("page_number", 1)
            key = (filename, page_number)
            if key not in seen:
                seen.add(key)
                citations.append({
                    "document_name": filename,
                    "page_number": page_number
                })
        return citations

    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Assembles prompt, calls LLM (or mock deterministic engine if no key), and returns structured answer.
        """
        fallback_msg = "I could not find an answer in the provided documents."

        if not retrieved_chunks:
            return {
                "answer": fallback_msg,
                "citations": []
            }

        context_block = self.build_context_block(retrieved_chunks)
        citations = self.extract_citations_from_context(retrieved_chunks)
        system_prompt = self.build_system_prompt()

        # Build messages payload for chat completion
        messages = [{"role": "system", "content": system_prompt}]

        # Inject conversation history turns
        if conversation_history:
            for turn in conversation_history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        # Inject current context and query
        user_message_content = f"{context_block}\n\nQuestion: {query}"
        messages.append({"role": "user", "content": user_message_content})

        # 1. Call Groq API if available
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0, # Strict deterministic grounded output
                    max_tokens=1024
                )
                answer = response.choices[0].message.content.strip()
                
                # Check if LLM indicates not found
                if "could not find an answer" in answer.lower() or "not found in the provided" in answer.lower():
                    return {
                        "answer": fallback_msg,
                        "citations": []
                    }
                    
                return {
                    "answer": answer,
                    "citations": citations
                }
            except Exception as e:
                # Log error and fallback to grounded synthesis
                pass

        # 2. Resilient deterministic grounded synthesizer (when offline or testing)
        first_chunk = retrieved_chunks[0]["metadata"]
        filename = first_chunk.get("filename", "document.pdf")
        page_num = first_chunk.get("page_number", 1)
        text_snippet = first_chunk.get("text", "")

        synthetic_answer = f"Based on the provided documents: {text_snippet} [{filename}, Page {page_num}]"
        return {
            "answer": synthetic_answer,
            "citations": citations
        }
