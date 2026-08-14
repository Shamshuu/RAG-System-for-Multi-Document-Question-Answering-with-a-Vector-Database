// Multi-Document RAG Client Logic

let currentSessionId = localStorage.getItem("rag_session_id") || null;

// DOM Elements
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const uploadPrompt = document.getElementById("uploadPrompt");
const uploadProgress = document.getElementById("uploadProgress");
const uploadStatusText = document.getElementById("uploadStatusText");
const docList = document.getElementById("docList");
const docCount = document.getElementById("docCount");
const refreshDocsBtn = document.getElementById("refreshDocsBtn");
const newChatBtn = document.getElementById("newChatBtn");
const sessionDisplay = document.getElementById("sessionDisplay");
const chatForm = document.getElementById("chatForm");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const messagesContainer = document.getElementById("messagesContainer");
const toast = document.getElementById("toast");
const quickScenarios = document.getElementById("quickScenarios");

// --- Initialization ---
document.addEventListener("DOMContentLoaded", () => {
    updateSessionUI();
    fetchDocuments();
    setupEventListeners();
});

function setupEventListeners() {
    // Dropzone upload
    dropZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", handleFileSelect);

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            uploadFiles(e.dataTransfer.files);
        }
    });

    // Refresh docs
    refreshDocsBtn.addEventListener("click", fetchDocuments);

    // New Chat
    newChatBtn.addEventListener("click", startNewSession);

    // Chat form submit
    chatForm.addEventListener("submit", handleChatSubmit);

    // Textarea enter-to-send (Shift+Enter for newline)
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // Quick scenarios click
    quickScenarios.addEventListener("click", (e) => {
        const pill = e.target.closest(".scenario-pill");
        if (pill && pill.dataset.query) {
            queryInput.value = pill.dataset.query;
            chatForm.dispatchEvent(new Event("submit"));
        }
    });
}

function showToast(message, isError = false) {
    toast.textContent = message;
    toast.style.borderColor = isError ? "var(--danger)" : "var(--border-highlight)";
    toast.classList.add("show");
    setTimeout(() => {
        toast.classList.remove("show");
    }, 4000);
}

function updateSessionUI() {
    if (currentSessionId) {
        sessionDisplay.innerHTML = `Session: <strong style="color: var(--primary);">${currentSessionId.substring(0, 8)}...</strong>`;
    } else {
        sessionDisplay.innerHTML = "Session: <em>New</em>";
    }
}

function startNewSession() {
    currentSessionId = null;
    localStorage.removeItem("rag_session_id");
    updateSessionUI();
    
    // Clear chat window except welcome card
    messagesContainer.innerHTML = `
        <div class="message assistant-message welcome-card">
            <div class="avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <path d="M12 16v-4M12 8h.01"></path>
                </svg>
            </div>
            <div class="message-body">
                <h3>New Conversation Started</h3>
                <p>Ask a question about your uploaded documents or use the quick scenario buttons above.</p>
            </div>
        </div>
    `;
    showToast("Started a fresh conversation session.");
}

// --- Document Upload ---
function handleFileSelect(e) {
    if (e.target.files && e.target.files.length > 0) {
        uploadFiles(e.target.files);
    }
}

async function uploadFiles(fileList) {
    const formData = new FormData();
    for (let i = 0; i < fileList.length; i++) {
        formData.append("files", fileList[i]);
    }

    uploadPrompt.style.display = "none";
    uploadProgress.style.display = "block";
    uploadStatusText.textContent = `Ingesting ${fileList.length} file(s)...`;

    try {
        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        if (response.ok) {
            showToast(`Successfully indexed ${data.total_processed} document(s)!`);
            fetchDocuments();
        } else {
            showToast(data.detail || "Failed to upload document", true);
        }
    } catch (err) {
        showToast("Error uploading file: " + err.message, true);
    } finally {
        fileInput.value = "";
        uploadPrompt.style.display = "block";
        uploadProgress.style.display = "none";
    }
}

// --- Fetch & Render Documents ---
async function fetchDocuments() {
    try {
        const response = await fetch("/api/documents");
        const data = await response.json();
        renderDocuments(data.documents || []);
    } catch (err) {
        console.error("Error fetching documents:", err);
    }
}

function renderDocuments(docs) {
    docCount.textContent = docs.length;
    if (docs.length === 0) {
        docList.innerHTML = `<div class="empty-hint">No documents indexed yet. Upload a file above to begin.</div>`;
        return;
    }

    docList.innerHTML = docs.map(doc => `
        <div class="doc-item">
            <div class="doc-info">
                <span class="doc-name" title="${doc.filename}">${doc.filename}</span>
                <span class="doc-meta">${doc.total_chunks} chunks • ${(doc.file_size / 1024).toFixed(1)} KB</span>
            </div>
            <button class="btn-delete-doc" onclick="deleteDocument('${doc.id}')" title="Delete document">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="3 6 5 6 21 6"></polyline>
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                </svg>
            </button>
        </div>
    `).join("");
}

async function deleteDocument(docId) {
    if (!confirm("Are you sure you want to delete this document from the vector store?")) return;
    try {
        const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
        if (res.ok) {
            showToast("Document deleted.");
            fetchDocuments();
        } else {
            showToast("Failed to delete document", true);
        }
    } catch (err) {
        showToast("Error deleting document: " + err.message, true);
    }
}

// --- Chat & Response Rendering ---
async function handleChatSubmit(e) {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    // Append User Message
    appendUserMessage(query);
    queryInput.value = "";
    sendBtn.disabled = true;

    // Append Assistant Loading Bubble
    const loadingBubbleId = appendLoadingBubble();

    try {
        const payload = {
            query: query,
            session_id: currentSessionId
        };

        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        // Remove loading bubble
        removeLoadingBubble(loadingBubbleId);

        if (response.ok) {
            // Update session ID
            if (data.session_id) {
                currentSessionId = data.session_id;
                localStorage.setItem("rag_session_id", currentSessionId);
                updateSessionUI();
            }

            appendAssistantMessage(data.answer, data.citations || []);
        } else {
            appendAssistantMessage("Error: " + (data.detail || "Failed to retrieve answer."), []);
        }
    } catch (err) {
        removeLoadingBubble(loadingBubbleId);
        appendAssistantMessage("Network error: " + err.message, []);
    } finally {
        sendBtn.disabled = false;
        queryInput.focus();
    }
}

function appendUserMessage(text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message user-message";
    msgDiv.innerHTML = `
        <div class="avatar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
            </svg>
        </div>
        <div class="message-body">${escapeHtml(text)}</div>
    `;
    messagesContainer.appendChild(msgDiv);
    scrollToBottom();
}

function appendLoadingBubble() {
    const id = "loading-" + Date.now();
    const msgDiv = document.createElement("div");
    msgDiv.id = id;
    msgDiv.className = "message assistant-message";
    msgDiv.innerHTML = `
        <div class="avatar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M12 16v-4M12 8h.01"></path>
            </svg>
        </div>
        <div class="message-body" style="display: flex; align-items: center; gap: 10px;">
            <div class="spinner" style="margin: 0; width: 18px; height: 18px;"></div>
            <span style="color: var(--text-muted); font-size: 0.85rem;">Searching vector space & generating grounded response...</span>
        </div>
    `;
    messagesContainer.appendChild(msgDiv);
    scrollToBottom();
    return id;
}

function removeLoadingBubble(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function appendAssistantMessage(answer, citations = []) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant-message";

    let citationsHtml = "";
    if (citations && citations.length > 0) {
        citationsHtml = `
            <div class="citations-box">
                <div class="citations-title">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                    </svg>
                    Verified Citations (${citations.length})
                </div>
                <div class="citations-list">
                    ${citations.map(c => `
                        <span class="citation-chip">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
                                <polyline points="13 2 13 9 20 9"></polyline>
                            </svg>
                            ${escapeHtml(c.document_name)} • Page ${c.page_number}
                        </span>
                    `).join("")}
                </div>
            </div>
        `;
    }

    msgDiv.innerHTML = `
        <div class="avatar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M12 16v-4M12 8h.01"></path>
            </svg>
        </div>
        <div class="message-body">
            <div class="answer-text">${escapeHtml(answer)}</div>
            ${citationsHtml}
        </div>
    `;
    messagesContainer.appendChild(msgDiv);
    scrollToBottom();
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function escapeHtml(text) {
    if (!text) return "";
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
