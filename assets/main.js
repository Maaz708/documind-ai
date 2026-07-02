// State Variables
let activeDocId = null;
let chatHistory = [];

// DOM Elements
const uploadForm = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const dropzone = document.getElementById('dropzone');
const uploadBtn = document.getElementById('upload-btn');

const statusContainer = document.getElementById('status-container');
const statusText = document.getElementById('status-text');
const progressBarContainer = document.getElementById('status-progress');
const progressBar = document.getElementById('progress-bar');

const docInfoPanel = document.getElementById('doc-info');
const infoFilename = document.getElementById('info-filename');
const infoPages = document.getElementById('info-pages');
const infoId = document.getElementById('info-id');
const resetDocBtn = document.getElementById('reset-doc-btn');

const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');

// File dropzone visual effects
fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        document.querySelector('.file-label strong').textContent = file.name;
        document.querySelector('.file-label span').textContent = `Size: ${(file.size / 1024 / 1024).toFixed(2)} MB`;
    }
});

// Upload and Index Document
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (fileInput.files.length === 0) return;

    const file = fileInput.files[0];
    
    // Simple client side file type verification
    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endswith('.pdf')) {
        showStatus('Error: Only PDF files are accepted.', true);
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Disable form UI and show progress
    setUploadFormDisabled(true);
    showStatus('Uploading and parsing document details...', false);
    showProgress(25);

    try {
        // Send request
        showStatus('Extracting content and indexing pages to Pinecone...', false);
        showProgress(60);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Failed to upload and index document.');
        }

        showProgress(100);
        showStatus('Ingestion complete!', false);

        // Set active document state
        activeDocId = result.doc_id;
        chatHistory = []; // Reset chat history for the new document

        // Show active document details
        infoFilename.textContent = result.filename;
        infoPages.textContent = result.page_count;
        infoId.textContent = result.doc_id;
        
        uploadForm.classList.add('hidden');
        statusContainer.classList.add('hidden');
        docInfoPanel.classList.remove('hidden');

        // Setup chat area
        chatMessages.innerHTML = '';
        addMessage('system', `Successfully indexed <strong>${result.filename}</strong> (${result.page_count} pages). You can now ask questions about its content.`);
        chatForm.classList.remove('hidden');

    } catch (err) {
        showStatus(`Error: ${err.message}`, true);
        setUploadFormDisabled(false);
    }
});

// Reset Document (Clear & Upload New)
resetDocBtn.addEventListener('click', () => {
    activeDocId = null;
    chatHistory = [];
    
    // Reset forms
    fileInput.value = '';
    document.querySelector('.file-label strong').textContent = 'Choose a PDF file';
    document.querySelector('.file-label span').textContent = 'or drag and drop it here';
    
    uploadForm.classList.remove('hidden');
    docInfoPanel.classList.add('hidden');
    chatForm.classList.add('hidden');
    
    setUploadFormDisabled(false);
    
    chatMessages.innerHTML = '<div class="message system-message">Please upload a PDF document on the left to start chatting.</div>';
});

// Chat Question Submit
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = chatInput.value.trim();
    if (!question || !activeDocId) return;

    // Print user message immediately
    addMessage('user', question);
    chatInput.value = '';
    
    // Disable inputs
    chatInput.disabled = true;
    sendBtn.disabled = true;

    // Loading visual representation
    const tempLoaderMessage = addMessage('assistant', 'Thinking...', true);

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                doc_id: activeDocId,
                question: question,
                history: chatHistory
            })
        });

        const result = await response.json();

        // Remove loader
        tempLoaderMessage.remove();

        if (!response.ok) {
            throw new Error(result.detail || 'Error communicating with RAG endpoint.');
        }

        // Print response
        addMessage('assistant', result.answer, false, result.sources);

        // Update local chat history memory
        chatHistory.push({ role: 'user', content: question });
        chatHistory.push({ role: 'assistant', content: result.answer });

    } catch (err) {
        tempLoaderMessage.remove();
        addMessage('assistant', `Failed to retrieve answer. Error: ${err.message}`);
    } finally {
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
        scrollToBottom();
    }
});

// Helper Functions
function setUploadFormDisabled(disabled) {
    fileInput.disabled = disabled;
    uploadBtn.disabled = disabled;
}

function showStatus(text, isError = false) {
    statusContainer.classList.remove('hidden');
    statusText.textContent = text;
    if (isError) {
        statusText.style.color = '#e53e3e';
        progressBarContainer.classList.add('hidden');
    } else {
        statusText.style.color = '#2b6cb0';
    }
}

function showProgress(percent) {
    progressBarContainer.classList.remove('hidden');
    progressBar.style.width = `${percent}%`;
}

function addMessage(sender, text, isLoader = false, sources = null) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', `${sender}-message`);
    
    if (isLoader) {
        msgDiv.id = 'chat-typing-loader';
    }
    
    // Render content safely
    msgDiv.innerHTML = text;

    // Append sources if provided
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.classList.add('sources-container');
        
        const title = document.createElement('div');
        title.classList.add('sources-title');
        title.textContent = 'Sources Referenced:';
        sourcesDiv.appendChild(title);

        // Deduplicate pages to show list cleanly
        const pageSet = new Set();
        sources.forEach(s => pageSet.add(s.page));
        const pagesStr = Array.from(pageSet).sort((a,b) => a - b).map(p => `Page ${p}`).join(', ');

        const pagesSpan = document.createElement('span');
        pagesSpan.innerHTML = `Retrieved matching blocks from: <strong>${pagesStr}</strong>`;
        sourcesDiv.appendChild(pagesSpan);
        
        msgDiv.appendChild(sourcesDiv);
    }

    chatMessages.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
