# DocuMind AI - Lightweight PDF Chat Assistant

DocuMind AI is a production-ready conversational RAG (Retrieval-Augmented Generation) application built using **FastAPI**, **LangChain**, **Pinecone**, and **Groq**. 

It allows users to upload a PDF document (up to 10 pages) and ask questions about its content. It features page-count validation, isolated database namespaces for concurrent users, and memory-optimized configuration settings to run reliably on Render's free tier.

---

## Technical Stack & Optimization
- **Backend:** FastAPI (extremely low memory footprint, ~25MB RAM).
- **LLM Engine:** Groq (`llama-3.1-8b-instant` or similar) via LangChain `ChatGroq`.
- **Vector DB:** Pinecone (using unique namespaces per document to isolate chat sessions).
- **Embeddings:** Hugging Face serverless Inference API (uses 0MB local RAM to prevent Render OOM crashes).

---

## Local Setup & Run

### 1. Clone the repository and navigate to the directory
```bash
cd documind-ai
```

### 2. Set up virtual environment
**On Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install packages
```bash
pip install -r requirements.txt
```

Create a `.env` file in the root of the `documind-ai` folder (copied from `.env.example`):
```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=documind-index
HF_TOKEN=your_huggingface_hub_token

# Choose 'huggingface-api' for Render free tier or 'huggingface-local' for local GPU/CPU running
EMBEDDING_PROVIDER=huggingface-api

# Groq model choice: llama-3.1-8b-instant, mixtral-8x7b-32768, etc.
GROQ_MODEL=llama-3.1-8b-instant
```

### 5. Run the application
```bash
uvicorn app:app --reload
```
Open `http://127.0.0.1:8000` in your web browser.

---

