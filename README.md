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

## Render Deployment Guide (Free Tier)

Render's free tier provides 512MB RAM. Running PyTorch locally consumes ~400MB+ RAM and will crash the web service. Follow these settings for a successful deployment:

1. **Create Web Service:**
   - Link your GitHub repository to Render.
   - Choose **Python** as the runtime environment.

2. **Configure Build & Start Commands:**
   - **Root Directory:** `documind-ai` (if your project sits inside a subdirectory, otherwise leave blank).
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`

3. **Add Environment Variables (in Render Settings Dashboard):**
   - `GROQ_API_KEY`: *(Get one from Groq Console)*
   - `PINECONE_API_KEY`: *(Get one from Pinecone console)*
   - `PINECONE_INDEX_NAME`: `documind-index`
   - `HF_TOKEN`: *(Get a free read-access Token from Hugging Face Settings)*
   - `EMBEDDING_PROVIDER`: `huggingface-api` *(Crucial: prevents Render OOM crashes by using serverless embeddings)*
   - `GROQ_MODEL`: `llama-3.1-8b-instant`
   - `PYTHON_VERSION`: `3.10` or higher

4. **Deploy:** Render will build the packages and start the FastAPI web server. Since we serve our assets directly via FastAPI, your app will be accessible directly via the URL provided by Render!
