import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
from utils import sanitize_filename, validate_pdf, get_pdf_page_count, compute_file_hash
from ingest import ingest_pdf
from rag import query_rag

# In-memory cache: maps file content hash -> {doc_id, filename, page_count}
# Prevents duplicate ingestion when the same PDF is uploaded again
_upload_cache: dict[str, dict] = {}

# Initialize FastAPI App
app = FastAPI(
    title="DocuMind AI API",
    description="Production-ready REST API for Document QA RAG system.",
    version="1.0.0"
)

# CORS configuration for local development and client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schemas for request validation
class ChatRequest(BaseModel):
    doc_id: str
    question: str
    history: list = []

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a PDF file, performs validation (file type, max 10 pages),
    and indexes the document chunks into Pinecone in a unique namespace.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
        
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    # Standardize filename and create a temporary document ID for saving
    safe_filename = sanitize_filename(file.filename)
    temp_id = uuid.uuid4().hex
    
    # Save the file to local disk temporarily for parsing
    temp_path = os.path.join(config.UPLOAD_DIR, f"{temp_id}_{safe_filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Validate file size & page limit
        validate_pdf(temp_path)
        page_count = get_pdf_page_count(temp_path)
        
        # Check for duplicate upload using file content hash
        file_hash = compute_file_hash(temp_path)
        if file_hash in _upload_cache:
            cached = _upload_cache[file_hash]
            print(f"[DUPLICATE] File already indexed as doc_id={cached['doc_id']}. Skipping re-ingestion.")
            return {
                "status": "success",
                "doc_id": cached["doc_id"],
                "filename": cached["filename"],
                "page_count": cached["page_count"],
                "cached": True
            }
        
        # New file — assign a fresh doc_id and ingest
        doc_id = temp_id
        ingest_pdf(temp_path, doc_id)
        
        # Store in cache for future duplicate checks
        _upload_cache[file_hash] = {
            "doc_id": doc_id,
            "filename": safe_filename,
            "page_count": page_count
        }
        
        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": safe_filename,
            "page_count": page_count
        }
        
    except ValueError as val_err:
        # Catch explicit validation issues (e.g. exceeded page counts)
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        print(f"[SERVER ERROR] Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        # Safely clean up local temporary files
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_err:
                print(f"[WARNING] Failed to clean up temp file {temp_path}: {str(cleanup_err)}")

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Processes follow-up questions from users using vector search context
    from the isolated namespace matching the doc_id.
    """
    if not request.doc_id or not request.question:
        raise HTTPException(status_code=400, detail="doc_id and question parameters are required.")
        
    try:
        response_data = query_rag(request.doc_id, request.question, request.history)
        return {
            "status": "success",
            "answer": response_data["answer"],
            "sources": response_data["sources"]
        }
    except Exception as e:
        print(f"[SERVER ERROR] Chat execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")

# Ensure static assets directories exist
assets_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets")
os.makedirs(assets_dir, exist_ok=True)

# Mount Assets folder for serving CSS, JS and media files
app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

@app.get("/")
async def serve_index():
    """
    Serves the primary UI interface.
    """
    index_path = os.path.join(assets_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "status": "online",
        "message": "DocuMind AI Server is running. UI is not loaded. Please create assets/index.html."
    }
