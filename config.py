import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "documind-index")
HF_TOKEN = os.getenv("HF_TOKEN")

# Model Configuration
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Embedding Configuration
# Allowed: 'huggingface-api', 'huggingface-local', 'google-api'
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface-api").lower()

# App Constraints
MAX_PDF_PAGES = 10
UPLOAD_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "uploads")

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

def validate_config():
    """
    Validates that the environment variables required for the current configuration are present.
    """
    missing = []
    
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
        
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
        
    if EMBEDDING_PROVIDER == "huggingface-api" and not HF_TOKEN:
        missing.append("HF_TOKEN")
        
    if EMBEDDING_PROVIDER == "google-api" and not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
        
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Please update your .env file."
        )
