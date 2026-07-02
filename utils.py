import os
import re
import hashlib
from pypdf import PdfReader
from config import MAX_PDF_PAGES

def compute_file_hash(file_path: str) -> str:
    """
    Computes a SHA-256 hash of the file contents.
    Used to detect duplicate uploads and skip re-ingestion.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a filename to prevent directory traversal and remove invalid characters.
    """
    # Get only the base name
    filename = os.path.basename(filename)
    # Replace non-alphanumeric characters (except dots, dashes, underscores) with underscores
    filename = re.sub(r'[^a-zA-Z0-9_\.-]', '_', filename)
    return filename

def get_pdf_page_count(file_path: str) -> int:
    """
    Reads the PDF file and returns its page count.
    Raises a ValueError if parsing fails.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        reader = PdfReader(file_path)
        return len(reader.pages)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF file. The file may be corrupt or invalid. Error: {str(e)}")

def validate_pdf(file_path: str) -> bool:
    """
    Checks if the PDF is valid and has at most MAX_PDF_PAGES.
    Raises ValueError if validations fail.
    """
    if not file_path.lower().endswith(".pdf"):
        raise ValueError("Unsupported file format. Only PDF files are allowed.")
        
    page_count = get_pdf_page_count(file_path)
    if page_count > MAX_PDF_PAGES:
        raise ValueError(
            f"The uploaded document has {page_count} pages, "
            f"which exceeds the maximum limit of {MAX_PDF_PAGES} pages."
        )
    return True
