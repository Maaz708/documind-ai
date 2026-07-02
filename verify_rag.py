import os
import unittest
from unittest.mock import patch, MagicMock

# Set mock env variables for configuration importing
os.environ["GROQ_API_KEY"] = "mock-groq-key"
os.environ["GOOGLE_API_KEY"] = "mock-google-key"
os.environ["PINECONE_API_KEY"] = "mock-pinecone-key"
os.environ["PINECONE_INDEX_NAME"] = "mock-index"
os.environ["HF_TOKEN"] = "mock-hf-token"
os.environ["EMBEDDING_PROVIDER"] = "huggingface-api"

import config
from utils import sanitize_filename, validate_pdf
import ingest

class TestDocuMindRAG(unittest.TestCase):
    
    def test_sanitize_filename(self):
        """Tests that filenames are sanitized correctly to prevent exploits."""
        self.assertEqual(sanitize_filename("test file.pdf"), "test_file.pdf")
        self.assertEqual(sanitize_filename("../../../etc/passwd"), "passwd")
        self.assertEqual(sanitize_filename("document-v1.0.pdf"), "document-v1.0.pdf")
        self.assertEqual(sanitize_filename("abc!@#def.pdf"), "abc___def.pdf")

    def test_config_validation_success(self):
        """Verifies configuration validates correctly when all keys exist."""
        # This should not raise an exception
        try:
            config.validate_config()
        except ValueError as e:
            self.fail(f"validate_config() failed unexpectedly: {e}")

    @patch("config.GROQ_API_KEY", None)
    def test_config_validation_missing_groq(self):
        """Verifies validation raises ValueError if GROQ_API_KEY is missing."""
        with self.assertRaises(ValueError) as context:
            config.validate_config()
        self.assertIn("GROQ_API_KEY", str(context.exception))

    @patch("config.EMBEDDING_PROVIDER", "google-api")
    @patch("config.GOOGLE_API_KEY", None)
    def test_config_validation_missing_google_embeddings(self):
        """Verifies validation raises ValueError if GOOGLE_API_KEY is missing for google-api embeddings."""
        with self.assertRaises(ValueError) as context:
            config.validate_config()
        self.assertIn("GOOGLE_API_KEY", str(context.exception))

    @patch("config.PINECONE_API_KEY", None)
    def test_config_validation_missing_pinecone(self):
        """Verifies validation raises ValueError if PINECONE_API_KEY is missing."""
        with self.assertRaises(ValueError) as context:
            config.validate_config()
        self.assertIn("PINECONE_API_KEY", str(context.exception))

    @patch("config.EMBEDDING_PROVIDER", "huggingface-api")
    @patch("config.HF_TOKEN", None)
    def test_config_validation_missing_hf_token(self):
        """Verifies validation raises ValueError if HF_TOKEN is missing for huggingface-api."""
        with self.assertRaises(ValueError) as context:
            config.validate_config()
        self.assertIn("HF_TOKEN", str(context.exception))

    @patch("utils.PdfReader")
    def test_validate_pdf_page_limit_exceeded(self, mock_pdf_reader):
        """Verifies that PDFs with >10 pages are rejected."""
        # Mock PdfReader to return 12 pages
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [MagicMock()] * 12
        mock_pdf_reader.return_value = mock_reader_instance
        
        # We need a file that exists
        temp_file = "temp_mock_test.pdf"
        with open(temp_file, "w") as f:
            f.write("%PDF-1.4 mock content")
            
        try:
            with self.assertRaises(ValueError) as context:
                validate_pdf(temp_file)
            self.assertIn("exceeds the maximum limit", str(context.exception))
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    @patch("utils.PdfReader")
    def test_validate_pdf_valid(self, mock_pdf_reader):
        """Verifies that PDFs with <=10 pages are accepted."""
        # Mock PdfReader to return 5 pages
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [MagicMock()] * 5
        mock_pdf_reader.return_value = mock_reader_instance
        
        temp_file = "temp_mock_test.pdf"
        with open(temp_file, "w") as f:
            f.write("%PDF-1.4 mock content")
            
        try:
            is_valid = validate_pdf(temp_file)
            self.assertTrue(is_valid)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_get_embedding_dimension(self):
        """Tests that get_embedding_dimension returns correct dimensions for each provider."""
        with patch("config.EMBEDDING_PROVIDER", "huggingface-api"):
            self.assertEqual(ingest.get_embedding_dimension(), 384)
            
        with patch("config.EMBEDDING_PROVIDER", "huggingface-local"):
            self.assertEqual(ingest.get_embedding_dimension(), 384)
            
        with patch("config.EMBEDDING_PROVIDER", "google-api"):
            self.assertEqual(ingest.get_embedding_dimension(), 768)

if __name__ == "__main__":
    unittest.main()
