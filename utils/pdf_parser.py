# utils/pdf_parser.py
from pypdf import PdfReader
import re

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF and attempts to find a title and abstract."""
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""
        
        # Simple regex to find title and abstract. This is a heuristic.
        title_match = re.search(r"^\s*([A-Z][^.\n]*)\n", full_text)
        abstract_match = re.search(r"Abstract\s*([^\n]*)(.*?)(\n\n|$)", full_text, re.DOTALL | re.IGNORECASE)

        title = title_match.group(1).strip() if title_match else "Could not find title"
        abstract = abstract_match.group(2).strip() if abstract_match else full_text[:1000] # Use first 1000 chars as a fallback

        return {
            "title": title,
            "abstract": abstract,
            "full_text": full_text
        }
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return None