import pypdf # if pypdf is installed, else use pdfplumber or fitz
import sys

try:
    import fitz # PyMuPDF
    doc = fitz.open("main.pdf")
    print(f"Number of pages: {len(doc)}")
    
    # Search for captions
    for i, page in enumerate(doc):
        text = page.get_text()
        if "Figure " in text:
            for line in text.splitlines():
                if "Figure " in line:
                    print(f"Page {i+1}: {line}")
except Exception as e:
    print(f"Error: {e}")
