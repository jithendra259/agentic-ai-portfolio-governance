import os
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.pdf_ingestion import PDFKnowledgeIngestor


class PDFKnowledgeIngestorTests(unittest.TestCase):
    def test_list_pdf_files_only_returns_pdfs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "paper_a.pdf").write_bytes(b"%PDF-1.4")
            (root / "notes.txt").write_text("ignore")
            nested = root / "nested"
            nested.mkdir()
            (nested / "paper_b.pdf").write_bytes(b"%PDF-1.4")
            (nested / "draft.docx").write_text("ignore")

            pdfs = PDFKnowledgeIngestor.list_pdf_files(root)

            self.assertEqual(sorted(path.name for path in pdfs), ["paper_a.pdf", "paper_b.pdf"])

    def test_chunk_text_creates_multiple_overlapping_chunks(self):
        ingestor = PDFKnowledgeIngestor(
            mongo_uri="",
            embed_model_name=None,
            chunk_size=180,
            chunk_overlap=40,
            min_chunk_chars=40,
        )
        text = " ".join(f"token{i}" for i in range(120))

        chunks = ingestor.chunk_text(text)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 180 for chunk in chunks))
        self.assertTrue(set(chunks[0].split()) & set(chunks[1].split()))


if __name__ == "__main__":
    unittest.main()
