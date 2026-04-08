from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient, UpdateOne


load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class PDFIngestionReport:
    files_seen: int = 0
    files_ingested: int = 0
    files_skipped_duplicate: int = 0
    files_failed: int = 0
    chunks_written: int = 0
    embeddings_generated: int = 0
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PDFKnowledgeIngestor:
    """
    Reusable PDF-to-Mongo ingestion pipeline for the thesis knowledge base.

    - scans a directory recursively for ``.pdf`` files
    - extracts text page by page
    - chunks text into retrieval-sized passages
    - stores chunks in MongoDB ``pdf_chunks``
    - adds optional sentence-transformer embeddings when available
    """

    def __init__(
        self,
        mongo_uri: str | None = None,
        db_name: str = "Stock_data",
        collection_name: str = "pdf_chunks",
        chunk_size: int = 1800,
        chunk_overlap: int = 250,
        min_chunk_chars: int = 120,
        embed_model_name: str | None = "all-MiniLM-L6-v2",
    ) -> None:
        resolved_mongo_uri = os.getenv("MONGO_URI") if mongo_uri is None else mongo_uri
        self.mongo_uri = (resolved_mongo_uri or "").strip()
        self.db_name = db_name
        self.collection_name = collection_name
        self.chunk_size = max(100, int(chunk_size))
        self.chunk_overlap = max(0, min(int(chunk_overlap), self.chunk_size // 2))
        self.min_chunk_chars = max(50, int(min_chunk_chars))
        self.embed_model_name = (embed_model_name or "").strip() or None

        self._client: MongoClient | None = None
        self._collection = None
        self._embedding_model = None
        self._embedding_attempted = False

        if self.mongo_uri:
            self._client = MongoClient(
                self.mongo_uri,
                tls=True,
                tlsAllowInvalidCertificates=True,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
                socketTimeoutMS=20000,
                appname="agentic-ai-portfolio-governance-pdf-ingestion",
            )
            self._collection = self._client[self.db_name][self.collection_name]

    @staticmethod
    def list_pdf_files(pdf_root: str | Path) -> list[Path]:
        root = Path(pdf_root)
        if not root.exists():
            raise FileNotFoundError(f"PDF directory does not exist: {root}")

        return sorted(
            (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
            key=lambda path: path.as_posix().lower(),
        )

    def chunk_text(self, text: str) -> list[str]:
        normalized = self.normalize_text(text)
        if not normalized:
            return []

        paragraphs = [part.strip() for part in re.split(r"\n{2,}", normalized) if part.strip()]
        if not paragraphs:
            paragraphs = [normalized]

        chunks: list[str] = []
        buffer = ""

        for paragraph in paragraphs:
            candidate = paragraph if not buffer else f"{buffer}\n\n{paragraph}"
            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue

            if buffer:
                chunks.append(buffer.strip())
                overlap_seed = self._tail_overlap(buffer)
                buffer = f"{overlap_seed}\n\n{paragraph}".strip() if overlap_seed else paragraph
            else:
                buffer = paragraph

            while len(buffer) > self.chunk_size:
                slice_text = buffer[: self.chunk_size]
                split_at = slice_text.rfind(" ")
                if split_at >= int(self.chunk_size * 0.6):
                    slice_text = slice_text[:split_at]

                cleaned_slice = slice_text.strip()
                if cleaned_slice:
                    chunks.append(cleaned_slice)

                overlap_seed = self._tail_overlap(cleaned_slice)
                remainder = buffer[len(slice_text) :].strip()
                buffer = f"{overlap_seed} {remainder}".strip() if overlap_seed else remainder

        if buffer.strip():
            chunks.append(buffer.strip())

        return [chunk for chunk in chunks if len(chunk) >= self.min_chunk_chars]

    @staticmethod
    def normalize_text(text: str) -> str:
        cleaned = text.replace("\x00", " ")
        cleaned = cleaned.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        cleaned = re.sub(r"-\s*\n\s*", "", cleaned)
        cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def ensure_indexes(self) -> None:
        collection = self._require_collection()
        collection.create_index([("chunk_id", ASCENDING)], unique=True, background=True)
        collection.create_index([("source_type", ASCENDING), ("source_path", ASCENDING)], background=True)
        collection.create_index([("source_paper", ASCENDING), ("page_number", ASCENDING)], background=True)
        collection.create_index([("file_hash", ASCENDING)], background=True)
        try:
            collection.create_index([("raw_text", "text")], name="pdf_text_search", background=True)
        except Exception as exc:
            logger.warning("Could not create text index for pdf_chunks: %s", exc)

    def clear_pdf_chunks(self) -> int:
        collection = self._require_collection()
        result = collection.delete_many({"source_type": "pdf"})
        return int(result.deleted_count)

    def ingest_directory(self, pdf_root: str | Path, clear_existing: bool = False) -> PDFIngestionReport:
        collection = self._require_collection()
        self.ensure_indexes()

        report = PDFIngestionReport()
        pdf_files = self.list_pdf_files(pdf_root)
        report.files_seen = len(pdf_files)

        if clear_existing:
            deleted = collection.delete_many({"source_type": "pdf"})
            logger.info("Cleared %s existing PDF chunks before ingestion.", deleted.deleted_count)

        seen_hashes: set[str] = set()

        for pdf_path in pdf_files:
            try:
                file_hash = self._hash_file(pdf_path)
                if file_hash in seen_hashes:
                    report.files_skipped_duplicate += 1
                    continue

                seen_hashes.add(file_hash)
                chunks_written, embeddings_generated = self.ingest_pdf(pdf_path, file_hash=file_hash)
                report.files_ingested += 1
                report.chunks_written += chunks_written
                report.embeddings_generated += embeddings_generated
            except Exception as exc:
                logger.exception("Failed to ingest PDF: %s", pdf_path)
                report.files_failed += 1
                report.failures.append(f"{pdf_path.name}: {exc}")

        return report

    def backfill_embeddings(self, source_type: str = "pdf", batch_size: int = 64) -> int:
        collection = self._require_collection()
        embedding_model = self._get_embedding_model()
        if embedding_model is None:
            raise RuntimeError("sentence-transformers is unavailable; cannot backfill embeddings.")

        docs = list(
            collection.find(
                {
                    "source_type": source_type,
                    "$or": [{"embedding": {"$exists": False}}, {"embedding": []}],
                },
                {"_id": 0, "chunk_id": 1, "raw_text": 1},
            )
        )
        if not docs:
            return 0

        updated = 0
        for start in range(0, len(docs), max(1, int(batch_size))):
            batch = docs[start : start + max(1, int(batch_size))]
            texts = [doc.get("raw_text", "") for doc in batch]
            embeddings = embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

            operations = []
            for doc, embedding in zip(batch, embeddings):
                operations.append(
                    UpdateOne(
                        {"chunk_id": doc["chunk_id"]},
                        {
                            "$set": {
                                "embedding": embedding.tolist(),
                                "embedding_model": self.embed_model_name,
                                "updated_at": datetime.now(timezone.utc),
                            }
                        },
                    )
                )

            if operations:
                collection.bulk_write(operations, ordered=False)
                updated += len(operations)

        return updated

    def ingest_pdf(self, pdf_path: str | Path, file_hash: str | None = None) -> tuple[int, int]:
        collection = self._require_collection()
        path = Path(pdf_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Only .pdf files are supported, got: {path.name}")

        pages = self._extract_pdf_pages(path)
        if not pages:
            raise ValueError("No readable text was extracted from the PDF.")

        file_hash = file_hash or self._hash_file(path)
        now = datetime.now(timezone.utc)
        relative_source = path.as_posix()

        chunk_records: list[dict[str, Any]] = []
        chunk_texts: list[str] = []

        for page_number, page_text in pages:
            chunks = self.chunk_text(page_text)
            for chunk_index, chunk_text in enumerate(chunks, start=1):
                chunk_id = hashlib.sha256(
                    f"{file_hash}:{page_number}:{chunk_index}:{chunk_text}".encode("utf-8")
                ).hexdigest()
                chunk_records.append(
                    {
                        "chunk_id": chunk_id,
                        "source_type": "pdf",
                        "source_path": relative_source,
                        "source_paper": path.stem,
                        "file_name": path.name,
                        "file_hash": file_hash,
                        "page_number": int(page_number),
                        "chunk_index": int(chunk_index),
                        "raw_text": chunk_text,
                        "text_length": len(chunk_text),
                        "updated_at": now,
                        "$setOnInsert": {"created_at": now},
                    }
                )
                chunk_texts.append(chunk_text)

        if not chunk_records:
            raise ValueError("Text was extracted, but no chunk met the minimum chunk size.")

        embeddings = self._embed_chunks(chunk_texts)
        embedding_model = self.embed_model_name if embeddings else None

        operations = []
        for index, record in enumerate(chunk_records):
            set_payload = {
                "source_type": record["source_type"],
                "source_path": record["source_path"],
                "source_paper": record["source_paper"],
                "file_name": record["file_name"],
                "file_hash": record["file_hash"],
                "page_number": record["page_number"],
                "chunk_index": record["chunk_index"],
                "raw_text": record["raw_text"],
                "text_length": record["text_length"],
                "updated_at": record["updated_at"],
                "embedding": embeddings[index] if embeddings else [],
                "embedding_model": embedding_model,
            }
            operations.append(
                UpdateOne(
                    {"chunk_id": record["chunk_id"]},
                    {"$set": set_payload, "$setOnInsert": record["$setOnInsert"]},
                    upsert=True,
                )
            )

        collection.bulk_write(operations, ordered=False)

        return len(chunk_records), len(embeddings)

    def _require_collection(self):
        if self._collection is None:
            raise RuntimeError("MongoDB is not configured. Set MONGO_URI before running PDF ingestion.")
        return self._collection

    def _extract_pdf_pages(self, pdf_path: Path) -> list[tuple[int, str]]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "pypdf is not installed. Install it in the project venv before running PDF ingestion."
            ) from exc

        reader = PdfReader(str(pdf_path))
        pages: list[tuple[int, str]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            normalized = self.normalize_text(text)
            if normalized:
                pages.append((page_number, normalized))
        return pages

    def _get_embedding_model(self):
        if self._embedding_attempted:
            return self._embedding_model

        self._embedding_attempted = True
        if not self.embed_model_name:
            return None

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.info("sentence-transformers not installed; storing PDF chunks without embeddings.")
            return None

        try:
            self._embedding_model = SentenceTransformer(self.embed_model_name)
        except Exception as exc:
            logger.warning("Failed to initialize embedding model %s: %s", self.embed_model_name, exc)
            self._embedding_model = None

        return self._embedding_model

    def _embed_chunks(self, chunk_texts: list[str]) -> list[list[float]]:
        model = self._get_embedding_model()
        if model is None or not chunk_texts:
            return []

        try:
            embeddings = model.encode(
                chunk_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.warning("Embedding generation failed; continuing with text-only chunks: %s", exc)
            return []

        return [embedding.tolist() for embedding in embeddings]

    def _tail_overlap(self, text: str) -> str:
        if self.chunk_overlap <= 0 or len(text) <= self.chunk_overlap:
            return text.strip()

        seed = text[-self.chunk_overlap :].strip()
        first_space = seed.find(" ")
        if 0 <= first_space < len(seed) // 2:
            seed = seed[first_space + 1 :]
        return seed.strip()

    @staticmethod
    def _hash_file(pdf_path: Path) -> str:
        digest = hashlib.sha256()
        with pdf_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
