from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.pdf_ingestion import PDFKnowledgeIngestor


logging.basicConfig(level=logging.INFO)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest all local PDFs into MongoDB pdf_chunks.")
    parser.add_argument("--pdf-dir", default="data/pdf", help="Directory to scan recursively for PDFs.")
    parser.add_argument("--db-name", default="Stock_data", help="MongoDB database name.")
    parser.add_argument("--collection-name", default="pdf_chunks", help="MongoDB collection name.")
    parser.add_argument("--chunk-size", type=int, default=1800, help="Maximum chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=250, help="Overlap in characters between chunks.")
    parser.add_argument("--min-chunk-chars", type=int, default=120, help="Minimum chunk size to keep.")
    parser.add_argument(
        "--embed-model",
        default="all-MiniLM-L6-v2",
        help="Optional sentence-transformers model name. Falls back to text-only ingestion if unavailable.",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete previously ingested PDF chunks before ingesting the current directory.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    ingestor = PDFKnowledgeIngestor(
        db_name=args.db_name,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_chars=args.min_chunk_chars,
        embed_model_name=args.embed_model,
    )

    report = ingestor.ingest_directory(Path(args.pdf_dir), clear_existing=args.clear_existing)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.files_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
