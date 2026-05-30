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
    parser = argparse.ArgumentParser(description="Backfill embeddings for existing pdf_chunks.")
    parser.add_argument("--db-name", default="Stock_data", help="MongoDB database name.")
    parser.add_argument("--collection-name", default="pdf_chunks", help="MongoDB collection name.")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2", help="sentence-transformers model name.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ingestor = PDFKnowledgeIngestor(
        db_name=args.db_name,
        collection_name=args.collection_name,
        embed_model_name=args.embed_model,
    )
    updated = ingestor.backfill_embeddings(batch_size=args.batch_size)
    print(json.dumps({"updated_chunks": updated, "embedding_model": args.embed_model}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
