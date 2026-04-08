from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from src.agents.graph_rag_a2 import GraphRAGAgent


load_dotenv()
logger = logging.getLogger(__name__)

DB_NAME = "Stock_data"
PDF_COLLECTION = "pdf_chunks"
TICKER_COLLECTION = "ticker"


class MethodologyVectorRAG:
    """
    Semantic retriever over ``pdf_chunks`` with vector search when embeddings are
    available and text/keyword fallback otherwise.
    """

    def __init__(
        self,
        mongo_uri: str | None = None,
        db_name: str = DB_NAME,
        collection_name: str = PDF_COLLECTION,
        embed_model_name: str | None = "all-MiniLM-L6-v2",
    ) -> None:
        self.mongo_uri = (mongo_uri if mongo_uri is not None else os.getenv("MONGO_URI") or "").strip()
        self.db_name = db_name
        self.collection_name = collection_name
        self.embed_model_name = (embed_model_name or "").strip() or None
        self._client = None
        self._collection = None
        self._embedding_model = None
        self._embedding_attempted = False

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        cleaned_query = str(query or "").strip()
        if not cleaned_query:
            return []

        docs = self._load_chunk_docs()
        if not docs:
            return []

        vector_results = self._vector_search(cleaned_query, docs, top_k=top_k)
        if vector_results:
            return vector_results

        text_results = self._text_search(cleaned_query, top_k=top_k)
        if text_results:
            return text_results

        return self._keyword_fallback_search(cleaned_query, docs, top_k=top_k)

    def render_markdown(self, query: str, top_k: int = 3) -> str:
        results = self.search(query=query, top_k=top_k)
        if not results:
            return (
                "Methodology Knowledge Search\n"
                f"No grounded PDF chunks were found for: {query}"
            )

        lines = [
            "Methodology Knowledge Search",
            f"Query: {query}",
            "",
            "Top grounded chunks:",
        ]

        for item in results:
            score = item.get("score")
            score_text = f"{float(score):.4f}" if isinstance(score, (float, int, np.floating)) else "n/a"
            lines.append(
                f"- [{score_text}] {item.get('source_paper', 'unknown')} | page {item.get('page_number', '?')} | chunk {item.get('chunk_id', 'n/a')}"
            )
            lines.append(f"  {item.get('raw_text', '').strip()}")

        return "\n".join(lines)

    def _load_chunk_docs(self) -> list[dict[str, Any]]:
        collection = self._get_collection()
        if collection is None:
            return []

        return list(
            collection.find(
                {"source_type": "pdf"},
                {
                    "_id": 0,
                    "chunk_id": 1,
                    "source_paper": 1,
                    "page_number": 1,
                    "raw_text": 1,
                    "embedding": 1,
                    "embedding_model": 1,
                },
            )
        )

    def _vector_search(self, query: str, docs: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        embedding_model = self._get_embedding_model()
        if embedding_model is None:
            return []

        vector_docs = [doc for doc in docs if isinstance(doc.get("embedding"), list) and doc.get("embedding")]
        if not vector_docs:
            return []

        try:
            query_embedding = embedding_model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        except Exception as exc:
            logger.warning("Vector query encoding failed: %s", exc)
            return []

        scored_docs = []
        for doc in vector_docs:
            try:
                score = float(np.dot(np.asarray(doc["embedding"], dtype=float), query_embedding))
            except Exception:
                continue

            enriched = dict(doc)
            enriched["score"] = score
            scored_docs.append(enriched)

        scored_docs.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return scored_docs[: max(1, int(top_k))]

    def _text_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        collection = self._get_collection()
        if collection is None:
            return []

        try:
            results = list(
                collection.find(
                    {"source_type": "pdf", "$text": {"$search": query}},
                    {
                        "_id": 0,
                        "chunk_id": 1,
                        "source_paper": 1,
                        "page_number": 1,
                        "raw_text": 1,
                        "score": {"$meta": "textScore"},
                    },
                )
                .sort([("score", {"$meta": "textScore"})])
                .limit(max(1, int(top_k)))
            )
        except Exception:
            return []

        normalized = []
        for item in results:
            enriched = dict(item)
            if "score" in enriched:
                try:
                    enriched["score"] = float(enriched["score"])
                except Exception:
                    pass
            normalized.append(enriched)
        return normalized

    def _keyword_fallback_search(self, query: str, docs: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scored_docs = []
        for doc in docs:
            doc_terms = self._tokenize(doc.get("raw_text", ""))
            if not doc_terms:
                continue

            overlap = query_terms.intersection(doc_terms)
            if not overlap:
                continue

            score = float(len(overlap) / max(1, len(query_terms)))
            enriched = dict(doc)
            enriched["score"] = score
            scored_docs.append(enriched)

        scored_docs.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return scored_docs[: max(1, int(top_k))]

    def _get_embedding_model(self):
        if self._embedding_attempted:
            return self._embedding_model

        self._embedding_attempted = True
        if not self.embed_model_name:
            return None

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.info("sentence-transformers is unavailable; methodology retrieval will use text fallback.")
            return None

        try:
            self._embedding_model = SentenceTransformer(self.embed_model_name)
        except Exception as exc:
            logger.warning("Failed to initialize sentence-transformers model %s: %s", self.embed_model_name, exc)
            self._embedding_model = None

        return self._embedding_model

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9_]+", str(text).lower()))

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        if not self.mongo_uri:
            return None

        self._client = MongoClient(
            self.mongo_uri,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=20000,
            appname="agentic-ai-portfolio-governance-methodology-rag",
        )
        self._collection = self._client[self.db_name][self.collection_name]
        return self._collection


class GraphContextRAG:
    """
    Graph-aware retriever for institutional overlap and structural-risk context.
    """

    def __init__(
        self,
        mongo_uri: str | None = None,
        db_name: str = DB_NAME,
        collection_name: str = TICKER_COLLECTION,
    ) -> None:
        self.mongo_uri = (mongo_uri if mongo_uri is not None else os.getenv("MONGO_URI") or "").strip()
        self.db_name = db_name
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def retrieve(
        self,
        tickers: Optional[list[str] | str] = None,
        universe: str | None = None,
        top_k_pairs: int = 5,
    ) -> dict[str, Any]:
        collection = self._get_collection()
        if collection is None:
            return {"status": "error", "message": "MongoDB is not configured for graph retrieval."}

        if isinstance(tickers, str):
            tickers = [token.strip().upper() for token in re.split(r"[,\s]+", tickers) if token.strip()]

        normalized_tickers = sorted(
            {str(ticker).strip().upper() for ticker in (tickers or []) if str(ticker).strip()}
        )
        normalized_universe = (universe or "").strip().upper()

        query: dict[str, Any]
        if normalized_tickers:
            query = {"ticker": {"$in": normalized_tickers}}
        elif normalized_universe:
            query = {"universes": normalized_universe}
        else:
            return {"status": "error", "message": "Provide tickers or a universe for graph retrieval."}

        docs = list(
            collection.find(
                query,
                {
                    "_id": 0,
                    "ticker": 1,
                    "universes": 1,
                    "graph_relationships.institutional_holders": 1,
                },
            )
        )
        if not docs:
            return {"status": "error", "message": "No graph documents matched the requested tickers or universe."}

        graph_agent = GraphRAGAgent(collection)
        bipartite_graph = nx.Graph()
        stock_nodes = []
        holdings_by_ticker: dict[str, dict[str, float]] = {}

        for doc in docs:
            ticker = str(doc.get("ticker", "")).upper()
            if not ticker:
                continue

            stock_nodes.append(ticker)
            bipartite_graph.add_node(ticker, bipartite=0, node_type="stock")
            parsed_holders = graph_agent._parse_institutional_holders(
                doc.get("graph_relationships", {}).get("institutional_holders", [])
            )
            holdings_by_ticker[ticker] = parsed_holders

            for institution_name, weight in parsed_holders.items():
                bipartite_graph.add_node(institution_name, bipartite=1, node_type="institution")
                bipartite_graph.add_edge(ticker, institution_name, weight=weight)

        projected_graph = graph_agent._build_stock_projection(bipartite_graph, stock_nodes)

        try:
            centrality = nx.eigenvector_centrality(projected_graph, max_iter=2000, weight="weight")
            method_used = "eigenvector"
        except Exception:
            centrality = nx.degree_centrality(projected_graph)
            method_used = "degree"

        top_pairs = []
        for left_stock, right_stock, data in sorted(
            projected_graph.edges(data=True),
            key=lambda edge: float(edge[2].get("weight", 0.0)),
            reverse=True,
        )[: max(1, int(top_k_pairs))]:
            shared_holders = sorted(
                set(holdings_by_ticker.get(left_stock, {}).keys()) & set(holdings_by_ticker.get(right_stock, {}).keys())
            )
            top_pairs.append(
                {
                    "source": left_stock,
                    "target": right_stock,
                    "weight": round(float(data.get("weight", 0.0)), 6),
                    "shared_institutions": shared_holders[:10],
                }
            )

        top_risky_stocks = [
            {"ticker": ticker, "score": round(float(score), 6)}
            for ticker, score in sorted(centrality.items(), key=lambda item: item[1], reverse=True)[:5]
        ]

        institution_summaries = []
        institution_nodes = [node for node, attrs in bipartite_graph.nodes(data=True) if attrs.get("bipartite") == 1]
        for institution_name in institution_nodes:
            degree = bipartite_graph.degree(institution_name)
            if degree > 0:
                linked_tickers = sorted(
                    stock for stock in bipartite_graph.neighbors(institution_name) if stock in stock_nodes
                )
                total_pct_held = sum(
                    float(bipartite_graph[stock][institution_name].get("weight", 0.0))
                    for stock in linked_tickers
                )
                average_pct_held = total_pct_held / max(1, len(linked_tickers))
                institution_summaries.append(
                    {
                        "institution": institution_name,
                        "linked_stocks": int(degree),
                        "total_pct_held": round(total_pct_held * 100.0, 4),
                        "average_pct_held": round(average_pct_held * 100.0, 4),
                        "tickers": linked_tickers,
                    }
                )
        top_institutions = sorted(
            institution_summaries,
            key=lambda item: (item["linked_stocks"], item["total_pct_held"]),
            reverse=True,
        )
        top_investors_by_total_pct_held = sorted(
            institution_summaries,
            key=lambda item: (item["total_pct_held"], item["linked_stocks"]),
            reverse=True,
        )

        single_ticker_holders = []
        if len(stock_nodes) == 1:
            only_ticker = stock_nodes[0]
            single_ticker_holders = [
                {
                    "institution": institution_name,
                    "pct_held": round(weight * 100.0, 4),
                }
                for institution_name, weight in sorted(
                    holdings_by_ticker.get(only_ticker, {}).items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]

        return {
            "status": "success",
            "tickers": stock_nodes,
            "universe": normalized_universe or None,
            "method_used": method_used,
            "graph_node_count": projected_graph.number_of_nodes(),
            "graph_edge_count": projected_graph.number_of_edges(),
            "top_overlap_pairs": top_pairs,
            "top_risky_stocks": top_risky_stocks,
            "top_institutions": top_institutions[:10],
            "top_investors_by_total_pct_held": top_investors_by_total_pct_held[:10],
            "single_ticker_holders": single_ticker_holders[:15],
        }

    def render_markdown(
        self,
        tickers: Optional[list[str] | str] = None,
        universe: str | None = None,
        top_k_pairs: int = 5,
    ) -> str:
        result = self.retrieve(tickers=tickers, universe=universe, top_k_pairs=top_k_pairs)
        if result.get("status") != "success":
            return f"Graph RAG Context\n{result.get('message', 'Unable to retrieve graph context.')}"

        lines = [
            "Graph RAG Context",
            f"Tickers: {', '.join(result.get('tickers', [])) or 'None'}",
            f"Method: {result.get('method_used', 'unknown')}",
            f"Projected graph: {result.get('graph_node_count', 0)} nodes, {result.get('graph_edge_count', 0)} edges",
            "",
            "Top structurally risky stocks:",
        ]

        for item in result.get("top_risky_stocks", []):
            lines.append(f"- {item['ticker']}: {item['score']:.4f}")

        lines.extend(["", "Top overlap pairs:"])
        for item in result.get("top_overlap_pairs", []):
            shared = ", ".join(item.get("shared_institutions", [])) or "No shared institutions listed"
            lines.append(
                f"- {item['source']} <-> {item['target']} | weight={item['weight']:.4f} | shared institutions: {shared}"
            )

        if result.get("single_ticker_holders"):
            lines.extend(["", "Institutions holding this ticker:"])
            for item in result["single_ticker_holders"]:
                lines.append(f"- {item['institution']}: {item['pct_held']:.2f}% held")

        if result.get("top_investors_by_total_pct_held") and len(result.get("tickers", [])) > 1:
            lines.extend(["", "Largest aggregate holders across selected tickers:"])
            for item in result["top_investors_by_total_pct_held"]:
                lines.append(
                    f"- {item['institution']}: aggregate held={item['total_pct_held']:.2f}%"
                    f" across {item['linked_stocks']} stocks"
                    f" | average holding={item['average_pct_held']:.2f}%"
                )

        if result.get("top_institutions"):
            lines.extend(["", "Most connected institutions by coverage:"])
            for item in result["top_institutions"]:
                ticker_list = ", ".join(item.get("tickers", []))
                lines.append(
                    f"- {item['institution']}: linked to {item['linked_stocks']} stocks"
                    f" | total held={item['total_pct_held']:.2f}%"
                    f" | average holding={item['average_pct_held']:.2f}%"
                    f" | tickers: {ticker_list}"
                )

        return "\n".join(lines)

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        if not self.mongo_uri:
            return None

        self._client = MongoClient(
            self.mongo_uri,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=20000,
            appname="agentic-ai-portfolio-governance-graph-rag",
        )
        self._collection = self._client[self.db_name][self.collection_name]
        return self._collection
