import os
import sys
import unittest


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.vector_graph_rag import GraphContextRAG, MethodologyVectorRAG


class _FakePDFCollection:
    def __init__(self, docs):
        self.docs = docs

    def find(self, query, projection):
        source_type = query.get("source_type")
        for doc in self.docs:
            if source_type and doc.get("source_type") != source_type:
                continue
            yield {
                "chunk_id": doc.get("chunk_id"),
                "source_paper": doc.get("source_paper"),
                "page_number": doc.get("page_number"),
                "raw_text": doc.get("raw_text"),
                "embedding": doc.get("embedding", []),
                "embedding_model": doc.get("embedding_model"),
            }


class _StubMethodologyRAG(MethodologyVectorRAG):
    def __init__(self, docs):
        super().__init__(mongo_uri="", embed_model_name=None)
        self._docs = docs
        self._collection = _FakePDFCollection(docs)

    def _load_chunk_docs(self):
        return list(self._collection.find({"source_type": "pdf"}, {}))


class _FakeTickerCollection:
    def __init__(self, docs):
        self.docs = docs

    def find(self, query, projection):
        if "ticker" in query:
            wanted = set(query["ticker"]["$in"])
            for doc in self.docs:
                if doc.get("ticker") in wanted:
                    yield doc
            return

        if "universes" in query:
            universe = query["universes"]
            for doc in self.docs:
                if universe in doc.get("universes", []):
                    yield doc


class MethodologyVectorRAGTests(unittest.TestCase):
    def test_keyword_fallback_returns_relevant_chunk(self):
        rag = _StubMethodologyRAG(
            [
                {
                    "chunk_id": "it_001",
                    "source_type": "pdf",
                    "source_paper": "methodology",
                    "page_number": 1,
                    "raw_text": "I_t is computed from volatility, correlation, and drawdown.",
                },
                {
                    "chunk_id": "hitl_001",
                    "source_type": "pdf",
                    "source_paper": "methodology",
                    "page_number": 2,
                    "raw_text": "HITL triggers when crisis thresholds or turnover thresholds fire.",
                },
            ]
        )

        results = rag.search("how is I_t computed?", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "it_001")


class GraphContextRAGTests(unittest.TestCase):
    def test_graph_context_surfaces_overlap_and_institutions(self):
        docs = [
            {
                "ticker": "AAA",
                "universes": ["U1"],
                "graph_relationships": {
                    "institutional_holders": [
                        {"Holder": "Vanguard", "pctHeld": "10%"},
                        {"Holder": "BlackRock", "pctHeld": "5%"},
                    ]
                },
            },
            {
                "ticker": "BBB",
                "universes": ["U1"],
                "graph_relationships": {
                    "institutional_holders": [
                        {"Holder": "Vanguard", "pctHeld": "8%"},
                        {"Holder": "State Street", "pctHeld": "4%"},
                    ]
                },
            },
            {
                "ticker": "CCC",
                "universes": ["U1"],
                "graph_relationships": {
                    "institutional_holders": [
                        {"Holder": "Independent", "pctHeld": "7%"},
                    ]
                },
            },
        ]

        rag = GraphContextRAG(mongo_uri="")
        rag._collection = _FakeTickerCollection(docs)
        result = rag.retrieve(tickers=["AAA", "BBB", "CCC"])

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["graph_edge_count"], 1)
        self.assertEqual(result["top_overlap_pairs"][0]["source"], "AAA")
        self.assertEqual(result["top_overlap_pairs"][0]["target"], "BBB")
        self.assertIn("Vanguard", result["top_overlap_pairs"][0]["shared_institutions"])
        self.assertEqual(result["top_investors_by_total_pct_held"][0]["institution"], "Vanguard")
        self.assertAlmostEqual(result["top_investors_by_total_pct_held"][0]["total_pct_held"], 18.0, places=3)

    def test_graph_context_accepts_string_tickers_and_surfaces_single_ticker_holders(self):
        docs = [
            {
                "ticker": "NVDA",
                "universes": ["U1"],
                "graph_relationships": {
                    "institutional_holders": [
                        {"Holder": "Vanguard", "pctHeld": "10%"},
                        {"Holder": "BlackRock", "pctHeld": "5%"},
                    ]
                },
            }
        ]

        rag = GraphContextRAG(mongo_uri="")
        rag._collection = _FakeTickerCollection(docs)

        result = rag.retrieve(tickers="NVDA")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tickers"], ["NVDA"])
        self.assertEqual(result["graph_node_count"], 1)
        self.assertEqual(result["graph_edge_count"], 0)
        self.assertEqual(result["single_ticker_holders"][0]["institution"], "Vanguard")
        self.assertAlmostEqual(result["single_ticker_holders"][0]["pct_held"], 10.0, places=3)

    def test_render_markdown_includes_aggregate_holder_amounts(self):
        docs = [
            {
                "ticker": "AAA",
                "universes": ["U1"],
                "graph_relationships": {
                    "institutional_holders": [
                        {"Holder": "Vanguard", "pctHeld": "10%"},
                        {"Holder": "BlackRock", "pctHeld": "5%"},
                    ]
                },
            },
            {
                "ticker": "BBB",
                "universes": ["U1"],
                "graph_relationships": {
                    "institutional_holders": [
                        {"Holder": "Vanguard", "pctHeld": "8%"},
                        {"Holder": "State Street", "pctHeld": "4%"},
                    ]
                },
            },
        ]

        rag = GraphContextRAG(mongo_uri="")
        rag._collection = _FakeTickerCollection(docs)

        markdown = rag.render_markdown(tickers="AAA BBB")

        self.assertIn("Largest aggregate holders across selected tickers:", markdown)
        self.assertIn("aggregate held=18.00%", markdown)
        self.assertIn("Most connected institutions by coverage:", markdown)


if __name__ == "__main__":
    unittest.main()
