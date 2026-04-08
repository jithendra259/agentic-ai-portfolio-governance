import os
import sys
import unittest


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.graph_rag_a2 import GraphRAGAgent


class FakeCollection:
    def __init__(self, docs):
        self.docs = docs

    def find(self, query, projection):
        universe_id = query.get("universes")
        for doc in self.docs:
            universes = doc.get("universes", [])
            if universe_id in universes:
                yield {
                    "ticker": doc.get("ticker"),
                    "graph_relationships": doc.get("graph_relationships", {}),
                }


class GraphRAGAgentTests(unittest.TestCase):
    def test_projected_stock_graph_prefers_shared_holder_overlap(self):
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

        agent = GraphRAGAgent(FakeCollection(docs))
        result = agent.execute("U1")
        c_vector = result["c_vector"]
        graph_context = result["graph_context"]

        self.assertEqual(result["graph_node_count"], 3)
        self.assertEqual(result["graph_edge_count"], 1)
        self.assertEqual(result["bipartite_edge_count"], 5)
        self.assertIn("AAA", c_vector.index.tolist())
        self.assertIn("BBB", c_vector.index.tolist())
        self.assertIn("CCC", c_vector.index.tolist())
        self.assertGreaterEqual(float(c_vector["AAA"]), float(c_vector["CCC"]))
        self.assertGreaterEqual(float(c_vector["BBB"]), float(c_vector["CCC"]))
        self.assertEqual(graph_context["universe_id"], "U1")
        self.assertEqual(graph_context["institution_count"], 4)
        self.assertTrue(graph_context["top_risky_stocks"])
        self.assertEqual(graph_context["top_overlap_pairs"][0]["source"], "AAA")
        self.assertEqual(graph_context["top_overlap_pairs"][0]["target"], "BBB")


if __name__ == "__main__":
    unittest.main()
