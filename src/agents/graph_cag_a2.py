import logging

import networkx as nx
import pandas as pd


logger = logging.getLogger(__name__)


class GraphCAGAgent:
    """
    Agent 2: Build the institutional bipartite graph and compute structural contagion risk.
    """

    def __init__(self, db_collection):
        self.collection = db_collection

    def execute(self, universe_id):
        print(f"[Agent 2] Extracting topological graph for {universe_id}")

        cursor = self.collection.find(
            {"universes": universe_id},
            {"ticker": 1, "graph_relationships.institutional_holders": 1},
        )

        graph = nx.Graph()
        stocks = []

        for doc in cursor:
            ticker = doc.get("ticker")
            if not ticker:
                continue

            stocks.append(ticker)
            graph.add_node(ticker, bipartite=0)

            holders = doc.get("graph_relationships", {}).get("institutional_holders", [])
            for holder in holders:
                inst_name = holder.get("Holder")
                pct_str = str(holder.get("pctHeld", "0")).replace("%", "")
                try:
                    weight = float(pct_str)
                except ValueError:
                    weight = 0.0

                if inst_name and weight > 0:
                    graph.add_node(inst_name, bipartite=1)
                    graph.add_edge(ticker, inst_name, weight=weight)

        fallback_applied = False
        try:
            centrality = nx.eigenvector_centrality(graph, max_iter=2000, weight="weight")
            method_used = "eigenvector"
        except Exception as exc:
            print(f"WARNING: Eigenvector convergence failed. Falling back to degree centrality. Error: {exc}")
            logger.warning(
                "Agent 2 fallback triggered for %s. Using degree centrality instead of eigenvector. Error: %s",
                universe_id,
                exc,
            )
            centrality = nx.degree_centrality(graph)
            method_used = "degree"
            fallback_applied = True

        stock_centrality = {node: score for node, score in centrality.items() if node in stocks}
        if not stock_centrality:
            raise ValueError(f"Could not compute centrality for {universe_id}")

        c_series = pd.Series(stock_centrality)
        c_min, c_max = c_series.min(), c_series.max()
        if c_max > c_min:
            c_normalized = (c_series - c_min) / (c_max - c_min)
        else:
            c_normalized = pd.Series(0.0, index=c_series.index)

        print(f"   -> Graph built: {graph.number_of_nodes()} total nodes, {graph.number_of_edges()} edges.")
        print(f"   -> Centrality method: {method_used}")
        print(f"   -> Top 3 most central (risky) stocks:\n{c_normalized.nlargest(3).to_string()}")

        return {
            "c_vector": c_normalized,
            "method_used": method_used,
            "fallback_applied": fallback_applied,
            "graph_node_count": graph.number_of_nodes(),
            "graph_edge_count": graph.number_of_edges(),
        }
