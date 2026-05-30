import logging

import networkx as nx
import pandas as pd


logger = logging.getLogger(__name__)


class GraphRAGAgent:
    """
    Agent 2: Build the institutional holdings graph and expose graph context for
    downstream optimization and explanation.

    The notebook's valid Agent 2 logic is still a deterministic graph-centrality
    pass. This class keeps that math intact while surfacing compact graph
    context that can be reused by the explainer and conversational layer.
    """

    def __init__(self, db_collection):
        self.collection = db_collection

    def execute(self, universe_id):
        print(f"[Agent 2] Building graph context for {universe_id}")

        cursor = self.collection.find(
            {"universes": universe_id},
            {"ticker": 1, "graph_relationships.institutional_holders": 1},
        )

        bipartite_graph = nx.Graph()
        stocks = []

        for doc in cursor:
            ticker = doc.get("ticker")
            if not ticker:
                continue

            stocks.append(ticker)
            bipartite_graph.add_node(ticker, bipartite=0, node_type="stock")

            holders = doc.get("graph_relationships", {}).get("institutional_holders", [])
            parsed_holders = self._parse_institutional_holders(holders)
            for institution_name, weight in parsed_holders.items():
                bipartite_graph.add_node(institution_name, bipartite=1, node_type="institution")
                bipartite_graph.add_edge(ticker, institution_name, weight=weight)

        projected_graph = self._build_stock_projection(bipartite_graph, stocks)

        fallback_applied = False
        try:
            centrality = nx.eigenvector_centrality(projected_graph, max_iter=2000, weight="weight")
            method_used = "eigenvector"
        except Exception as exc:
            print(f"WARNING: Eigenvector convergence failed. Falling back to degree centrality. Error: {exc}")
            logger.warning(
                "Agent 2 fallback triggered for %s. Using degree centrality instead of eigenvector. Error: %s",
                universe_id,
                exc,
            )
            centrality = nx.degree_centrality(projected_graph)
            method_used = "degree"
            fallback_applied = True

        stock_centrality = {node: score for node, score in centrality.items() if node in stocks}
        if not stock_centrality:
            raise ValueError(f"Could not compute centrality for {universe_id}")

        c_series = pd.Series(stock_centrality, dtype=float)
        c_min, c_max = c_series.min(), c_series.max()
        if c_max > c_min:
            c_normalized = (c_series - c_min) / (c_max - c_min)
        else:
            c_normalized = pd.Series(0.0, index=c_series.index, dtype=float)

        graph_context = self._build_graph_context(
            universe_id=universe_id,
            bipartite_graph=bipartite_graph,
            projected_graph=projected_graph,
            c_normalized=c_normalized,
        )

        print(
            f"   -> Bipartite graph: {bipartite_graph.number_of_nodes()} total nodes, "
            f"{bipartite_graph.number_of_edges()} edges."
        )
        print(
            f"   -> Projected stock graph: {projected_graph.number_of_nodes()} nodes, "
            f"{projected_graph.number_of_edges()} weighted edges."
        )
        print(f"   -> Centrality method: {method_used}")
        print(f"   -> Top 3 structurally risky stocks:\n{c_normalized.nlargest(3).to_string()}")

        return {
            "c_vector": c_normalized,
            "method_used": method_used,
            "fallback_applied": fallback_applied,
            "graph_node_count": projected_graph.number_of_nodes(),
            "graph_edge_count": projected_graph.number_of_edges(),
            "bipartite_node_count": bipartite_graph.number_of_nodes(),
            "bipartite_edge_count": bipartite_graph.number_of_edges(),
            "graph_context": graph_context,
        }

    def _parse_institutional_holders(self, holders):
        parsed_holders = {}

        if isinstance(holders, dict):
            for institution_name, raw_weight in holders.items():
                weight = self._normalize_weight(raw_weight)
                if institution_name and weight > 0.0:
                    parsed_holders[str(institution_name)] = weight
            return parsed_holders

        if not isinstance(holders, list):
            return parsed_holders

        for index, holder in enumerate(holders):
            if not isinstance(holder, dict):
                continue

            institution_name = (
                holder.get("Holder")
                or holder.get("holder")
                or holder.get("name")
                or f"Institution_{index}"
            )
            raw_weight = (
                holder.get("pctHeld")
                or holder.get("% Out")
                or holder.get("Pct Held")
                or holder.get("weight")
                or 0.0
            )

            weight = self._normalize_weight(raw_weight)
            if institution_name and weight > 0.0:
                parsed_holders[str(institution_name)] = weight

        return parsed_holders

    def _normalize_weight(self, raw_weight):
        pct_str = str(raw_weight).replace("%", "").strip()
        try:
            weight = float(pct_str)
        except ValueError:
            return 0.0

        if weight > 1.0:
            weight = weight / 100.0
        return max(weight, 0.0)

    def _build_stock_projection(self, bipartite_graph, stocks):
        projected_graph = nx.Graph()
        projected_graph.add_nodes_from(stocks)

        for left_index in range(len(stocks)):
            for right_index in range(left_index + 1, len(stocks)):
                left_stock = stocks[left_index]
                right_stock = stocks[right_index]

                shared_institutions = set(bipartite_graph.neighbors(left_stock)) & set(
                    bipartite_graph.neighbors(right_stock)
                )
                if not shared_institutions:
                    continue

                overlap_weight = 0.0
                for institution_name in shared_institutions:
                    left_weight = float(bipartite_graph[left_stock][institution_name].get("weight", 0.0))
                    right_weight = float(bipartite_graph[right_stock][institution_name].get("weight", 0.0))
                    overlap_weight += (left_weight + right_weight) / 2.0

                if overlap_weight > 0.0:
                    projected_graph.add_edge(left_stock, right_stock, weight=overlap_weight)

        return projected_graph

    def _build_graph_context(self, universe_id, bipartite_graph, projected_graph, c_normalized):
        top_risky_stocks = [
            {"ticker": ticker, "score": round(float(score), 6)}
            for ticker, score in c_normalized.sort_values(ascending=False).head(5).items()
        ]

        top_overlap_pairs = []
        ranked_edges = sorted(
            projected_graph.edges(data=True),
            key=lambda edge: float(edge[2].get("weight", 0.0)),
            reverse=True,
        )
        for left_stock, right_stock, data in ranked_edges[:5]:
            top_overlap_pairs.append(
                {
                    "source": left_stock,
                    "target": right_stock,
                    "weight": round(float(data.get("weight", 0.0)), 6),
                }
            )

        institution_count = len(
            [node for node, attrs in bipartite_graph.nodes(data=True) if attrs.get("bipartite") == 1]
        )

        return {
            "universe_id": universe_id,
            "top_risky_stocks": top_risky_stocks,
            "top_overlap_pairs": top_overlap_pairs,
            "institution_count": institution_count,
        }
