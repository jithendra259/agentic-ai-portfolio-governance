import networkx as nx
import pandas as pd
import numpy as np

class GraphCAGAgent:
    """
    Agent 2: Extracts institutional overlap from MongoDB, builds a bipartite 
    network graph, and calculates the Eigenvector Centrality (c_t) to quantify 
    structural contagion risk for each asset.
    """
    def __init__(self, db_collection):
        self.collection = db_collection

    def execute(self, universe_id):
        print(f"🕸️ [Agent 2] Waking up. Extracting Topological Graph for {universe_id}")
        
        # 1. Fetch only the data we need (Tickers + Institutional Holders)
        cursor = self.collection.find(
            {"universes": universe_id}, 
            {"ticker": 1, "graph_relationships.institutional_holders": 1}
        )
        
        B = nx.Graph()
        stocks = []
        
        # 2. Build the Bipartite Graph
        for doc in cursor:
            ticker = doc.get("ticker")
            if not ticker:
                continue
                
            stocks.append(ticker)
            B.add_node(ticker, bipartite=0) # 0 = Stock Node
            
            # Extract institutions
            holders = doc.get("graph_relationships", {}).get("institutional_holders", [])
            for holder in holders:
                inst_name = holder.get("Holder")
                # Use Shares or pctHeld as the edge weight (cleaning string % if needed)
                pct_str = str(holder.get("pctHeld", "0")).replace('%', '')
                try:
                    weight = float(pct_str)
                except ValueError:
                    weight = 0.0
                
                if inst_name and weight > 0:
                    B.add_node(inst_name, bipartite=1) # 1 = Institution Node
                    B.add_edge(ticker, inst_name, weight=weight)
        
        # 3. Calculate Structural Vulnerability (Eigenvector Centrality)
        try:
            # max_iter increased to ensure convergence on dense institutional graphs
            centrality = nx.eigenvector_centrality(B, max_iter=2000, weight='weight')
        except Exception as e:
            print(f"⚠️ Eigenvector convergence failed. Falling back to Degree Centrality. Error: {e}")
            centrality = nx.degree_centrality(B)
        
        # 4. Isolate and Normalize Stock Nodes
        # We don't need the centrality of Vanguard, only the centrality of AAPL
        stock_centrality = {node: score for node, score in centrality.items() if node in stocks}
        
        if not stock_centrality:
            raise ValueError(f"❌ Could not compute centrality for {universe_id}")
            
        c_series = pd.Series(stock_centrality)
        
        # Min-Max Normalization: Forces the highest risk stock to 1.0, lowest to 0.0
        c_min, c_max = c_series.min(), c_series.max()
        if c_max > c_min:
            c_normalized = (c_series - c_min) / (c_max - c_min)
        else:
            c_normalized = pd.Series(0.0, index=c_series.index)
            
        print(f"   -> Graph Built: {B.number_of_nodes()} Total Nodes, {B.number_of_edges()} Edges.")
        print(f"   -> Top 3 Most Central (Risky) Stocks:\n{c_normalized.nlargest(3).to_string()}")
        
        # Return the "Blackboard" state updates
        return {
            "c_vector": c_normalized
        }