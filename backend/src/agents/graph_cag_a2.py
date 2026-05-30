"""
Backward-compatible shim.

The project now uses the notebook-aligned ``GraphRAGAgent`` naming so Agent 2
is no longer described as CAG. Imports from the old module path still work
while the rest of the codebase migrates.
"""

from src.agents.graph_rag_a2 import GraphRAGAgent


GraphCAGAgent = GraphRAGAgent

__all__ = ["GraphRAGAgent", "GraphCAGAgent"]
