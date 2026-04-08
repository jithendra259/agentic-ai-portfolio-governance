from __future__ import annotations

import re

from langchain_core.tools import tool

from src.rag.vector_graph_rag import GraphContextRAG, MethodologyVectorRAG


_methodology_rag = None
_graph_rag = None


def _get_methodology_rag() -> MethodologyVectorRAG:
    global _methodology_rag
    if _methodology_rag is None:
        _methodology_rag = MethodologyVectorRAG()
    return _methodology_rag


def _get_graph_rag() -> GraphContextRAG:
    global _graph_rag
    if _graph_rag is None:
        _graph_rag = GraphContextRAG()
    return _graph_rag


@tool
def search_methodology_knowledge_base(question: str, top_k: int = 3) -> str:
    """
    Search the PDF methodology knowledge base and return the most relevant grounded chunks.
    Use this for architecture, methodology, HITL, statistical framing, and paper-style explanation questions.
    """

    return _get_methodology_rag().render_markdown(query=question, top_k=top_k)


@tool
def retrieve_graph_rag_context(
    tickers: list[str] | str | None = None,
    universe: str = "",
    top_k_pairs: int = 5,
) -> str:
    """
    Retrieve graph-aware context from institutional ownership data.
    Use this for questions about shared institutions, overlap, contagion structure, and which stocks are most central.
    """
    normalized_tickers: list[str]
    if isinstance(tickers, str):
        normalized_tickers = [token.strip().upper() for token in re.split(r"[,\s]+", tickers) if token.strip()]
    else:
        normalized_tickers = tickers or []

    return _get_graph_rag().render_markdown(
        tickers=normalized_tickers,
        universe=universe,
        top_k_pairs=top_k_pairs,
    )
