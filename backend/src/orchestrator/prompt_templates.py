"""Prompt Engineering & Assembly Layer.

Defines the core institutional system prompt, guidelines loading,
and blackboard state assembly for the orchestrator.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from src.memory.conversation_memory import conversation_prompt_block
from src.orchestrator.formatters import _message_content_to_text

logger = logging.getLogger(__name__)

_MAX_TOOL_MSG_CHARS = 1000
_MAX_CONTEXT_MESSAGES = 5
_MAX_SUMMARY_CHARS = 800

CHATBOT_CONVERSATION_GUIDANCE_DIR = Path(__file__).resolve().parents[1] / "rag" / "knowledge" / "chatbot_conversation"

SYSTEM_PROMPT = """You are an elite Quantitative Portfolio Governance Agent.
You strictly use historical market data (2005-2025) from local MongoDB.

ABSOLUTE RULES:
1. Advisory only. NEVER buy, sell, or execute trades.
2. NEVER hallucinate or invent data. If a tool fails, report the failure directly.
3. For math, use LaTeX delimiters: inline \\(...\\) and display \\[...\\] or $$...$$. Never use single-dollar signs.
4. Prefer action over questions. Act immediately when ticker, date, or strategy is known.

TOOL SELECTION:
- Governance & Allocation: Use run_full_governance_pipeline for optimization, G-CVaR, and allocation.
- Price Charts: Use plot_historical_prices for simple historical closing price charts.
- Custom Charts: Use generate_financial_plot (line, bar, pie, scatter, heatmap, etc.) or run_data_analysis_plot for analytics plots.
- Full Stock EDA: Use calculate_full_stock_eda for statistical moments (skew, kurtosis), Sharpe, Sortino, VaR/CVaR, and Ledoit-Wolf correlation.
- Stationarity & Econometrics: Use run_stationarity_and_econometric_tests for ADF, KPSS, and Ljung-Box autocorrelation tests.
- Volatility Modeling: Use fit_garch_volatility_model for GARCH(1,1) volatility persistence and term forecasts.
- Portfolio Optimization: Use compute_optimal_portfolio_frontier for Markowitz Max Sharpe, Min Volatility, and Risk Parity allocations.
- ARIMA Price Forecasting: Use forecast_price_arima for short-term price paths with 95% confidence intervals.
- Analysis & Statistics: Use get_price_series_for_analysis for returns, volatility, drawdowns, correlations, or correlation heatmap. Never tell the user you cannot do this analysis.
- Ownership & Graph: Use retrieve_graph_rag_context for institutional holdings and ownership overlap.
- Discovery: Use list_available_universes, list_available_sectors, get_stocks_by_sector, get_stocks_by_universe, get_stock_database_snapshot.
- Methodology: Use search_methodology_knowledge_base for questions about the research paper, ARIMA, GARCH, ADF, or G-CVaR.

FOLLOW-UP & MEMORY:
- Maintain context across conversation turns. Reuse previously selected tickers and dates unless changed.
- Never output attachment:// or unregistered plot:// links. Reference charts using registered plot tokens (__PLOTSPEC__:<plot_id>) or tool output links."""


def _latest_human_text(messages: list[BaseMessage] | None) -> str:
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            return _message_content_to_text(message)
    return ""


@lru_cache(maxsize=1)
def _load_chatbot_conversation_guidance() -> str:
    if not CHATBOT_CONVERSATION_GUIDANCE_DIR.exists():
        return ""

    chunks: list[str] = []
    priority_files = ["hallucination_guardrails.md", "portfolio_followups.md"]
    for filename in priority_files:
        path = CHATBOT_CONVERSATION_GUIDANCE_DIR / filename
        if path.exists():
            try:
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    chunks.append(text)
            except OSError as exc:
                logger.warning("Unable to read guidance %s: %s", path, exc)
    if not chunks:
        return ""
    return "### CHATBOT CONVERSATION GUIDANCE ###\n" + "\n\n".join(chunks)


def assemble_system_prompt(state: dict[str, Any]) -> str:
    """Build the dynamic system prompt with blackboard state injected."""
    original_goal = state.get("original_goal") or _latest_human_text(state.get("messages", [])) or "Portfolio governance conversation"
    scratchpad = state.get("scratchpad") or {}
    historical_summary = state.get("historical_summary") or state.get("summary") or ""
    session_state = state.get("session_state") if isinstance(state.get("session_state"), dict) else {}
    conversation_memory = conversation_prompt_block(session_state)
    conversation_guidance = _load_chatbot_conversation_guidance()

    if scratchpad:
        scratchpad_lines = "\n".join(
            f"- {name}: {payload}" for name, payload in sorted(scratchpad.items())
        )
    else:
        scratchpad_lines = "- No exact financial metrics saved yet."

    return (
        f"{SYSTEM_PROMPT}\n\n"
        "### BLACKBOARD STATE ###\n"
        f"Original goal: {original_goal}\n\n"
        "Exact financial scratchpad metrics. These are authoritative. Read these before recalculating:\n"
        f"{scratchpad_lines}\n\n"
        "Historical summary of older context:\n"
        f"{historical_summary or 'No distant context summarized yet.'}\n\n"
        f"{conversation_guidance}\n\n"
        f"{conversation_memory}\n"
        "### SCRATCHPAD RULES ###\n"
        "- Always read from the scratchpad before calculating a metric.\n"
        "- If a needed metric already exists in the scratchpad, use that exact value and do not recalculate.\n"
        "- After any tool/API/math calculation of a critical metric, call save_financial_metric with the exact output.\n"
        "- Preserve precise values for betas, correlations, weights, returns, volatility, Sharpe, CVaR, prices, and allocation changes.\n"
    )
