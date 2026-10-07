"""Specialized Domain Agents for Institutional Portfolio Governance.

Defines discrete domain agents with dedicated system prompts, tool subsets,
and execution nodes designed for LangGraph integration and orchestrator routing:
1. MarketDataAgent: Universe discovery, sector analysis, historical price bundles.
2. EconometricAgent: Time-series diagnostics (ADF, KPSS, Ljung-Box) and ARIMA forecasting.
3. RiskVolatilityAgent: GARCH(1,1)/EGARCH modeling, VaR/CVaR, Ledoit-Wolf shrinkage, and full stock EDA.
4. PortfolioOptimizationAgent: CVXPY Markowitz Max Sharpe, Min Volatility, Risk Parity, G-CVaR conic optimization.
5. NetworkGraphRAGAgent: SEC 13F bipartite ownership networks, centrality, Graph RAG context retrieval.
6. TechnicalAnalysisAgent: Technical indicators (RSI, MACD, Bollinger Bands, ATR) and custom math plots.
7. GovernanceAuditAgent: Fiduciary compliance checks, audit trail inspection, metric ledger logging.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

# =====================================================================
# 1. SPECIALIZED SYSTEM PROMPTS (Imported from dedicated prompts module)
# =====================================================================

from src.orchestrator.prompts import (
    DATA_SENTINEL_PROMPT,
    ECONOMETRIC_PROMPT,
    EXPLAINABILITY_PROMPT,
    GOVERNANCE_AUDIT_PROMPT,
    MARKET_DATA_PROMPT,
    NETWORK_GRAPH_RAG_PROMPT,
    PORTFOLIO_OPTIMIZATION_PROMPT,
    RISK_VOLATILITY_PROMPT,
    STRESS_TESTING_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TECHNICAL_ANALYSIS_PROMPT,
    get_agent_prompt,
    get_chatbot_prompt,
    get_prompt,
    list_available_prompts,
)


# =====================================================================
# 2. SPECIALIZED AGENT ROUTING & CLASSIFICATION
# =====================================================================

def classify_agent_for_query(user_text: str, state: dict | None = None) -> str:
    """
    Deterministic router that maps a user query to the most appropriate specialized agent.
    Falls back to 'chatbot' (conversational supervisor) for general or cross-domain queries.
    """
    text = (user_text or "").lower().strip()
    if not text:
        return "chatbot"

    # 1. Stress Testing & Crisis Simulation
    if any(k in text for k in [
        "stress test", "stress testing", "scenario analysis", "crisis replay", 
        "2008 gfc", "great financial crisis", "covid shock", "macro shock", 
        "tail risk simulation", "drawdown simulation", "stress scenario", "stagflation shock"
    ]):
        return "stress_testing_agent"

    # 2. Explainability & Board Memos
    if any(k in text for k in [
        "executive memo", "board memo", "explain weights", "why this allocation", 
        "shadow cost", "shadow price", "dual variable", "plain english", 
        "explainability", "committee memo", "fiduciary summary", "fiduciary memo"
    ]):
        return "explainability_agent"

    # 3. Data Sentinel & Quality Assurance
    if any(k in text for k in [
        "data integrity", "audit data", "data quality", "missing bars", "stale price", 
        "split anomaly", "data sentinel", "pre-flight data", "data completeness", "bad ticks"
    ]):
        return "data_sentinel_agent"

    # 4. Econometrics & Time-Series Diagnostics
    if any(k in text for k in [
        "adf", "dickey-fuller", "dickey fuller", "kpss", "stationarity", "unit root", 
        "ljung-box", "ljung box", "autocorrelation", "arima", "price forecast", 
        "forecast price", "time series test", "econometric"
    ]):
        return "econometrics_agent"

    # 5. Risk & Volatility Modeling
    if any(k in text for k in [
        "garch", "egarch", "volatility model", "volatility forecast", "full eda", 
        "stock eda", "eda", "var 95", "var 99", "value at risk", "cvar", 
        "expected shortfall", "skewness", "kurtosis", "shrinkage", "missing data heatmap",
        "correlation heatmap"
    ]):
        return "risk_volatility_agent"

    # 6. Portfolio Optimization & Capital Allocation
    if any(k in text for k in [
        "optimize", "efficient frontier", "frontier", "markowitz", "max sharpe", 
        "min vol", "risk parity", "cvar optimization", "suggested weights", 
        "rebalance", "turnover", "hhi", "allocation", "portfolio weights"
    ]):
        return "portfolio_optimizer_agent"

    # 7. Institutional Network & Graph RAG
    if any(k in text for k in [
        "institutional holder", "institutional holders", "13f", "common holders", 
        "bipartite", "graph rag", "co-holding", "coholding", "centrality", 
        "contagion", "knowledge base", "methodology", "research paper"
    ]):
        return "network_graph_rag_agent"

    # 8. Technical Analysis & Visualization
    if any(k in text for k in [
        "rsi", "macd", "bollinger", "atr", "moving average", "sma", "ema", 
        "technical analysis", "technical indicator", "technical chart", 
        "custom math plot", "financial plot"
    ]):
        return "technical_analysis_agent"

    # 9. Governance, Compliance & Audit Trail
    if any(k in text for k in [
        "audit", "compliance", "past analysis", "past weights", "user history", 
        "analysis history", "audit trail", "hitl", "governance log", "saved metric"
    ]):
        return "governance_audit_agent"

    # 10. Market Data & Universe Discovery
    if any(k in text for k in [
        "sector", "universe", "snapshot", "bundle", "yfinance", "unemployment", 
        "gdp", "market data", "database snapshot", "list stocks", "stocks in", 
        "price of", "historical price", "available universes"
    ]):
        return "market_data_agent"

    return "chatbot"


# =====================================================================
# 3. AGENT METADATA REGISTRY
# =====================================================================

SPECIALIZED_AGENTS_METADATA = {
    "market_data_agent": {
        "name": "Market Data & Universe Discovery Agent",
        "role": "Discovers universes, sectors, fundamentals, historical OHLCV data bundles and macro charts.",
        "tools": [
            "list_available_sectors",
            "list_available_universes",
            "get_stocks_by_sector",
            "get_stocks_by_universe",
            "get_universe_overview",
            "get_stock_database_snapshot",
            "get_market_data_bundle",
            "get_yfinance_market_data",
            "plot_historical_prices",
            "plot_us_economic_indicators",
        ],
    },
    "econometrics_agent": {
        "name": "Econometric & Time-Series Diagnostic Agent",
        "role": "Conducts ADF, KPSS, Ljung-Box autocorrelation tests, and ARIMA forecasting.",
        "tools": [
            "run_stationarity_and_econometric_tests",
            "forecast_price_arima",
            "get_price_series_for_analysis",
            "execute_dynamic_portfolio_code",
        ],
    },
    "risk_volatility_agent": {
        "name": "Quantitative Risk & Volatility Agent",
        "role": "Computes GARCH(1,1)/EGARCH models, VaR/CVaR, Ledoit-Wolf shrinkage, and full stock EDA.",
        "tools": [
            "fit_garch_volatility_model",
            "calculate_full_stock_eda",
            "generate_ohlc_correlation_heatmap",
            "generate_missing_data_heatmap",
            "save_financial_metric",
            "execute_dynamic_portfolio_code",
        ],
    },
    "portfolio_optimizer_agent": {
        "name": "Conic Portfolio Optimization Agent",
        "role": "Calculates Markowitz Frontier, Max Sharpe, Min Volatility, Risk Parity, and G-CVaR.",
        "tools": [
            "compute_optimal_portfolio_frontier",
            "governance_pipeline_with_cache",
            "get_detailed_past_weights",
            "save_financial_metric",
            "execute_dynamic_portfolio_code",
        ],
    },
    "network_graph_rag_agent": {
        "name": "Institutional Network & Graph RAG Agent",
        "role": "Analyzes SEC 13F ownership graphs, centrality, contagion, and methodology retrieval.",
        "tools": [
            "retrieve_graph_rag_context",
            "compare_common_institutional_holders",
            "search_methodology_knowledge_base",
        ],
    },
    "technical_analysis_agent": {
        "name": "Technical Analysis & Quantitative Plotting Agent",
        "role": "Calculates RSI, MACD, Bollinger Bands, ATR, and custom statistical charts.",
        "tools": [
            "get_price_series_for_analysis",
            "generate_financial_plot",
            "generate_custom_math_plot",
            "run_data_analysis_plot",
        ],
    },
    "governance_audit_agent": {
        "name": "Governance Policy & Audit Trail Agent",
        "role": "Audits session histories, past allocations, HITL safeguards, and compliance metrics.",
        "tools": [
            "get_user_analysis_history",
            "save_financial_metric",
        ],
    },
    "stress_testing_agent": {
        "name": "Macro Stress-Testing & Crisis Simulation Agent",
        "role": "Executes macro crisis scenarios (2008 GFC, 2020 COVID, Rates), tail CVaR, and breach analyses.",
        "tools": [
            "run_portfolio_stress_test",
            "fit_garch_volatility_model",
            "get_price_series_for_analysis",
            "save_financial_metric",
        ],
    },
    "explainability_agent": {
        "name": "Institutional Explainability & Fiduciary Communications Agent",
        "role": "Translates optimization outputs and shadow costs into executive board memos and HITL summaries.",
        "tools": [
            "generate_executive_governance_memo",
            "search_methodology_knowledge_base",
            "get_detailed_past_weights",
            "save_financial_metric",
        ],
    },
    "data_sentinel_agent": {
        "name": "Data Sentinel & Pre-Flight Quality Assurance Agent",
        "role": "Audits time series continuity, missing bars, price jumps, and certifies data readiness.",
        "tools": [
            "audit_data_integrity",
            "generate_missing_data_heatmap",
            "get_price_series_for_analysis",
            "save_financial_metric",
        ],
    },
}


def run_specialized_agent(agent_name: str, user_query: str, portfolio: list[str] | None = None) -> str:
    """Execute a specialized agent directly for fast standalone or subagent queries."""
    from src.orchestrator.chatbot_orchestrator import (
        chatbot_node,
        data_sentinel_agent_node,
        econometrics_agent_node,
        explainability_agent_node,
        governance_audit_agent_node,
        market_data_agent_node,
        network_graph_rag_agent_node,
        portfolio_optimizer_agent_node,
        risk_volatility_agent_node,
        stress_testing_agent_node,
        technical_analysis_agent_node,
    )
    node_map = {
        "market_data_agent": market_data_agent_node,
        "econometrics_agent": econometrics_agent_node,
        "risk_volatility_agent": risk_volatility_agent_node,
        "portfolio_optimizer_agent": portfolio_optimizer_agent_node,
        "network_graph_rag_agent": network_graph_rag_agent_node,
        "technical_analysis_agent": technical_analysis_agent_node,
        "governance_audit_agent": governance_audit_agent_node,
        "stress_testing_agent": stress_testing_agent_node,
        "explainability_agent": explainability_agent_node,
        "data_sentinel_agent": data_sentinel_agent_node,
        "chatbot": chatbot_node,
    }
    node = node_map.get(agent_name, chatbot_node)
    state = {
        "messages": [HumanMessage(content=user_query)],
        "user_portfolio": portfolio or [],
    }
    result = node(state)
    msgs = result.get("messages", [])
    if msgs:
        content = getattr(msgs[-1], "content", str(msgs[-1]))
        return str(content)
    return "No response produced."

