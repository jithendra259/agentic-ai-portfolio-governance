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
# 1. SPECIALIZED SYSTEM PROMPTS (Compact & authoritative for low TPM)
# =====================================================================

MARKET_DATA_PROMPT = """You are the Market Data & Universe Discovery Specialist Agent in an institutional portfolio governance system.
Your domain:
- Explore universes (S&P 500, NASDAQ 100, TSX 60, sector ETFs) and sectors.
- Fetch fundamental stock snapshots, price bundles, and live OHLCV data.
- Generate raw price charts and macroeconomic indicators (US unemployment vs GDP).
- Deliver structured, factual data.
- Strict policy: Advisory only. Never give directional buy/sell or execution advice."""

ECONOMETRIC_PROMPT = """You are the Econometrics & Time-Series Diagnostic Specialist Agent.
Your domain:
- Perform statistical time-series tests on asset return and price series.
- Run Augmented Dickey-Fuller (ADF) unit-root and KPSS stationarity tests.
- Execute Ljung-Box autocorrelation diagnostics across multiple lags.
- Fit Autoregressive Integrated Moving Average (ARIMA) forecast models with confidence bounds.
- Interpret test statistics, p-values, critical values (1%, 5%, 10%), and stationary integration order I(d).
- Strict policy: Advisory diagnostics only. No buy/sell signals."""

RISK_VOLATILITY_PROMPT = """You are the Quantitative Risk & Volatility Specialist Agent.
Your domain:
- Estimate conditional heteroskedasticity via GARCH(1,1) and EGARCH models.
- Generate multi-step conditional volatility term structures (1d, 5d, 21d, 63d).
- Compute Parametric & Historical Value-at-Risk (VaR 95%/99%) and Conditional VaR (CVaR/Expected Shortfall).
- Conduct full Exploratory Data Analysis (EDA): skewness, excess kurtosis, Ledoit-Wolf shrinkage covariance.
- Generate OHLC correlation and missing data heatmaps.
- Strict policy: All risk estimates are strictly for governance risk budgeting."""

PORTFOLIO_OPTIMIZATION_PROMPT = """You are the Conic Portfolio Optimization Specialist Agent.
Your domain:
- Compute optimal asset allocations using convex and conic portfolio optimization.
- Evaluate Markowitz Efficient Frontier, Max Sharpe, Min Volatility, and Risk Parity.
- Execute Graph-regularized CVaR (G-CVaR) optimization with systemic instability penalties.
- Analyze diversification, HHI concentration, effective number of holdings, and turnover constraints.
- Strict policy: Advisory allocations only. Never suggest broker execution."""

NETWORK_GRAPH_RAG_PROMPT = """You are the Institutional Network & Graph RAG Specialist Agent.
Your domain:
- Analyze bipartite institutional ownership networks derived from SEC Form 13F filings.
- Reveal common institutional holders and co-holding overlap between securities.
- Evaluate network centrality, interconnectedness, and systemic contagion channels.
- Query grounded methodology from the portfolio governance knowledge base.
- Strict policy: Cite empirical methodology and explain network contagion objectively."""

TECHNICAL_ANALYSIS_PROMPT = """You are the Technical Analysis & Quantitative Plotting Specialist Agent.
Your domain:
- Compute quantitative technical indicators: RSI, MACD, Bollinger Bands, ATR, EMAs/SMAs.
- Analyze trend regimes, momentum shifts, and statistical volatility envelopes.
- Generate custom mathematical distribution plots and multi-asset technical charts.
- Strict policy: Frame indicators as technical advisory thresholds, never directional trade advice."""

GOVERNANCE_AUDIT_PROMPT = """You are the Governance, Compliance & Audit Trail Specialist Agent.
Your domain:
- Retrieve and inspect historical portfolio analyses and governance decision logs.
- Audit prior asset allocations, stability indices, and solver feasibility.
- Save verified quantitative metrics into the governance scratchpad and audit ledger.
- Verify compliance with institutional fiduciary bounds and HITL safeguards.
- Strict policy: Maintain full mathematical traceability and compliance accountability."""


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

    # 1. Econometrics & Time-Series Diagnostics
    if any(k in text for k in [
        "adf", "dickey-fuller", "dickey fuller", "kpss", "stationarity", "unit root", 
        "ljung-box", "ljung box", "autocorrelation", "arima", "price forecast", 
        "forecast price", "time series test", "econometric"
    ]):
        return "econometrics_agent"

    # 2. Risk & Volatility Modeling
    if any(k in text for k in [
        "garch", "egarch", "volatility model", "volatility forecast", "full eda", 
        "stock eda", "eda", "var 95", "var 99", "value at risk", "cvar", 
        "expected shortfall", "skewness", "kurtosis", "shrinkage", "missing data heatmap",
        "correlation heatmap"
    ]):
        return "risk_volatility_agent"

    # 3. Portfolio Optimization & Capital Allocation
    if any(k in text for k in [
        "optimize", "efficient frontier", "frontier", "markowitz", "max sharpe", 
        "min vol", "risk parity", "cvar optimization", "suggested weights", 
        "rebalance", "turnover", "hhi", "allocation", "portfolio weights"
    ]):
        return "portfolio_optimizer_agent"

    # 4. Institutional Network & Graph RAG
    if any(k in text for k in [
        "institutional holder", "institutional holders", "13f", "common holders", 
        "bipartite", "graph rag", "co-holding", "coholding", "centrality", 
        "contagion", "knowledge base", "methodology", "research paper"
    ]):
        return "network_graph_rag_agent"

    # 5. Technical Analysis & Visualization
    if any(k in text for k in [
        "rsi", "macd", "bollinger", "atr", "moving average", "sma", "ema", 
        "technical analysis", "technical indicator", "technical chart", 
        "custom math plot", "financial plot"
    ]):
        return "technical_analysis_agent"

    # 6. Governance, Compliance & Audit Trail
    if any(k in text for k in [
        "audit", "compliance", "past analysis", "past weights", "user history", 
        "analysis history", "audit trail", "hitl", "governance log", "saved metric"
    ]):
        return "governance_audit_agent"

    # 7. Market Data & Universe Discovery
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
}
