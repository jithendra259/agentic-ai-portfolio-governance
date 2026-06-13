from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from src.intent.intent_schema import (
    RouterCache,
    RouterClarification,
    RouterEntities,
    RouterExecution,
    RouterPlan,
    RouterResponse,
    RouterSafety,
    RouterTarget,
)


ANALYTICS_ENDPOINTS = {
    "data_quality": "/api/analytics/eda",
    "stock_eda_full": "/api/analytics/stock-eda-full",
    "eda": "/api/analytics/eda",
    "correlation_covariance": "/api/analytics/eda",
    "instability_regime": "/api/analytics/instability",
    "diversification": "/api/analytics/diversification",
    "risk_governance": "/api/analytics/risk-governance",
    "advisory_allocation": "/api/governance/decision/advisory-allocation",
    "graph_contagion": "/api/analytics/contagion",
    "hitl_governance": "/api/analytics/agent-governance",
    "backtesting": "/api/analytics/backtesting",
    "smart_plot_selection": "/api/governance/decision/plots",
    "full_plot_coverage": "/api/governance/decision/plots",
    "response_validation": "/api/governance/decision/validate",
}

DEFAULT_TABS = {
    "data_quality": "Data EDA",
    "stock_eda_full": "Full Stock EDA",
    "eda": "Data EDA",
    "correlation_covariance": "Correlation & Covariance EDA",
    "instability_regime": "Instability Monitor",
    "diversification": "Diversification Diagnostics",
    "risk_governance": "Risk Governance",
    "advisory_allocation": "Advisory Diversification",
    "graph_contagion": "Contagion Graph Analysis",
    "hitl_governance": "Agent Governance and Explainability",
    "backtesting": "Evaluation and Backtesting",
    "smart_plot_selection": "Smart View",
    "full_plot_coverage": "Full Analytics View",
    "response_validation": "Agent Governance and Explainability",
}

MODULE_MATRIX = {
    "data_quality": dict(math=True, optimizer=False, rag=False, plot=True, validator=True, frontend=True),
    "stock_eda_full": dict(math=True, optimizer=False, rag=False, plot=True, validator=True, frontend=True),
    "eda": dict(math=True, optimizer=False, rag=False, plot=True, validator=True, frontend=True),
    "correlation_covariance": dict(math=True, optimizer=False, rag=False, plot=True, validator=True, frontend=True),
    "instability_regime": dict(math=True, optimizer=False, rag=False, plot=True, validator=True, frontend=True),
    "diversification": dict(math=True, optimizer=False, rag=False, plot=True, validator=True, frontend=True),
    "risk_governance": dict(math=True, optimizer=False, rag=False, plot=True, validator=True, frontend=True),
    "advisory_allocation": dict(math=True, optimizer=True, rag=False, plot=True, validator=True, frontend=True),
    "graph_contagion": dict(math=True, optimizer=False, rag=True, plot=True, validator=True, frontend=True),
    "hitl_governance": dict(math=True, optimizer=False, rag=False, plot=True, validator=True, frontend=True),
    "backtesting": dict(math=True, optimizer=False, rag=False, plot=True, validator=True, frontend=True),
    "smart_plot_selection": dict(math=False, optimizer=False, rag=False, plot=True, validator=True, frontend=True),
    "full_plot_coverage": dict(math=False, optimizer=False, rag=False, plot=True, validator=True, frontend=True),
    "response_validation": dict(math=False, optimizer=False, rag=False, plot=False, validator=True, frontend=False),
}

SUB_INTENT_RULES = [
    ("response_validation", 0.96, ("validate your last", "validate my last", "check your last", "forbidden", "numeric claims")),
    ("stock_eda_full", 0.94, ("full stock eda", "complete stock eda", "notebook eda", "seasonal stock analysis", "skewness", "kurtosis", "day-of-week", "quarterly returns", "monthly returns", "ohlcv eda", "volume trend")),
    ("full_plot_coverage", 0.95, ("all plots", "full analytics", "88 plots", "plot coverage", "render all")),
    ("smart_plot_selection", 0.93, ("important plots", "smart view", "most relevant plots", "only relevant", "best plots")),
    ("advisory_allocation", 0.92, ("advisory allocation", "allocation change", "recommended allocation", "current allocation", "weights sum")),
    ("graph_contagion", 0.9, ("graph centrality", "centrality", "co-ownership", "contagion", "institutional overlap", "shared holder")),
    ("hitl_governance", 0.89, ("hitl", "human in the loop", "audit trail", "governance log", "decision log")),
    ("backtesting", 0.9, ("backtest", "benchmark", "performance comparison", "equal weight", "transaction cost", "oos sharpe")),
    ("risk_governance", 0.9, ("cvar", "var ", "drawdown", "sharpe", "sortino", "downside risk", "risk contribution")),
    ("instability_regime", 0.88, ("critical", "crisis", "regime", "instability", "stress score", "volatility spike")),
    ("diversification", 0.88, ("diversified", "diversification", "hhi", "effective number", "concentration", "sector exposure")),
    ("correlation_covariance", 0.88, ("correlation", "covariance", "pca", "eigenvalue", "eigenvalues")),
    ("data_quality", 0.86, ("data quality", "missing values", "stale data", "duplicate dates", "imputation")),
    ("eda", 0.84, ("eda", "log returns", "cumulative returns", "return distribution", "rolling volatility", "outlier")),
    ("chatbot_accuracy_debug", 0.82, ("debug", "not working", "router", "plot not showing", "backend")),
]

METHODOLOGY_TERMS = (
    "what is",
    "explain",
    "define",
    "methodology",
    "formula",
    "how does",
    "why does",
)

METRIC_ALIASES = {
    "cvar": "cvar",
    "var": "var",
    "drawdown": "drawdown",
    "sharpe": "sharpe",
    "sortino": "sortino",
    "hhi": "hhi",
    "volatility": "volatility",
    "correlation": "correlation",
    "covariance": "covariance",
    "instability": "instability",
    "centrality": "centrality",
}

PLOT_NAME_MARKERS = (
    "heatmap",
    "scatter",
    "sankey",
    "funnel",
    "radar",
    "gauge",
    "pie",
    "bar",
    "line",
    "candlestick",
    "radial",
)

INSTABILITY_ALLOWED_PLOTS = [
    "plot_19_composite_instability_index_plot",
    "plot_20_regime_classification_timeline",
    "plot_21_market_stress_index_plot",
    "plot_22_volatility_spike_component_plot",
    "plot_23_correlation_spike_component_plot",
    "plot_24_maximum_drawdown_component_plot",
    "plot_25_instability_component_contribution_plot",
    "plot_26_crisis_window_activation_plot",
    "plot_27_regime_frequency_plot",
    "plot_28_threshold_sensitivity_plot",
]

NEGATIVE_ALLOCATION_MARKERS = (
    "do not generate allocation",
    "don't generate allocation",
    "do not show optimal weights",
    "don't show optimal weights",
    "only answer regime question",
    "regime question only",
)

GRAPH_REQUEST_MARKERS = (
    "graph",
    "centrality",
    "co-ownership",
    "institutional",
    "contagion",
)


@dataclass(frozen=True)
class RuleMatch:
    sub_intent: str
    confidence: float
    matched_keywords: list[str]


def build_router_plan(user_message: str, previous_analysis: dict | None = None) -> dict:
    query = str(user_message or "").strip()
    normalized = re.sub(r"\s+", " ", query.lower())
    entities = _extract_entities(query, normalized)
    match = _match_sub_intent(normalized)
    main_intent = _main_intent_for(match.sub_intent, normalized)

    if main_intent == "methodology_explanation":
        execution = RouterExecution(needs_rag=True, needs_validator=True)
        routing = RouterTarget(endpoint=None, default_tab=None, plot_mode="none", max_plots=0)
        safety = _safety_for("methodology_explanation")
    elif main_intent == "dashboard_navigation":
        execution = RouterExecution(needs_validator=False, needs_frontend_trigger=True)
        routing = RouterTarget(endpoint=None, default_tab=_dashboard_tab_from_text(normalized), plot_mode="none", max_plots=0)
        safety = _safety_for(match.sub_intent)
    elif match.sub_intent == "unknown":
        execution = RouterExecution(needs_validator=True)
        routing = RouterTarget()
        safety = _safety_for(match.sub_intent)
    else:
        execution = _execution_for(match.sub_intent)
        routing = _routing_for(match.sub_intent)
        safety = _safety_for(match.sub_intent)

    if match.sub_intent == "instability_regime" and _is_regime_only(normalized):
        execution.needs_optimizer = False
        execution.needs_allocation = False
        execution.needs_graph_network = _explicit_graph_requested(normalized)
        execution.needs_rag = False
        routing.allowed_plots = INSTABILITY_ALLOWED_PLOTS.copy()
        routing.max_plots = len(routing.allowed_plots)
        for skipped in ("optimizer", "allocation_engine"):
            if skipped not in safety.modules_skipped:
                safety.modules_skipped.append(skipped)
        if not execution.needs_graph_network and "graph_contagion_agent" not in safety.modules_skipped:
            safety.modules_skipped.append("graph_contagion_agent")

    response = _response_for(entities.response_mode, main_intent, match.sub_intent)
    cache = _cache_for(previous_analysis, entities, normalized)
    if cache.reuse_previous_analysis and previous_analysis:
        prev_entities = previous_analysis.get("entities", {})
        if not entities.tickers:
            entities.tickers = list(prev_entities.get("tickers", []))
        entities.universe = entities.universe or prev_entities.get("universe")
        if not entities.current_weights:
            entities.current_weights = dict(prev_entities.get("current_weights", {}))
        entities.start_date = entities.start_date or prev_entities.get("start_date")
        entities.end_date = entities.end_date or prev_entities.get("end_date")
        entities.benchmark = entities.benchmark or prev_entities.get("benchmark")

    clarification = _clarification_for(match.confidence, main_intent, match.sub_intent)

    plan = RouterPlan(
        main_intent=main_intent,
        sub_intent=match.sub_intent,
        confidence=match.confidence,
        entities=entities,
        execution=execution,
        routing=routing,
        response=response,
        safety=safety,
        cache=cache,
        clarification=clarification,
    )
    return plan.to_dict()


def _match_sub_intent(normalized: str) -> RuleMatch:
    if not normalized:
        return RuleMatch("unknown", 0.0, [])

    matched = []
    for sub_intent, confidence, keywords in SUB_INTENT_RULES:
        hits = [keyword for keyword in keywords if keyword in normalized]
        if hits:
            return RuleMatch(sub_intent, confidence, hits)
        matched.extend(hits)

    if "dashboard" in normalized or normalized.startswith("open "):
        return RuleMatch("smart_plot_selection", 0.82, ["dashboard"])

    methodology_metric = any(term in normalized for term in METHODOLOGY_TERMS) and any(
        metric in normalized for metric in METRIC_ALIASES
    )
    if methodology_metric:
        return RuleMatch("unknown", 0.86, ["methodology_metric"])

    if any(word in normalized for word in ("portfolio", "analyze", "analyse", "risk", "governance")):
        return RuleMatch("unknown", 0.52, ["domain_general"])

    return RuleMatch("unknown", 0.35, matched)


def _main_intent_for(sub_intent: str, normalized: str) -> str:
    if sub_intent == "response_validation":
        return "response_validation"
    if sub_intent == "chatbot_accuracy_debug":
        return "system_debug"
    if "dashboard" in normalized or normalized.startswith("open "):
        return "dashboard_navigation"
    if sub_intent == "graph_contagion":
        return "portfolio_analysis"
    if any(term in normalized for term in METHODOLOGY_TERMS) and (
        sub_intent == "unknown" or any(metric in normalized for metric in METRIC_ALIASES)
    ):
        return "methodology_explanation"
    if sub_intent in {"smart_plot_selection", "full_plot_coverage"}:
        return "plot_request"
    if sub_intent == "advisory_allocation":
        return "advisory_allocation"
    if sub_intent == "unknown":
        return "general_chat"
    return "portfolio_analysis"


def _extract_entities(query: str, normalized: str) -> RouterEntities:
    tickers = _extract_tickers(query)
    start_date, end_date = _extract_date_range(normalized)
    return RouterEntities(
        universe=_extract_universe(normalized),
        tickers=tickers,
        current_weights=_extract_weights(query),
        start_date=start_date,
        end_date=end_date,
        benchmark=_extract_benchmark(query),
        sectors=_extract_sectors(normalized),
        metrics=_extract_metrics(normalized),
        plot_names=[name for name in PLOT_NAME_MARKERS if name in normalized],
        response_mode=_extract_response_mode(normalized),
        user_intent_keywords=[keyword for keyword in METRIC_ALIASES if keyword in normalized],
    )


def _extract_tickers(query: str) -> list[str]:
    stopwords = {
        "CV", "CVAR", "VAR", "HHI", "PCA", "EDA", "HITL", "RAG", "LLM", "API",
        "WHY", "SHOW", "GIVE", "ONLY", "ALL", "NOW", "AND", "FOR", "FROM", "TO",
        "THE", "RISK", "WHAT", "IS", "MY", "YOUR", "LAST", "THIS", "THAT",
        "DO", "NOT", "DONT", "DON", "NO", "ANSWER", "QUESTION",
        "DAY", "WEEK", "MONTH", "YEAR", "TREND", "TRENDS", "RETURN", "RETURNS",
        "VOLUME", "SECTOR", "STOCK", "STOCKS", "FULL", "COMPLETE",
    }
    
    # 1. Try to find explicit lists like ["AAPL", "MSFT"] or [AAPL, MSFT]
    list_matches = re.findall(r"\[\s*([^\]]+)\s*\]", query)
    if list_matches:
        explicit_tickers = []
        for match in list_matches:
            words = re.findall(r"\b[A-Z]{1,5}\b", match.upper())
            for w in words:
                if w not in stopwords and not (w.startswith("U") and w[1:].isdigit()):
                    if w not in explicit_tickers:
                        explicit_tickers.append(w)
        if explicit_tickers:
            return explicit_tickers

    # 2. Otherwise, check word-by-word with context rules
    query_upper = query.upper()
    # Normalize punctuation to spaces except keep letters/numbers
    normalized_query = re.sub(r"[^A-Z0-9\s]", " ", query_upper)
    tokens = normalized_query.split()
    
    context_preceding = {
        "TICKER", "TICKERS", "COMPARE", "COMPARING", "COMPARISON", "FOR",
        "BETWEEN", "OF", "SYMBOL", "SYMBOLS", "USED", "TICKERS_USED"
    }
    context_any = {
        "AND", "WITH", "VERSUS", "VS"
    }
    context_keywords = context_preceding.union(context_any)
    
    # Find all candidate tokens (their indices in tokens)
    candidate_indices = []
    for idx, token in enumerate(tokens):
        if re.match(r"^[A-Z]{1,5}$", token) and token not in stopwords and token not in context_keywords:
            if not (token.startswith("U") and token[1:].isdigit()):
                candidate_indices.append(idx)
                
    # Now, identify which candidates have direct context or are known tickers
    accepted_indices = set()
    for idx in candidate_indices:
        token = tokens[idx]
        # Check if known universe ticker
        is_known = False
        from src.memory.session_state import KNOWN_UNIVERSE_TICKERS
        for u_tickers in KNOWN_UNIVERSE_TICKERS.values():
            if token in u_tickers:
                is_known = True
                break
        if is_known:
            accepted_indices.add(idx)
            continue
            
        # Check context
        has_context = False
        # Preceding context within 2 words
        for offset in (1, 2):
            if idx - offset >= 0:
                prev_token = tokens[idx - offset]
                if prev_token in context_preceding or prev_token in context_any:
                    has_context = True
                    break
        # Succeeding context within 2 words
        if not has_context:
            for offset in (1, 2):
                if idx + offset < len(tokens):
                    next_token = tokens[idx + offset]
                    if next_token in context_any:
                        has_context = True
                        break
        if has_context:
            accepted_indices.add(idx)
            
    # Propagate: if a candidate index is adjacent to an accepted candidate index
    # (i.e. within 2 positions), we accept it to handle lists.
    changed = True
    while changed:
        changed = False
        for idx in candidate_indices:
            if idx in accepted_indices:
                continue
            for accepted_idx in accepted_indices:
                if abs(idx - accepted_idx) <= 2:
                    accepted_indices.add(idx)
                    changed = True
                    break
                    
    # Finally, build the list of tickers in order of appearance
    tickers = []
    for idx in sorted(accepted_indices):
        token = tokens[idx]
        if token not in tickers:
            tickers.append(token)
            
    return tickers


def _extract_universe(normalized: str) -> str | None:
    match = re.search(r"\b(u\d{1,2})\b", normalized)
    return match.group(1).upper() if match else None


def _extract_weights(query: str) -> dict[str, float]:
    weights = {}
    for ticker, value in re.findall(r"\b([A-Z]{1,5})\b\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%", query.upper()):
        try:
            weights[ticker] = float(value) / 100.0
        except ValueError:
            continue
    return weights


def _extract_date_range(normalized: str) -> tuple[str | None, str | None]:
    iso_dates = re.findall(r"\b(20\d{2}|19\d{2})-(\d{2})-(\d{2})\b", normalized)
    if len(iso_dates) >= 2:
        start = "-".join(iso_dates[0])
        end = "-".join(iso_dates[1])
        return start, end

    year_match = re.search(r"\bfrom\s+((?:19|20)\d{2})\s+(?:to|through|-)\s+((?:19|20)\d{2})\b", normalized)
    if year_match:
        return f"{year_match.group(1)}-01-01", f"{year_match.group(2)}-12-31"

    years = re.findall(r"\b((?:19|20)\d{2})\b", normalized)
    if len(years) >= 2:
        return f"{years[0]}-01-01", f"{years[1]}-12-31"
    return None, None


def _extract_benchmark(query: str) -> str | None:
    match = re.search(r"\bbenchmark(?:ed)?\s+(?:against|to|with)?\s*([A-Z]{2,6})\b", query.upper())
    return match.group(1) if match else None


def _extract_sectors(normalized: str) -> list[str]:
    sectors = []
    for sector in ("technology", "financial services", "healthcare", "energy", "utilities", "industrials", "real estate"):
        if sector in normalized:
            sectors.append(sector)
    return sectors


def _extract_metrics(normalized: str) -> list[str]:
    metrics = []
    for marker, metric in METRIC_ALIASES.items():
        if re.search(rf"\b{re.escape(marker)}\b", normalized) and metric not in metrics:
            metrics.append(metric)
    return metrics


def _extract_response_mode(normalized: str) -> str:
    if any(term in normalized for term in ("quick", "brief", "short")):
        return "brief"
    if "viva" in normalized or "thesis defense" in normalized:
        return "viva"
    if "debug" in normalized:
        return "debug"
    if "full report" in normalized or "complete analysis" in normalized:
        return "full_report"
    if "formula" in normalized or "technical" in normalized:
        return "technical"
    return "standard"


def _execution_for(sub_intent: str) -> RouterExecution:
    config = MODULE_MATRIX.get(sub_intent, {})
    return RouterExecution(
        needs_math_agent=bool(config.get("math")),
        needs_optimizer=bool(config.get("optimizer")),
        needs_allocation=sub_intent == "advisory_allocation",
        needs_graph_network=sub_intent == "graph_contagion",
        needs_rag=bool(config.get("rag")),
        needs_plot_selector=bool(config.get("plot")),
        needs_validator=bool(config.get("validator", True)),
        needs_frontend_trigger=bool(config.get("frontend")),
    )


def _routing_for(sub_intent: str) -> RouterTarget:
    if sub_intent == "full_plot_coverage":
        return RouterTarget(ANALYTICS_ENDPOINTS[sub_intent], DEFAULT_TABS[sub_intent], "full_analytics", 88)
    if sub_intent == "smart_plot_selection":
        return RouterTarget(ANALYTICS_ENDPOINTS[sub_intent], DEFAULT_TABS[sub_intent], "smart_view", 6)
    if sub_intent == "response_validation":
        return RouterTarget(ANALYTICS_ENDPOINTS[sub_intent], DEFAULT_TABS[sub_intent], "none", 0)
    return RouterTarget(
        endpoint=ANALYTICS_ENDPOINTS.get(sub_intent),
        default_tab=DEFAULT_TABS.get(sub_intent),
        plot_mode="smart_view" if sub_intent in ANALYTICS_ENDPOINTS else "none",
        max_plots=6 if sub_intent in ANALYTICS_ENDPOINTS else 0,
        allowed_plots=INSTABILITY_ALLOWED_PLOTS.copy() if sub_intent == "instability_regime" else [],
    )


def _response_for(mode: str, main_intent: str, sub_intent: str) -> RouterResponse:
    return RouterResponse(
        mode=mode,
        include_formulas=mode in {"technical", "viva", "full_report"} or main_intent == "methodology_explanation",
        include_traceability=sub_intent not in {"smart_plot_selection"},
        max_bullets=4 if mode == "brief" else 12 if mode in {"viva", "full_report"} else 8,
    )


def _safety_for(sub_intent: str) -> RouterSafety:
    skipped = ["raw_price_rows", "all_88_plot_registry", "full_knowledge_base"]
    if sub_intent != "advisory_allocation":
        skipped.append("optimizer")
    if sub_intent != "graph_contagion":
        skipped.append("graph_contagion_agent")
    if sub_intent not in {"backtesting", "full_plot_coverage"}:
        skipped.append("backtesting_engine")
    return RouterSafety(modules_skipped=skipped, rag_top_k=3)


def _cache_for(previous_analysis: dict | None, entities: RouterEntities, normalized: str) -> RouterCache:
    if not previous_analysis:
        return RouterCache()
    follow_up_markers = ("now ", "explain", "what about", "show me", "risk contribution", "why", "only answer", "regime question")
    if any(marker in normalized for marker in follow_up_markers):
        return RouterCache(True, previous_analysis.get("analysis_id"))

    previous_entities = previous_analysis.get("entities", {})
    same_tickers = entities.tickers and entities.tickers == previous_entities.get("tickers", [])
    same_dates = (
        entities.start_date == previous_entities.get("start_date")
        and entities.end_date == previous_entities.get("end_date")
    )
    if same_tickers and same_dates:
        return RouterCache(True, previous_analysis.get("analysis_id"))
    return RouterCache()


def _is_regime_only(normalized: str) -> bool:
    if any(marker in normalized for marker in NEGATIVE_ALLOCATION_MARKERS):
        return True
    return "regime" in normalized and not any(term in normalized for term in ("allocation", "weights", "optimize"))


def _explicit_graph_requested(normalized: str) -> bool:
    return any(marker in normalized for marker in GRAPH_REQUEST_MARKERS)


def _clarification_for(confidence: float, main_intent: str, sub_intent: str) -> RouterClarification:
    if confidence < 0.5:
        return RouterClarification(
            True,
            "Do you want diversification analysis, risk analysis, advisory allocation, or plot selection?",
        )
    if 0.5 <= confidence < 0.8 and sub_intent == "unknown" and main_intent != "methodology_explanation":
        return RouterClarification(
            True,
            "Should I run a safe portfolio summary, diversification diagnostics, or downside-risk analysis?",
        )
    return RouterClarification()


def _dashboard_tab_from_text(normalized: str) -> str | None:
    for tab in DEFAULT_TABS.values():
        if tab.lower() in normalized:
            return tab
    return "Smart View"
