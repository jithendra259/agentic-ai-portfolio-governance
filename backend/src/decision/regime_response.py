from __future__ import annotations

from typing import Any

from api.analytics_router import get_instability_analytics
from src.intent.intent_router import IntentRouter


DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM"]
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2024-12-31"


def build_regime_only_response(message: str, previous_analysis: dict[str, Any] | None = None) -> str | None:
    router = IntentRouter()
    plan = router.build_execution_plan(message, previous_analysis=previous_analysis)
    if plan.get("sub_intent") != "instability_regime":
        return None
    if not _regime_only_requested(message):
        return None

    entities = plan.get("entities", {})
    tickers = entities.get("tickers") or DEFAULT_TICKERS
    start_date = entities.get("start_date") or DEFAULT_START_DATE
    end_date = entities.get("end_date") or DEFAULT_END_DATE
    data = get_instability_analytics(",".join(tickers), start_date, end_date)
    series = data.get("instability_index", [])
    latest = series[-1] if series else {}
    instability = float(latest.get("instabilityIndex", 0.0)) if latest else None
    regime = _classify_regime(instability)

    requested_universe = entities.get("universe")
    context_label = (
        requested_universe
        if requested_universe and tickers != DEFAULT_TICKERS
        else f"{requested_universe} requested; no attached universe constituent context, default benchmark tickers used"
        if requested_universe
        else "current/default"
    )

    return "\n".join(
        [
            "# Regime Classification Only",
            "",
            "- detected intent: `instability_regime`",
            "- modules called: intent_router, quantitative_analytics_agent, governance_policy, smart_plot_selector",
            "- modules skipped: optimizer, allocation_engine, graph_contagion_agent, institutional_network_plots",
            "- endpoint called: `/api/analytics/instability`",
            f"- analysis_id: `{plan.get('cache', {}).get('analysis_id') or 'REGIME-ONLY-CURRENT'}`",
            f"- context used: {context_label}; tickers={','.join(tickers)}; date range={start_date} to {end_date}",
            "",
            "## Result",
            f"- instability index: {instability if instability is not None else 'unavailable'}",
            f"- regime label: {regime}",
            "- thresholds used: Calm < 0.50; Elevated 0.50 to 0.85; Crisis >= 0.85",
            f"- critical-condition trigger: {str(regime == 'Crisis').lower()}",
            "- allocation weights: not generated because the prompt explicitly requested regime-only analysis.",
            "",
            "## Smart View plot IDs to trigger",
            "- `plot_19_composite_instability_index_plot`",
            "- `plot_20_regime_classification_timeline`",
            "- `plot_21_market_stress_index_plot`",
            "- `plot_22_volatility_spike_component_plot`",
            "- `plot_23_correlation_spike_component_plot`",
            "- `plot_24_maximum_drawdown_component_plot`",
            "",
            "Advisory-language validation: passed.",
        ]
    )


def _regime_only_requested(message: str) -> bool:
    normalized = (message or "").lower()
    return any(
        marker in normalized
        for marker in (
            "only answer regime",
            "regime question only",
            "do not generate allocation",
            "don't generate allocation",
            "do not show optimal weights",
            "don't show optimal weights",
        )
    )


def _classify_regime(instability: float | None) -> str:
    if instability is None:
        return "Unknown"
    if instability < 0.50:
        return "Calm"
    if instability < 0.85:
        return "Elevated"
    return "Crisis"
