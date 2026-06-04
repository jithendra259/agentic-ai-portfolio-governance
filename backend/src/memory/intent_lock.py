from __future__ import annotations

from typing import Any


INTENT_LOCKS = {
    "data_quality_only": {
        "allowed_modules": [
            "missing_value_check",
            "duplicate_date_check",
            "stale_record_check",
            "abnormal_jump_check",
            "data_quality_plots",
        ],
        "blocked_modules": [
            "optimizer",
            "advisory_allocation",
            "recommended_weights",
            "rebalance_analysis",
            "broker_action",
        ],
    },
    "diversification_diagnosis_only": {
        "allowed_modules": [
            "concentration",
            "sector_exposure",
            "hhi",
            "effective_holdings",
            "correlation",
            "ownership_overlap",
            "diversification_plots",
        ],
        "blocked_modules": ["optimizer", "advisory_allocation", "recommended_weights", "broker_action"],
    },
    "regime_diagnosis_only": {
        "allowed_modules": [
            "instability_index",
            "volatility_spike",
            "correlation_spike",
            "drawdown_component",
            "regime_plots",
        ],
        "blocked_modules": ["optimizer", "advisory_allocation", "optimal_weights", "recommended_weights", "broker_action"],
    },
    "plot_only": {
        "allowed_modules": ["requested_plot", "plot_validation"],
        "blocked_modules": ["optimizer", "advisory_allocation", "extra_recommendations", "broker_action"],
    },
    "plot_plus_diagnosis": {
        "allowed_modules": ["requested_plot", "diagnosis", "plot_validation"],
        "blocked_modules": ["optimizer", "broker_action"],
    },
    "advisory_allocation": {
        "allowed_modules": ["optimizer", "advisory_allocation", "constraint_validation", "allocation_plots"],
        "blocked_modules": ["broker_action"],
    },
    "backtest_evaluation": {
        "allowed_modules": ["backtesting", "strategy_metrics", "benchmark_comparison"],
        "blocked_modules": ["broker_action"],
    },
    "general_question": {
        "allowed_modules": ["answer_question", "methodology_lookup"],
        "blocked_modules": ["optimizer", "advisory_allocation", "broker_action"],
    },
}


def build_intent_lock(message: str, router_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = " ".join(str(message or "").lower().split())
    sub_intent = (router_plan or {}).get("sub_intent")

    if "only data quality" in normalized or normalized.startswith("data quality only"):
        mode = "data_quality_only"
    elif "only diversification" in normalized or "diversification diagnosis only" in normalized:
        mode = "diversification_diagnosis_only"
    elif (
        "only regime" in normalized
        or "regime question only" in normalized
        or "only stress" in normalized
        or "only calm" in normalized
        or "only elevated" in normalized
        or "only crisis" in normalized
        or "do not generate allocation" in normalized
        or "do not generate advisory allocation" in normalized
    ):
        mode = "regime_diagnosis_only" if sub_intent == "instability_regime" or "regime" in normalized else "general_question"
    elif "do not run optimization" in normalized or "don't run optimization" in normalized:
        mode = "plot_plus_diagnosis" if "plot" in normalized else "general_question"
    elif normalized.startswith("plot ") or normalized.startswith("show plot ") or "plot only" in normalized:
        mode = "plot_only"
    elif "advisory allocation" in normalized or sub_intent == "advisory_allocation":
        mode = "advisory_allocation"
    elif "backtest" in normalized or sub_intent == "backtesting":
        mode = "backtest_evaluation"
    elif sub_intent == "diversification":
        mode = "diversification_diagnosis_only"
    elif sub_intent == "data_quality":
        mode = "data_quality_only"
    elif sub_intent == "instability_regime":
        mode = "plot_plus_diagnosis"
    else:
        mode = "general_question"

    lock = INTENT_LOCKS[mode]
    blocked = list(lock["blocked_modules"])
    if "do not generate advisory allocation" in normalized and "advisory_allocation" not in blocked:
        blocked.extend(["optimizer", "advisory_allocation", "recommended_weights"])
    if "do not run optimization" in normalized and "optimizer" not in blocked:
        blocked.append("optimizer")

    return {
        "intent": mode,
        "allowed_modules": list(lock["allowed_modules"]),
        "blocked_modules": sorted(set(blocked)),
        "response_scope": "only_answer_user_question",
    }


def is_module_blocked(intent_lock: dict[str, Any], module_name: str) -> bool:
    blocked = {str(item).lower() for item in intent_lock.get("blocked_modules", [])}
    return str(module_name or "").lower() in blocked
