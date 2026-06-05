from __future__ import annotations

from typing import Any


def build_response_contract(
    message: str,
    resolved: dict[str, Any],
    result: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = resolved.get("session_state", {})
    context = resolved.get("resolved_context", {})
    lock = resolved.get("intent_lock", {})
    fallback = resolved.get("fallback_result", {})
    plot_payload = fallback.get("plot_payload")
    validation = resolved.get("validation_result") or fallback.get("validation_result") or {}

    return {
        "user_intent": lock.get("intent") or state.get("last_sub_intent") or state.get("last_intent"),
        "resolved_context": {
            "universe": state.get("active_universe"),
            "tickers": state.get("active_tickers", []),
            "ticker_count": state.get("active_ticker_count", 0),
            "date_range": state.get("active_date_range", {}),
        },
        "allowed_actions": lock.get("allowed_modules", []),
        "blocked_actions": lock.get("blocked_modules", []),
        "data_used": _data_used(fallback, state),
        "fallback_used": bool(fallback.get("fallback_used")),
        "missing_inputs": state.get("missing_inputs", []),
        "planner": state.get("current_plan"),
        "result": result or fallback.get("status") or "",
        "plots": [plot_payload] if isinstance(plot_payload, dict) and validation.get("can_render", True) else [],
        "warnings": state.get("warnings", []) + validation.get("warnings", []),
        "next_options": validation.get("next_options", []),
        "debug_user_message": message,
    }


def contract_summary(contract: dict[str, Any]) -> str:
    plot_count = len(contract.get("plots", []))
    missing = ", ".join(contract.get("missing_inputs", [])) or "none"
    return (
        f"intent={contract.get('user_intent')}; "
        f"universe={contract.get('resolved_context', {}).get('universe')}; "
        f"ticker_count={contract.get('resolved_context', {}).get('ticker_count')}; "
        f"plots={plot_count}; missing={missing}; "
        f"fallback_used={str(contract.get('fallback_used')).lower()}"
    )


def _data_used(fallback: dict[str, Any], state: dict[str, Any]) -> list[str]:
    items = []
    if fallback.get("data_source"):
        items.append(fallback["data_source"])
    weight_type = state.get("active_weights", {}).get("type")
    if weight_type and weight_type != "missing":
        items.append(weight_type)
    return sorted(set(items))
