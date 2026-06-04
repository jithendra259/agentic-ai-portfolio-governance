from __future__ import annotations

from typing import Any

from src.memory.session_state import KNOWN_UNIVERSE_TICKERS


def validate_plot_payload(request: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if not request:
        return {"can_render": True, "errors": [], "warnings": []}

    if request.get("requested_plot_id") and payload.get("plot_id") != request.get("requested_plot_id"):
        errors.append("requested_plot_id does not match returned plot_id")
    if request.get("requested_chart_type") and payload.get("chart_type") != request.get("requested_chart_type"):
        errors.append("requested_chart_type does not match returned chart_type")
    if request.get("requested_bar_mode") and payload.get("bar_mode") != request.get("requested_bar_mode"):
        errors.append("requested_bar_mode does not match returned bar_mode")

    requested_universe = request.get("requested_universe")
    returned_universe = payload.get("universe")
    if requested_universe and returned_universe and requested_universe != returned_universe:
        errors.append("requested_universe does not match returned_universe")

    returned_tickers = payload.get("tickers")
    requested_tickers = request.get("requested_tickers") or []
    if returned_tickers and requested_tickers and set(returned_tickers) != set(requested_tickers):
        errors.append("returned tickers differ from requested tickers")

    requested_universe = request.get("requested_universe")
    if requested_universe == "U1" and not request.get("requested_subset_explicit"):
        expected = set(KNOWN_UNIVERSE_TICKERS["U1"])
        actual = set(returned_tickers or [])
        if actual != expected:
            errors.append("U1 plot must use all 20 U1 tickers unless an explicit subset was requested")

    if _contains_placeholder_x(payload):
        errors.append("placeholder ticker/value X is not allowed")

    for field in request.get("required_fields") or []:
        if not _field_exists(payload.get("data"), field):
            errors.append(f"required field missing: {field}")

    if not payload.get("data_source"):
        errors.append("data source is missing")
    if payload.get("proxy_used") and not payload.get("proxy_declared"):
        errors.append("proxy/default use is not declared")
    if payload.get("fallback_used") and not payload.get("data_source"):
        errors.append("fallback/proxy data source is not declared")
    if payload.get("status") in {"missing_data", "unavailable", "failed", "failed_validation"}:
        errors.append(payload.get("reason") or "plot data unavailable")

    return {
        "can_render": not errors,
        "render": not errors,
        "status": "pass" if not errors else "blocked",
        "reason": "; ".join(errors) if errors else None,
        "errors": errors,
        "warnings": warnings,
        "missing_fields": [item.removeprefix("required field missing: ") for item in errors if item.startswith("required field missing: ")],
        "next_options": [] if not errors else ["fix_payload", "recover_missing_data", "ask_user"],
    }


def _field_exists(data: Any, field: str) -> bool:
    if isinstance(data, list):
        return all(isinstance(row, dict) and field in row for row in data)
    if isinstance(data, dict):
        return field in data
    return False


def _contains_placeholder_x(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            (key.lower() in {"ticker", "tickerx", "tickery", "symbol"} and str(item).upper() == "X")
            or _contains_placeholder_x(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_placeholder_x(item) for item in value)
    return False
