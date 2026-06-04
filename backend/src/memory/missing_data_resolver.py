from __future__ import annotations

from typing import Any

from src.memory.fallback_computation import (
    DEFAULT_SECTOR_MAP,
    compute_allocation_change,
    compute_hhi_bundle,
    compute_risk_contribution,
    compute_sector_weights,
)
from src.memory.plot_validation import validate_plot_payload


class MissingDataResolver:
    """Context recovery plus deterministic fallback computation."""

    def resolve(self, resolved: dict[str, Any]) -> dict[str, Any]:
        state = dict(resolved.get("session_state") or {})
        context = dict(resolved.get("resolved_context") or {})
        plot_request = context.get("plot_request") or _plot_request_from_state(state)
        if not plot_request:
            resolved["fallback_result"] = {"fallback_used": False, "status": "not_applicable"}
            return resolved

        plot_id = plot_request.get("requested_plot_id")
        result = {"fallback_used": False, "status": "not_applicable"}

        if plot_id in {"plot_42_ticker_concentration_plot", "plot_43_sector_concentration_plot"}:
            result = self._resolve_concentration(plot_id, state, plot_request)
        elif plot_id in {"plot_39_hhi_concentration_index", "plot_40_effective_number_of_holdings"}:
            result = self._resolve_hhi(state, plot_request)
        elif plot_id == "plot_55_risk_contribution_by_ticker":
            result = self._resolve_risk_contribution(state, plot_request)
        elif plot_id in {"plot_29_current_vs_advisory_allocation_by_ticker", "plot_32_allocation_change_by_ticker"}:
            result = self._resolve_current_vs_advisory(plot_id, state, plot_request)
        elif plot_id == "plot_61_eigenvector_centrality_by_ticker":
            result = {
                "status": "unavailable",
                "reason": "institutional graph data is required; no placeholder centrality values were generated",
                "missing_inputs": ["institutional_graph_data"],
                "fallback_used": False,
            }
        elif plot_id == "plot_81_strategy_performance_comparison":
            result = {
                "status": "requires_confirmation",
                "reason": "optimized strategy comparison requires explicit optimizer permission",
                "missing_inputs": ["optimizer_permission"],
                "fallback_used": False,
            }

        state = self._apply_result_to_state(state, result, plot_request)
        resolved["session_state"] = state
        resolved["resolved_context"] = {**context, "plot_request": plot_request}
        resolved["fallback_result"] = result
        resolved["validation_result"] = result.get("validation_result", resolved.get("validation_result"))
        return resolved

    def _resolve_concentration(self, plot_id: str, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = state.get("active_weights", {})
        if not weights.get("weights"):
            return _missing("current_weights", "Current weights are required for concentration plots.")
        if weights.get("type") == "equal_weight_proxy" and not weights.get("approved_by_user"):
            return _missing("current_weights", "Equal-weight proxy requires user approval.")

        ticker_weights = weights["weights"]
        if plot_id == "plot_43_sector_concentration_plot":
            sector_weights = compute_sector_weights(ticker_weights, DEFAULT_SECTOR_MAP)
            data = [
                {"sector": sector, "allocation_percent": weight * 100.0}
                for sector, weight in sorted(sector_weights.items(), key=lambda item: item[1], reverse=True)
            ]
            payload = _payload(request, state, data, "sector_weights_from_ticker_weights")
        else:
            data = [
                {"ticker": ticker, "allocation_percent": weight}
                for ticker, weight in sorted(ticker_weights.items(), key=lambda item: item[1], reverse=True)
            ]
            payload = _payload(request, state, data, weights.get("type"))

        return _validated_success(payload, request, weights.get("type") == "equal_weight_proxy")

    def _resolve_hhi(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = state.get("active_weights", {})
        if not weights.get("weights"):
            return _missing("current_weights", "Current weights are required for HHI metrics.")
        metrics = compute_hhi_bundle(weights["weights"], DEFAULT_SECTOR_MAP)
        payload = _payload(request, state, [{"portfolio": "Current", **metrics}], weights.get("type"))
        payload["metrics"] = metrics
        return _validated_success(payload, request, weights.get("type") == "equal_weight_proxy")

    def _resolve_risk_contribution(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = state.get("active_weights", {})
        tickers = state.get("active_tickers", [])
        if not weights.get("weights"):
            return _missing("current_weights", "Current weights are required for risk contribution.")
        date_range = state.get("active_date_range") or {}
        computed = compute_risk_contribution(
            tickers=tickers,
            weights=weights["weights"],
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        if computed.get("status") != "success":
            return computed
        payload = _payload(request, state, computed["data"], weights.get("type"))
        payload.update(
            {
                "formula": computed["formula"],
                "confidence": computed["confidence"],
                "limitations": computed["limitations"],
                "fallback_used": computed["fallback_used"] or weights.get("type") == "equal_weight_proxy",
                "covariance_matrix": computed["covariance_matrix"],
            }
        )
        return _validated_success(payload, request, weights.get("type") == "equal_weight_proxy")

    def _resolve_current_vs_advisory(self, plot_id: str, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        current = state.get("active_weights", {})
        advisory = state.get("active_advisory_weights", {})
        missing = []
        if not current.get("weights"):
            missing.append("current_weights")
        if not advisory.get("available") or not advisory.get("weights"):
            missing.append("advisory_weights")
        if missing:
            return {
                "status": "missing_data",
                "reason": "Current and advisory weights are both required; no grouped/changed allocation plot was rendered.",
                "missing_inputs": missing,
                "fallback_used": False,
            }

        changes = compute_allocation_change(current["weights"], advisory["weights"])
        if plot_id == "plot_32_allocation_change_by_ticker":
            data = [{"ticker": ticker, "allocation_change_percent": change * 100.0} for ticker, change in changes.items()]
        else:
            data = [
                {
                    "ticker": ticker,
                    "current_allocation_percent": current["weights"].get(ticker, 0.0),
                    "advisory_allocation_percent": advisory["weights"].get(ticker, 0.0),
                }
                for ticker in sorted(set(current["weights"]) | set(advisory["weights"]))
            ]
        payload = _payload(request, state, data, "current_and_advisory_weights")
        return _validated_success(payload, request, current.get("type") == "equal_weight_proxy")

    def _apply_result_to_state(self, state: dict[str, Any], result: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        next_state = dict(state)
        if result.get("status") == "success":
            payload = result.get("plot_payload", {})
            next_state["last_plot_status"] = "success"
            next_state["last_successful_plot"] = payload
            next_state["last_successful_analysis"] = {
                "plot_id": payload.get("plot_id"),
                "data_source": payload.get("data_source"),
                "fallback_used": payload.get("fallback_used", False),
            }
            next_state["missing_inputs"] = []
        elif result.get("missing_inputs"):
            next_state["missing_inputs"] = result.get("missing_inputs", [])
            next_state["last_plot_status"] = "missing_data"
        next_state["last_plot_id"] = request.get("requested_plot_id")
        next_state["last_chart_type"] = request.get("requested_chart_type")
        next_state["last_bar_mode"] = request.get("requested_bar_mode")
        return next_state


def _plot_request_from_state(state: dict[str, Any]) -> dict[str, Any] | None:
    if not state.get("last_plot_id"):
        return None
    return {
        "requested_plot_id": state.get("last_plot_id"),
        "requested_chart_type": state.get("last_chart_type"),
        "requested_bar_mode": state.get("last_bar_mode"),
        "requested_universe": state.get("active_universe"),
        "requested_tickers": state.get("active_tickers", []),
        "required_fields": [],
        "data_source": state.get("active_weights", {}).get("type"),
        "status": state.get("last_plot_status"),
    }


def _payload(request: dict[str, Any], state: dict[str, Any], data: list[dict[str, Any]], data_source: str | None) -> dict[str, Any]:
    metadata = _metadata_for_plot(request.get("requested_plot_id"))
    tickers = state.get("active_tickers", [])
    weights = state.get("active_weights", {}) if isinstance(state.get("active_weights"), dict) else {}
    return {
        "plot_id": request.get("requested_plot_id"),
        "plot_short_id": metadata.get("plot_short_id"),
        "chart_type": request.get("requested_chart_type"),
        "bar_mode": request.get("requested_bar_mode"),
        "universe": state.get("active_universe"),
        "ticker_count": len(tickers),
        "tickers": tickers,
        "tickers_used": tickers,
        "x_axis": metadata["x_axis"],
        "y_axis": metadata["y_axis"],
        "unit": metadata["unit"],
        "data": data,
        "data_source": data_source or "computed",
        "proxy_declared": data_source == "equal_weight_proxy",
        "proxy_used": data_source == "equal_weight_proxy",
        "fallback_used": data_source == "equal_weight_proxy",
        "approved_by_user": bool(weights.get("approved_by_user")) if data_source == "equal_weight_proxy" else False,
        "optimizer_called": False,
        "advisory_allocation_generated": False,
        "confidence": "Medium" if data_source == "equal_weight_proxy" else "High",
        "limitations": ["Actual current holdings were unavailable."] if data_source == "equal_weight_proxy" else [],
    }


def _validated_success(payload: dict[str, Any], request: dict[str, Any], proxy_used: bool) -> dict[str, Any]:
    payload["proxy_used"] = bool(proxy_used)
    payload["proxy_declared"] = bool(proxy_used)
    validation = validate_plot_payload(request, payload)
    return {
        "status": "success" if validation["can_render"] else "failed_validation",
        "plot_payload": payload,
        "validation_result": validation,
        "fallback_used": bool(payload.get("fallback_used")),
        "data_source": payload.get("data_source"),
        "confidence": payload.get("confidence", "Medium" if proxy_used else "High"),
        "limitations": payload.get("limitations", []),
    }


def _missing(field: str, reason: str) -> dict[str, Any]:
    return {
        "status": "missing_data",
        "reason": reason,
        "missing_inputs": [field],
        "fallback_used": False,
    }


def _metadata_for_plot(plot_id: str | None) -> dict[str, str]:
    mapping = {
        "plot_42_ticker_concentration_plot": {
            "plot_short_id": "ticker_concentration_plot",
            "x_axis": "allocation_percent",
            "y_axis": "ticker",
            "unit": "%",
        },
        "plot_43_sector_concentration_plot": {
            "plot_short_id": "sector_concentration_plot",
            "x_axis": "allocation_percent",
            "y_axis": "sector",
            "unit": "%",
        },
        "plot_39_hhi_concentration_index": {
            "plot_short_id": "hhi_concentration_index",
            "x_axis": "portfolio",
            "y_axis": "hhi_value",
            "unit": "ratio",
        },
        "plot_40_effective_number_of_holdings": {
            "plot_short_id": "effective_number_of_holdings",
            "x_axis": "portfolio",
            "y_axis": "effective_count",
            "unit": "count",
        },
        "plot_55_risk_contribution_by_ticker": {
            "plot_short_id": "risk_contribution_by_ticker",
            "x_axis": "risk_contribution_percent",
            "y_axis": "ticker",
            "unit": "%",
        },
        "plot_29_current_vs_advisory_allocation_by_ticker": {
            "plot_short_id": "current_vs_advisory_allocation_by_ticker",
            "x_axis": "ticker",
            "y_axis": "allocation_percent",
            "unit": "%",
        },
        "plot_32_allocation_change_by_ticker": {
            "plot_short_id": "allocation_change_by_ticker",
            "x_axis": "ticker",
            "y_axis": "allocation_change_percent",
            "unit": "%",
        },
        "plot_61_eigenvector_centrality_by_ticker": {
            "plot_short_id": "eigenvector_centrality_by_ticker",
            "x_axis": "centrality_score",
            "y_axis": "ticker",
            "unit": "decimal",
        },
    }
    return mapping.get(
        str(plot_id or ""),
        {"plot_short_id": str(plot_id or ""), "x_axis": "metric", "y_axis": "value", "unit": "none"},
    )
