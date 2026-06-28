from __future__ import annotations

from typing import Any

from src.memory.fallback_computation import (
    DEFAULT_SECTOR_MAP,
    compute_allocation_change,
    compute_allocation_vs_risk_contribution_scatter,
    compute_bubble_risk_return_scatter,
    compute_cvar_return_scatter,
    compute_drawdown_return_scatter,
    compute_drawdown_over_time,
    compute_hhi_bundle,
    compute_historical_adjusted_close,
    compute_instability_index_over_time,
    compute_normalized_price_comparison,
    compute_pairwise_return_regression_scatter,
    compute_portfolio_value_over_time,
    compute_portfolio_return_waterfall,
    compute_return_distribution_histogram,
    compute_return_range_by_ticker,
    compute_risk_return_scatter,
    compute_risk_contribution,
    compute_risk_contribution_waterfall,
    compute_rolling_average_correlation,
    compute_rolling_var_cvar,
    compute_rolling_volatility_over_time,
    compute_portfolio_health_donut,
    compute_risk_contribution_donut,
    compute_sector_allocation_donut,
    compute_sector_ticker_nested_donut,
    compute_sector_weights,
    compute_semi_donut_risk_gauge,
    compute_ticker_allocation_donut,
    compute_volatility_range_by_ticker,
)
from src.memory.chart_registry import get_chart_definition, premium_charts_enabled
from src.memory.planner import GovernanceTaskPlanner
from src.memory.plot_validation import validate_plot_payload


PIE_CHART_TYPES = {"pie", "donut", "center_label_donut", "nested_donut", "semi_donut"}
SCATTER_CHART_TYPES = {"scatter", "bubble_scatter", "scatter_regression", "webgl_scatter"}


class MissingDataResolver:
    """Context recovery plus deterministic fallback computation."""

    def __init__(self) -> None:
        self.task_planner = GovernanceTaskPlanner()

    def resolve(self, resolved: dict[str, Any]) -> dict[str, Any]:
        state = dict(resolved.get("session_state") or {})
        context = dict(resolved.get("resolved_context") or {})
        if resolved.get("direct_response"):
            resolved["fallback_result"] = {"fallback_used": False, "status": "not_applicable"}
            return resolved
        plot_request = context.get("plot_request") or _plot_request_from_state(state)
        if not plot_request:
            resolved["fallback_result"] = {"fallback_used": False, "status": "not_applicable"}
            return resolved
        if state.get("current_plan"):
            plot_request["planner"] = state.get("current_plan")

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
        elif plot_id == "return_range_by_ticker":
            result = self._resolve_return_range(state, plot_request)
        elif plot_id == "volatility_range_by_ticker":
            result = self._resolve_volatility_range(state, plot_request)
        elif plot_id == "portfolio_return_waterfall":
            result = self._resolve_portfolio_return_waterfall(state, plot_request)
        elif plot_id == "risk_contribution_waterfall":
            result = self._resolve_risk_contribution_waterfall(state, plot_request)
        elif plot_id == "current_vs_advisory_mirrored_bar":
            result = self._resolve_current_vs_advisory_mirrored(state, plot_request)
        elif plot_id == "return_distribution_histogram":
            result = self._resolve_return_distribution_histogram(state, plot_request)
        elif plot_id == "historical_adjusted_close":
            result = self._resolve_historical_adjusted_close(state, plot_request)
        elif plot_id == "normalized_price_comparison":
            result = self._resolve_normalized_price_comparison(state, plot_request)
        elif plot_id == "portfolio_value_over_time":
            result = self._resolve_portfolio_value_over_time(state, plot_request)
        elif plot_id == "drawdown_over_time":
            result = self._resolve_drawdown_over_time(state, plot_request)
        elif plot_id == "rolling_volatility_over_time":
            result = self._resolve_rolling_volatility_over_time(state, plot_request)
        elif plot_id == "rolling_correlation_over_time":
            result = self._resolve_rolling_correlation_over_time(state, plot_request)
        elif plot_id == "instability_index_over_time":
            result = self._resolve_instability_index_over_time(state, plot_request)
        elif plot_id == "cvar_over_time":
            result = self._resolve_cvar_over_time(state, plot_request)
        elif plot_id == "sector_allocation_donut":
            result = self._resolve_sector_allocation_donut(state, plot_request)
        elif plot_id == "ticker_allocation_donut":
            result = self._resolve_ticker_allocation_donut(state, plot_request)
        elif plot_id == "risk_contribution_donut":
            result = self._resolve_risk_contribution_donut(state, plot_request)
        elif plot_id == "sector_ticker_nested_donut":
            result = self._resolve_sector_ticker_nested_donut(state, plot_request)
        elif plot_id == "portfolio_health_donut":
            result = self._resolve_portfolio_health_donut(state, plot_request)
        elif plot_id == "semi_donut_risk_gauge":
            result = self._resolve_semi_donut_risk_gauge(state, plot_request)
        elif plot_id == "risk_return_scatter":
            result = self._resolve_risk_return_scatter(state, plot_request)
        elif plot_id == "cvar_return_scatter":
            result = self._resolve_cvar_return_scatter(state, plot_request)
        elif plot_id == "drawdown_return_scatter":
            result = self._resolve_drawdown_return_scatter(state, plot_request)
        elif plot_id == "allocation_vs_risk_contribution_scatter":
            result = self._resolve_allocation_vs_risk_contribution_scatter(state, plot_request)
        elif plot_id == "bubble_risk_return":
            result = self._resolve_bubble_risk_return(state, plot_request)
        elif plot_id == "scatter_with_regression_line":
            result = self._resolve_scatter_with_regression_line(state, plot_request)
        elif plot_id == "ownership_overlap_correlation_scatter":
            result = {
                "status": "unavailable",
                "reason": "institutional ownership graph data is required; ownership-overlap values were not invented",
                "missing_inputs": ["institutional_graph_data"],
                "fallback_used": False,
            }
        elif plot_id == "beta_return_scatter":
            result = {
                "status": "unavailable",
                "reason": "benchmark return series is required for beta-return scatter",
                "missing_inputs": ["benchmark_series"],
                "fallback_used": False,
            }
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
        state = self.task_planner.update_after_fallback(state, result, plot_request)
        if isinstance(result.get("plot_payload"), dict) and state.get("current_plan"):
            result["plot_payload"]["planner"] = state["current_plan"]
        resolved["session_state"] = state
        resolved["resolved_context"] = {**context, "plot_request": plot_request}
        resolved["fallback_result"] = result
        resolved["validation_result"] = result.get("validation_result", resolved.get("validation_result"))
        return resolved

    def _resolve_concentration(self, plot_id: str, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = state.get("active_weights", {})
        if not weights.get("weights"):
            return _missing("current_weights", "Current weights are required for concentration plots.")

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
        data_row = {
            "portfolio": "Current",
            "hhi_value": metrics.get("ticker_hhi"),
            "effective_count": metrics.get("ticker_effective_holdings"),
            **metrics,
        }
        payload = _payload(request, state, [data_row], weights.get("type"))
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

    def _resolve_return_range(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        date_range = state.get("active_date_range") or {}
        computed = compute_return_range_by_ticker(
            tickers=state.get("active_tickers", []),
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        if computed.get("status") != "success":
            return computed
        payload = _payload(request, state, computed["data"], computed.get("data_source"))
        payload.update(
            {
                "fallback_used": bool(computed.get("fallback_used")),
                "confidence": computed.get("confidence", "High"),
                "limitations": computed.get("limitations", []),
            }
        )
        return _validated_success(payload, request, False)

    def _resolve_volatility_range(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        date_range = state.get("active_date_range") or {}
        computed = compute_volatility_range_by_ticker(
            tickers=state.get("active_tickers", []),
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        if computed.get("status") != "success":
            return computed
        payload = _payload(request, state, computed["data"], computed.get("data_source"))
        payload.update(
            {
                "fallback_used": bool(computed.get("fallback_used")),
                "confidence": computed.get("confidence", "High"),
                "limitations": computed.get("limitations", []),
            }
        )
        return _validated_success(payload, request, False)

    def _resolve_return_distribution_histogram(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = state.get("active_weights", {})
        if not weights.get("weights"):
            return _missing("current_weights", "Current weights are required for a portfolio return histogram.")
        date_range = state.get("active_date_range") or {}
        computed = compute_return_distribution_histogram(
            tickers=state.get("active_tickers", []),
            weights=weights["weights"],
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        if computed.get("status") != "success":
            return computed
        payload = _payload(request, state, computed["data"], computed.get("data_source"))
        payload.update(
            {
                "fallback_used": bool(computed.get("fallback_used") or weights.get("type") == "equal_weight_proxy"),
                "confidence": computed.get("confidence", "High"),
                "limitations": computed.get("limitations", []),
            }
        )
        return _validated_success(payload, request, weights.get("type") == "equal_weight_proxy")

    def _resolve_portfolio_return_waterfall(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = state.get("active_weights", {})
        if not weights.get("weights"):
            return _missing("current_weights", "Current weights are required for portfolio return contribution waterfall.")
        date_range = state.get("active_date_range") or {}
        computed = compute_portfolio_return_waterfall(
            tickers=state.get("active_tickers", []),
            weights=weights["weights"],
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        if computed.get("status") != "success":
            return computed
        payload = _payload(request, state, computed["data"], computed.get("data_source"))
        payload.update(
            {
                "fallback_used": bool(computed.get("fallback_used") or weights.get("type") == "equal_weight_proxy"),
                "confidence": computed.get("confidence", "High"),
                "limitations": computed.get("limitations", []),
            }
        )
        return _validated_success(payload, request, weights.get("type") == "equal_weight_proxy")

    def _resolve_risk_contribution_waterfall(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = state.get("active_weights", {})
        if not weights.get("weights"):
            return _missing("current_weights", "Current weights are required for risk contribution waterfall.")
        date_range = state.get("active_date_range") or {}
        computed = compute_risk_contribution_waterfall(
            tickers=state.get("active_tickers", []),
            weights=weights["weights"],
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        if computed.get("status") != "success":
            return computed
        payload = _payload(request, state, computed["data"], computed.get("data_source"))
        payload.update(
            {
                "fallback_used": bool(computed.get("fallback_used") or weights.get("type") == "equal_weight_proxy"),
                "confidence": computed.get("confidence", "High"),
                "limitations": computed.get("limitations", []),
            }
        )
        return _validated_success(payload, request, weights.get("type") == "equal_weight_proxy")

    def _resolve_current_vs_advisory_mirrored(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
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
                "reason": "Current and advisory weights are required for mirrored exposure comparison.",
                "missing_inputs": missing,
                "fallback_used": False,
            }
        current_weights = current["weights"]
        advisory_weights = advisory["weights"]
        data = [
            {
                "ticker": ticker,
                "current_weight": float(current_weights.get(ticker, 0.0)),
                "advisory_weight": float(advisory_weights.get(ticker, 0.0)),
            }
            for ticker in sorted(set(current_weights) | set(advisory_weights))
        ]
        payload = _payload(request, state, data, "current_and_advisory_weights")
        return _validated_success(payload, request, current.get("type") == "equal_weight_proxy")

    def _resolve_historical_adjusted_close(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        date_range = state.get("active_date_range") or {}
        computed = compute_historical_adjusted_close(
            tickers=state.get("active_tickers", []),
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        return _line_success_from_computed(request, state, computed)

    def _resolve_normalized_price_comparison(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        date_range = state.get("active_date_range") or {}
        computed = compute_normalized_price_comparison(
            tickers=state.get("active_tickers", []),
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        return _line_success_from_computed(request, state, computed)

    def _resolve_portfolio_value_over_time(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = state.get("active_weights", {})
        if not weights.get("weights"):
            return _missing("current_weights", "Current weights are required for portfolio value over time.")
        date_range = state.get("active_date_range") or {}
        computed = compute_portfolio_value_over_time(
            tickers=state.get("active_tickers", []),
            weights=weights["weights"],
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
            initial_capital=state.get("active_initial_capital"),
        )
        return _line_success_from_computed(request, state, computed, proxy_used=weights.get("type") == "equal_weight_proxy")

    def _resolve_drawdown_over_time(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = state.get("active_weights", {})
        if not weights.get("weights"):
            return _missing("current_weights", "Current weights are required for drawdown over time.")
        date_range = state.get("active_date_range") or {}
        computed = compute_drawdown_over_time(
            tickers=state.get("active_tickers", []),
            weights=weights["weights"],
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        return _line_success_from_computed(request, state, computed, proxy_used=weights.get("type") == "equal_weight_proxy")

    def _resolve_rolling_volatility_over_time(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        date_range = state.get("active_date_range") or {}
        weights = state.get("active_weights", {})
        computed = compute_rolling_volatility_over_time(
            tickers=state.get("active_tickers", []),
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
            weights=weights.get("weights", {}),
        )
        return _line_success_from_computed(request, state, computed, proxy_used=weights.get("type") == "equal_weight_proxy")

    def _resolve_rolling_correlation_over_time(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        date_range = state.get("active_date_range") or {}
        computed = compute_rolling_average_correlation(
            tickers=state.get("active_tickers", []),
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        return _line_success_from_computed(request, state, computed)

    def _resolve_instability_index_over_time(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        date_range = state.get("active_date_range") or {}
        computed = compute_instability_index_over_time(
            tickers=state.get("active_tickers", []),
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        return _line_success_from_computed(request, state, computed)

    def _resolve_cvar_over_time(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = state.get("active_weights", {})
        if not weights.get("weights"):
            return _missing("current_weights", "Current weights are required for CVaR over time.")
        date_range = state.get("active_date_range") or {}
        computed = compute_rolling_var_cvar(
            tickers=state.get("active_tickers", []),
            weights=weights["weights"],
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        return _line_success_from_computed(request, state, computed, proxy_used=weights.get("type") == "equal_weight_proxy")

    def _resolve_sector_allocation_donut(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = _resolved_current_weights(state, "sector allocation donut")
        if weights.get("status") != "success":
            return weights
        computed = compute_sector_allocation_donut(weights["weights"])
        return _pie_success_from_computed(request, state, computed, proxy_used=weights.get("type") == "equal_weight_proxy")

    def _resolve_ticker_allocation_donut(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = _resolved_current_weights(state, "ticker allocation donut")
        if weights.get("status") != "success":
            return weights
        definition = get_chart_definition(request.get("requested_plot_id")) or {}
        computed = compute_ticker_allocation_donut(weights["weights"], max_slices=int(definition.get("max_slices", 8)))
        if computed.get("status") == "success" and computed.get("slice_count", 0) > int(definition.get("max_slices", 8)):
            payload = _payload(request, state, computed.get("data", []), computed.get("data_source") or weights.get("type"))
            payload.update(
                {
                    "plot_type": "bar",
                    "chart_type": "bar",
                    "component": "BarChart",
                    "bar_mode": "horizontal",
                    "x_axis": "weight_percent",
                    "y_axis": "ticker",
                    "unit": "%",
                    "series": [{"key": "weight_percent", "label": "Exposure weight (%)"}],
                    "fallback_rendered": True,
                    "fallback_from_chart_type": "donut",
                    "fallback_chart": "ticker_concentration_bar",
                    "fallback_method": "ticker_donut_slice_count_exceeded",
                    "slice_count": computed.get("slice_count"),
                    "max_slices": definition.get("max_slices", 8),
                    "warnings": [
                        f"Ticker allocation donut was converted to a horizontal bar because {computed.get('slice_count')} slices exceeds the {definition.get('max_slices', 8)}-slice readability limit."
                    ],
                    "limitations": computed.get("limitations", []),
                    "confidence": computed.get("confidence", "High"),
                }
            )
            return _validated_success(payload, request, weights.get("type") == "equal_weight_proxy")
        return _pie_success_from_computed(request, state, computed, proxy_used=weights.get("type") == "equal_weight_proxy")

    def _resolve_risk_contribution_donut(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = _resolved_current_weights(state, "risk contribution donut")
        if weights.get("status") != "success":
            return weights
        date_range = state.get("active_date_range") or {}
        computed = compute_risk_contribution_donut(
            tickers=state.get("active_tickers", []),
            weights=weights["weights"],
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        return _pie_success_from_computed(request, state, computed, proxy_used=weights.get("type") == "equal_weight_proxy")

    def _resolve_sector_ticker_nested_donut(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = _resolved_current_weights(state, "sector-ticker nested donut")
        if weights.get("status") != "success":
            return weights
        computed = compute_sector_ticker_nested_donut(weights["weights"])
        return _pie_success_from_computed(request, state, computed, proxy_used=weights.get("type") == "equal_weight_proxy")

    def _resolve_portfolio_health_donut(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = _resolved_current_weights(state, "portfolio health donut")
        if weights.get("status") != "success":
            return weights
        computed = compute_portfolio_health_donut(weights["weights"])
        return _pie_success_from_computed(request, state, computed, proxy_used=weights.get("type") == "equal_weight_proxy")

    def _resolve_semi_donut_risk_gauge(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = _resolved_current_weights(state, "semi-donut risk gauge")
        if weights.get("status") != "success":
            return weights
        computed = compute_semi_donut_risk_gauge(weights["weights"])
        return _pie_success_from_computed(request, state, computed, proxy_used=weights.get("type") == "equal_weight_proxy")

    def _resolve_risk_return_scatter(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        date_range = state.get("active_date_range") or {}
        weights = state.get("active_weights", {}) if isinstance(state.get("active_weights"), dict) else {}
        computed = compute_risk_return_scatter(
            tickers=state.get("active_tickers", []),
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
            weights=weights.get("weights") if isinstance(weights.get("weights"), dict) else None,
        )
        return _scatter_success_from_computed(request, state, computed, proxy_used=False)

    def _resolve_cvar_return_scatter(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        date_range = state.get("active_date_range") or {}
        computed = compute_cvar_return_scatter(
            tickers=state.get("active_tickers", []),
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        return _scatter_success_from_computed(request, state, computed, proxy_used=False)

    def _resolve_drawdown_return_scatter(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        date_range = state.get("active_date_range") or {}
        computed = compute_drawdown_return_scatter(
            tickers=state.get("active_tickers", []),
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        return _scatter_success_from_computed(request, state, computed, proxy_used=False)

    def _resolve_allocation_vs_risk_contribution_scatter(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = _resolved_current_weights(state, "allocation vs risk contribution scatter")
        if weights.get("status") != "success":
            return weights
        date_range = state.get("active_date_range") or {}
        computed = compute_allocation_vs_risk_contribution_scatter(
            tickers=state.get("active_tickers", []),
            weights=weights["weights"],
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        return _scatter_success_from_computed(request, state, computed, proxy_used=weights.get("type") == "equal_weight_proxy")

    def _resolve_bubble_risk_return(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        weights = _resolved_current_weights(state, "risk-return bubble scatter")
        if weights.get("status") != "success":
            return weights
        date_range = state.get("active_date_range") or {}
        computed = compute_bubble_risk_return_scatter(
            tickers=state.get("active_tickers", []),
            weights=weights["weights"],
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        return _scatter_success_from_computed(request, state, computed, proxy_used=weights.get("type") == "equal_weight_proxy")

    def _resolve_scatter_with_regression_line(self, state: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        date_range = state.get("active_date_range") or {}
        computed = compute_pairwise_return_regression_scatter(
            tickers=state.get("active_tickers", []),
            start_date=date_range.get("start"),
            end_date=date_range.get("end"),
        )
        return _scatter_success_from_computed(request, state, computed, proxy_used=False)

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
    chart_definition = get_chart_definition(request.get("requested_plot_id")) or {}
    tickers = state.get("active_tickers", [])
    weights = state.get("active_weights", {}) if isinstance(state.get("active_weights"), dict) else {}
    chart_type = request.get("requested_chart_type") or chart_definition.get("chart_type", "bar")
    if chart_type in {"line", "line_area", "stacked_area", "dual_axis_line"}:
        plot_type = "line"
    elif chart_type in PIE_CHART_TYPES:
        plot_type = "pie"
    elif chart_type in SCATTER_CHART_TYPES:
        plot_type = "scatter"
    else:
        plot_type = "bar"
    return {
        "plot_id": request.get("requested_plot_id"),
        "plot_short_id": metadata.get("plot_short_id"),
        "plot_type": plot_type,
        "chart_type": chart_type,
        "chart_tier": chart_definition.get("chart_tier", "free"),
        "component": chart_definition.get(
            "component",
            "LineChart" if plot_type == "line" else "PieChart" if plot_type == "pie" else "ScatterChart" if plot_type == "scatter" else "BarChart",
        ),
        "requires_premium": bool(chart_definition.get("requires_premium", False)),
        "premium_enabled": premium_charts_enabled(),
        "fallback_chart": chart_definition.get("fallback_chart"),
        "required_fields": list(chart_definition.get("required_fields", [])),
        "optional_fields": list(chart_definition.get("optional_fields", [])),
        "range_start_field": chart_definition.get("range_start_field"),
        "range_end_field": chart_definition.get("range_end_field"),
        "bar_mode": request.get("requested_bar_mode"),
        "title": metadata.get("title"),
        "universe": state.get("active_universe"),
        "ticker_count": len(tickers),
        "slice_count": len(data) if plot_type == "pie" else None,
        "point_count": len(data) if plot_type == "scatter" else None,
        "category_field": chart_definition.get("category_field"),
        "value_field": chart_definition.get("value_field"),
        "point_id": chart_definition.get("point_id"),
        "color_axis": chart_definition.get("color_axis"),
        "size_axis": chart_definition.get("size_axis"),
        "max_slices": chart_definition.get("max_slices"),
        "max_inner_slices": chart_definition.get("max_inner_slices"),
        "max_outer_slices": chart_definition.get("max_outer_slices"),
        "tickers": tickers,
        "tickers_used": tickers,
        "x_axis": metadata["x_axis"],
        "y_axis": metadata["y_axis"],
        "x_unit": chart_definition.get("x_unit") or metadata["unit"],
        "y_unit": chart_definition.get("y_unit") or metadata["unit"],
        "unit": metadata["unit"],
        "date_range": state.get("active_date_range") or {},
        "connect_nulls": False,
        "curve": "linear",
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


def _resolved_current_weights(state: dict[str, Any], chart_label: str) -> dict[str, Any]:
    weights = state.get("active_weights", {}) if isinstance(state.get("active_weights"), dict) else {}
    if not weights.get("weights"):
        return _missing("current_weights", f"Current weights are required for {chart_label}.")
    return {"status": "success", "weights": weights["weights"], "type": weights.get("type")}


def _line_success_from_computed(
    request: dict[str, Any],
    state: dict[str, Any],
    computed: dict[str, Any],
    proxy_used: bool = False,
) -> dict[str, Any]:
    if computed.get("status") != "success":
        return computed
    payload = _payload(request, state, computed.get("data", []), computed.get("data_source"))
    tickers = computed.get("tickers") or payload.get("tickers_used") or []
    payload.update(
        {
            "series": computed.get("series", []),
            "tickers": tickers,
            "tickers_used": tickers,
            "ticker_or_universe": state.get("active_universe") or (tickers[0] if len(tickers) == 1 else None),
            "ticker_count": len(tickers),
            "date_range": computed.get("date_range") or payload.get("date_range"),
            "data_source": computed.get("data_source") or payload.get("data_source"),
            "fallback_used": bool(computed.get("fallback_used") or proxy_used),
            "fallback_method": computed.get("fallback_method"),
            "formula": computed.get("formula"),
            "confidence": computed.get("confidence", payload.get("confidence", "High")),
            "limitations": computed.get("limitations", []),
            "connect_nulls": False,
            "curve": "linear",
            "optimizer_called": False,
            "advisory_allocation_generated": False,
        }
    )
    if computed.get("initial_capital") is not None:
        payload["initial_capital"] = computed["initial_capital"]
    return _validated_success(payload, request, proxy_used)


def _pie_success_from_computed(
    request: dict[str, Any],
    state: dict[str, Any],
    computed: dict[str, Any],
    proxy_used: bool = False,
) -> dict[str, Any]:
    if computed.get("status") != "success":
        return computed
    payload = _payload(request, state, computed.get("data", []), computed.get("data_source"))
    payload.update(
        {
            "series": computed.get("series", []),
            "slice_count": computed.get("slice_count", len(computed.get("data", []))),
            "total_value": computed.get("total_value"),
            "center_label": computed.get("center_label"),
            "metrics": computed.get("metrics", {}),
            "data_source": computed.get("data_source") or payload.get("data_source"),
            "fallback_used": bool(computed.get("fallback_used") or proxy_used),
            "fallback_method": computed.get("fallback_method"),
            "formula": computed.get("formula"),
            "confidence": computed.get("confidence", payload.get("confidence", "High")),
            "limitations": computed.get("limitations", []),
            "optimizer_called": False,
            "advisory_allocation_generated": False,
        }
    )
    if computed.get("covariance_matrix") is not None:
        payload["covariance_matrix"] = computed["covariance_matrix"]
    return _validated_success(payload, request, proxy_used)


def _scatter_success_from_computed(
    request: dict[str, Any],
    state: dict[str, Any],
    computed: dict[str, Any],
    proxy_used: bool = False,
) -> dict[str, Any]:
    if computed.get("status") != "success":
        return computed
    payload = _payload(request, state, computed.get("data", []), computed.get("data_source"))
    tickers = computed.get("tickers") or payload.get("tickers_used") or []
    payload.update(
        {
            "series": computed.get("series", []),
            "xAxis": computed.get("xAxis"),
            "yAxis": computed.get("yAxis"),
            "zAxis": computed.get("zAxis"),
            "tickers": tickers,
            "tickers_used": tickers,
            "ticker_count": len(tickers),
            "point_count": computed.get("point_count", len(computed.get("data", []))),
            "x_axis": computed.get("x_axis") or payload.get("x_axis"),
            "y_axis": computed.get("y_axis") or payload.get("y_axis"),
            "x_unit": computed.get("x_unit") or payload.get("x_unit"),
            "y_unit": computed.get("y_unit") or payload.get("y_unit"),
            "point_id": computed.get("point_id") or payload.get("point_id"),
            "color_axis": computed.get("color_axis") or payload.get("color_axis"),
            "size_axis": computed.get("size_axis") or payload.get("size_axis"),
            "size_unit": computed.get("size_unit"),
            "title": computed.get("title") or payload.get("title"),
            "grid": computed.get("grid", {"horizontal": True, "vertical": True}),
            "hitAreaRadius": computed.get("hitAreaRadius", 24),
            "date_range": computed.get("date_range") or payload.get("date_range"),
            "data_source": computed.get("data_source") or payload.get("data_source"),
            "fallback_used": bool(computed.get("fallback_used") or proxy_used),
            "fallback_method": computed.get("fallback_method"),
            "formula": computed.get("formula"),
            "confidence": computed.get("confidence", payload.get("confidence", "High")),
            "limitations": computed.get("limitations", []),
            "regression_used": bool(computed.get("regression_used")),
            "regression_method": computed.get("regression_method"),
            "regression_line": computed.get("regression_line"),
            "r_squared": computed.get("r_squared"),
            "sampled_points": bool(computed.get("sampled_points")),
            "raw_point_count": computed.get("raw_point_count"),
            "optimizer_called": False,
            "advisory_allocation_generated": False,
        }
    )
    for optional_key in ("covariance_matrix", "component", "chart_type", "chart_tier", "requires_premium", "fallback_chart"):
        if computed.get(optional_key) is not None:
            payload[optional_key] = computed[optional_key]
    return _validated_success(payload, request, proxy_used)


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
            "title": "Ticker Concentration",
            "x_axis": "allocation_percent",
            "y_axis": "ticker",
            "unit": "%",
        },
        "plot_43_sector_concentration_plot": {
            "plot_short_id": "sector_concentration_plot",
            "title": "Sector Concentration",
            "x_axis": "allocation_percent",
            "y_axis": "sector",
            "unit": "%",
        },
        "plot_39_hhi_concentration_index": {
            "plot_short_id": "hhi_concentration_index",
            "title": "HHI Concentration Index",
            "x_axis": "portfolio",
            "y_axis": "hhi_value",
            "unit": "ratio",
        },
        "plot_40_effective_number_of_holdings": {
            "plot_short_id": "effective_number_of_holdings",
            "title": "Effective Number of Holdings",
            "x_axis": "portfolio",
            "y_axis": "effective_count",
            "unit": "count",
        },
        "plot_55_risk_contribution_by_ticker": {
            "plot_short_id": "risk_contribution_by_ticker",
            "title": "Risk Contribution by Ticker",
            "x_axis": "risk_contribution_percent",
            "y_axis": "ticker",
            "unit": "%",
        },
        "plot_29_current_vs_advisory_allocation_by_ticker": {
            "plot_short_id": "current_vs_advisory_allocation_by_ticker",
            "title": "Current vs Advisory Allocation by Ticker",
            "x_axis": "ticker",
            "y_axis": "allocation_percent",
            "unit": "%",
        },
        "plot_32_allocation_change_by_ticker": {
            "plot_short_id": "allocation_change_by_ticker",
            "title": "Allocation Change by Ticker",
            "x_axis": "ticker",
            "y_axis": "allocation_change_percent",
            "unit": "%",
        },
        "plot_61_eigenvector_centrality_by_ticker": {
            "plot_short_id": "eigenvector_centrality_by_ticker",
            "title": "Eigenvector Centrality by Ticker",
            "x_axis": "centrality_score",
            "y_axis": "ticker",
            "unit": "decimal",
        },
        "return_range_by_ticker": {
            "plot_short_id": "return_range_by_ticker",
            "title": "Return Range by Ticker",
            "x_axis": "ticker",
            "y_axis": "return_range_percent",
            "unit": "%",
        },
        "volatility_range_by_ticker": {
            "plot_short_id": "volatility_range_by_ticker",
            "title": "Volatility Range by Ticker",
            "x_axis": "ticker",
            "y_axis": "volatility_range_percent",
            "unit": "%",
        },
        "portfolio_return_waterfall": {
            "plot_short_id": "portfolio_return_waterfall",
            "title": "Portfolio Return Contribution Waterfall",
            "x_axis": "component",
            "y_axis": "return_contribution_percent",
            "unit": "%",
        },
        "risk_contribution_waterfall": {
            "plot_short_id": "risk_contribution_waterfall",
            "title": "Risk Contribution Waterfall",
            "x_axis": "component",
            "y_axis": "risk_contribution_percent",
            "unit": "%",
        },
        "current_vs_advisory_mirrored_bar": {
            "plot_short_id": "current_vs_advisory_mirrored_bar",
            "title": "Current Exposure vs Advisory Exposure",
            "x_axis": "weight_percent",
            "y_axis": "ticker",
            "unit": "%",
        },
        "return_distribution_histogram": {
            "plot_short_id": "return_distribution_histogram",
            "title": "Return Distribution Histogram",
            "x_axis": "return_bin",
            "y_axis": "count",
            "unit": "count",
        },
        "historical_adjusted_close": {
            "plot_short_id": "historical_adjusted_close",
            "title": "Historical Adjusted Close",
            "x_axis": "date",
            "y_axis": "adjusted_close",
            "unit": "price",
        },
        "normalized_price_comparison": {
            "plot_short_id": "normalized_price_comparison",
            "title": "Normalized Price Comparison",
            "x_axis": "date",
            "y_axis": "normalized_price",
            "unit": "index",
        },
        "portfolio_value_over_time": {
            "plot_short_id": "portfolio_value_over_time",
            "title": "Portfolio Value Over Time",
            "x_axis": "date",
            "y_axis": "portfolio_value",
            "unit": "currency",
        },
        "drawdown_over_time": {
            "plot_short_id": "drawdown_over_time",
            "title": "Drawdown Over Time",
            "x_axis": "date",
            "y_axis": "drawdown_percent",
            "unit": "%",
        },
        "rolling_volatility_over_time": {
            "plot_short_id": "rolling_volatility_over_time",
            "title": "Rolling Volatility Over Time",
            "x_axis": "date",
            "y_axis": "rolling_volatility_percent",
            "unit": "%",
        },
        "rolling_correlation_over_time": {
            "plot_short_id": "rolling_correlation_over_time",
            "title": "Rolling Average Correlation",
            "x_axis": "date",
            "y_axis": "average_correlation",
            "unit": "decimal",
        },
        "instability_index_over_time": {
            "plot_short_id": "instability_index_over_time",
            "title": "Instability Index Over Time",
            "x_axis": "date",
            "y_axis": "instability_index",
            "unit": "decimal",
        },
        "cvar_over_time": {
            "plot_short_id": "cvar_over_time",
            "title": "CVaR Over Time",
            "x_axis": "date",
            "y_axis": "cvar_95",
            "unit": "%",
        },
        "governance_metric_threshold_over_time": {
            "plot_short_id": "governance_metric_threshold_over_time",
            "title": "Governance Metric Threshold Over Time",
            "x_axis": "date",
            "y_axis": "metric_value",
            "unit": "decimal",
        },
        "sector_allocation_donut": {
            "plot_short_id": "sector_allocation_donut",
            "title": "Sector Allocation Donut",
            "x_axis": "sector",
            "y_axis": "weight_percent",
            "unit": "%",
        },
        "ticker_allocation_donut": {
            "plot_short_id": "ticker_allocation_donut",
            "title": "Ticker Allocation Donut",
            "x_axis": "ticker",
            "y_axis": "weight_percent",
            "unit": "%",
        },
        "risk_contribution_donut": {
            "plot_short_id": "risk_contribution_donut",
            "title": "Risk Contribution Donut",
            "x_axis": "name",
            "y_axis": "risk_contribution_percent",
            "unit": "%",
        },
        "sector_ticker_nested_donut": {
            "plot_short_id": "sector_ticker_nested_donut",
            "title": "Sector and Ticker Nested Donut",
            "x_axis": "ticker",
            "y_axis": "ticker_weight_percent",
            "unit": "%",
        },
        "portfolio_health_donut": {
            "plot_short_id": "portfolio_health_donut",
            "title": "Portfolio Health Donut",
            "x_axis": "health_component",
            "y_axis": "score",
            "unit": "score",
        },
        "semi_donut_risk_gauge": {
            "plot_short_id": "semi_donut_risk_gauge",
            "title": "Portfolio Risk Gauge",
            "x_axis": "health_component",
            "y_axis": "score",
            "unit": "score",
        },
        "risk_return_scatter": {
            "plot_short_id": "risk_return_scatter",
            "title": "Risk-Return Scatter by Ticker",
            "x_axis": "annualized_volatility_percent",
            "y_axis": "annualized_return_percent",
            "unit": "%",
        },
        "cvar_return_scatter": {
            "plot_short_id": "cvar_return_scatter",
            "title": "CVaR-Return Scatter by Ticker",
            "x_axis": "cvar_95_percent",
            "y_axis": "annualized_return_percent",
            "unit": "%",
        },
        "drawdown_return_scatter": {
            "plot_short_id": "drawdown_return_scatter",
            "title": "Drawdown-Return Scatter by Ticker",
            "x_axis": "max_drawdown_percent",
            "y_axis": "annualized_return_percent",
            "unit": "%",
        },
        "beta_return_scatter": {
            "plot_short_id": "beta_return_scatter",
            "title": "Beta-Return Scatter by Ticker",
            "x_axis": "beta",
            "y_axis": "annualized_return_percent",
            "unit": "mixed",
        },
        "allocation_vs_risk_contribution_scatter": {
            "plot_short_id": "allocation_vs_risk_contribution_scatter",
            "title": "Allocation vs Risk Contribution Scatter",
            "x_axis": "allocation_percent",
            "y_axis": "risk_contribution_percent",
            "unit": "%",
        },
        "bubble_risk_return": {
            "plot_short_id": "bubble_risk_return",
            "title": "Risk-Return Bubble by Ticker",
            "x_axis": "annualized_volatility_percent",
            "y_axis": "annualized_return_percent",
            "unit": "%",
        },
        "scatter_with_regression_line": {
            "plot_short_id": "scatter_with_regression_line",
            "title": "Return Regression Scatter",
            "x_axis": "x_return_percent",
            "y_axis": "y_return_percent",
            "unit": "%",
        },
        "ownership_overlap_correlation_scatter": {
            "plot_short_id": "ownership_overlap_correlation_scatter",
            "title": "Ownership Overlap vs Return Correlation",
            "x_axis": "ownership_overlap",
            "y_axis": "return_correlation",
            "unit": "decimal",
        },
        "premium_webgl_scatter": {
            "plot_short_id": "premium_webgl_scatter",
            "title": "Premium WebGL Scatter",
            "x_axis": "x",
            "y_axis": "y",
            "unit": "none",
        },
    }
    return mapping.get(
        str(plot_id or ""),
        {"plot_short_id": str(plot_id or ""), "title": str(plot_id or "Chart"), "x_axis": "metric", "y_axis": "value", "unit": "none"},
    )
