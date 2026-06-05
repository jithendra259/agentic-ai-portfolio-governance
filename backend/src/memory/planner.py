from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import uuid4

from src.memory.todo_state import build_todo, set_todo_status


PLOTS_REQUIRING_CURRENT_WEIGHTS = {
    "plot_29_current_vs_advisory_allocation_by_ticker",
    "plot_32_allocation_change_by_ticker",
    "plot_39_hhi_concentration_index",
    "plot_40_effective_number_of_holdings",
    "plot_42_ticker_concentration_plot",
    "plot_43_sector_concentration_plot",
    "plot_55_risk_contribution_by_ticker",
    "plot_56_allocation_vs_risk_contribution_plot",
    "portfolio_return_waterfall",
    "risk_contribution_waterfall",
    "return_distribution_histogram",
    "portfolio_value_over_time",
    "drawdown_over_time",
    "cvar_over_time",
    "sector_allocation_donut",
    "ticker_allocation_donut",
    "risk_contribution_donut",
    "sector_ticker_nested_donut",
    "portfolio_health_donut",
    "semi_donut_risk_gauge",
    "allocation_vs_risk_contribution_scatter",
    "bubble_risk_return",
}

PLOTS_REQUIRING_ADVISORY_WEIGHTS = {
    "plot_29_current_vs_advisory_allocation_by_ticker",
    "plot_32_allocation_change_by_ticker",
    "plot_56_allocation_vs_risk_contribution_plot",
    "current_vs_advisory_mirrored_bar",
}

PLOTS_REQUIRING_COVARIANCE = {
    "plot_55_risk_contribution_by_ticker",
    "plot_56_allocation_vs_risk_contribution_plot",
    "risk_contribution_waterfall",
    "risk_contribution_donut",
    "allocation_vs_risk_contribution_scatter",
}

PLOTS_REQUIRING_GRAPH_DATA = {
    "plot_61_eigenvector_centrality_by_ticker",
    "ownership_overlap_correlation_scatter",
}

PLOT_INTENTS = {
    "plot_29_current_vs_advisory_allocation_by_ticker": ("current_vs_advisory_allocation_plot", "advisory_allocation"),
    "plot_32_allocation_change_by_ticker": ("allocation_change_plot", "advisory_allocation"),
    "plot_39_hhi_concentration_index": ("hhi_concentration_plot", "diversification_only"),
    "plot_40_effective_number_of_holdings": ("effective_holdings_plot", "diversification_only"),
    "plot_42_ticker_concentration_plot": ("ticker_concentration_plot", "plot_only"),
    "plot_43_sector_concentration_plot": ("sector_concentration_plot", "plot_only"),
    "plot_55_risk_contribution_by_ticker": ("risk_contribution_plot", "plot_plus_diagnosis"),
    "plot_56_allocation_vs_risk_contribution_plot": ("allocation_vs_risk_contribution_plot", "plot_plus_diagnosis"),
    "plot_61_eigenvector_centrality_by_ticker": ("eigenvector_centrality_plot", "plot_plus_diagnosis"),
    "plot_81_strategy_performance_comparison": ("strategy_performance_comparison", "plot_plus_diagnosis"),
    "return_range_by_ticker": ("return_range_by_ticker", "plot_plus_diagnosis"),
    "volatility_range_by_ticker": ("volatility_range_by_ticker", "plot_plus_diagnosis"),
    "portfolio_return_waterfall": ("portfolio_return_waterfall", "plot_plus_diagnosis"),
    "risk_contribution_waterfall": ("risk_contribution_waterfall", "plot_plus_diagnosis"),
    "current_vs_advisory_mirrored_bar": ("current_vs_advisory_mirrored_bar", "advisory_allocation"),
    "return_distribution_histogram": ("return_distribution_histogram", "plot_plus_diagnosis"),
    "historical_adjusted_close": ("historical_adjusted_close", "plot_only"),
    "normalized_price_comparison": ("normalized_price_comparison", "plot_only"),
    "portfolio_value_over_time": ("portfolio_value_over_time", "plot_plus_diagnosis"),
    "drawdown_over_time": ("drawdown_over_time", "plot_plus_diagnosis"),
    "rolling_volatility_over_time": ("rolling_volatility_over_time", "plot_plus_diagnosis"),
    "rolling_correlation_over_time": ("rolling_correlation_over_time", "plot_plus_diagnosis"),
    "instability_index_over_time": ("instability_index_over_time", "regime_only"),
    "cvar_over_time": ("cvar_over_time", "plot_plus_diagnosis"),
    "governance_metric_threshold_over_time": ("governance_metric_threshold_over_time", "plot_plus_diagnosis"),
    "sector_allocation_donut": ("sector_allocation_donut", "plot_only"),
    "ticker_allocation_donut": ("ticker_allocation_donut", "plot_only"),
    "risk_contribution_donut": ("risk_contribution_donut", "plot_plus_diagnosis"),
    "sector_ticker_nested_donut": ("sector_ticker_nested_donut", "plot_only"),
    "portfolio_health_donut": ("portfolio_health_donut", "diversification_only"),
    "semi_donut_risk_gauge": ("semi_donut_risk_gauge", "plot_plus_diagnosis"),
    "risk_return_scatter": ("risk_return_scatter", "plot_plus_diagnosis"),
    "cvar_return_scatter": ("cvar_return_scatter", "plot_plus_diagnosis"),
    "drawdown_return_scatter": ("drawdown_return_scatter", "plot_plus_diagnosis"),
    "beta_return_scatter": ("beta_return_scatter", "plot_plus_diagnosis"),
    "allocation_vs_risk_contribution_scatter": ("allocation_vs_risk_contribution_scatter", "plot_plus_diagnosis"),
    "bubble_risk_return": ("bubble_risk_return", "plot_plus_diagnosis"),
    "scatter_with_regression_line": ("scatter_with_regression_line", "plot_plus_diagnosis"),
    "ownership_overlap_correlation_scatter": ("ownership_overlap_correlation_scatter", "plot_plus_diagnosis"),
}

SUB_INTENT_SCOPES = {
    "data_quality": ("data_quality", "data_quality_only"),
    "eda": ("eda_analysis", "plot_plus_diagnosis"),
    "correlation_covariance": ("correlation_covariance_analysis", "plot_plus_diagnosis"),
    "instability_regime": ("instability_regime", "regime_only"),
    "diversification": ("diversification_diagnostics", "diversification_only"),
    "risk_governance": ("risk_governance", "plot_plus_diagnosis"),
    "advisory_allocation": ("advisory_allocation", "advisory_allocation"),
    "graph_contagion": ("graph_contagion", "plot_plus_diagnosis"),
    "backtesting": ("backtesting_comparison", "plot_plus_diagnosis"),
    "smart_plot_selection": ("smart_plot_selection", "plot_only"),
}


class GovernanceTaskPlanner:
    """Deterministic todo planner for memory, fallback, and plot safety gates."""

    def build_plan(
        self,
        *,
        message: str,
        session_state: dict[str, Any],
        router_plan: dict[str, Any],
        intent_lock: dict[str, Any],
        universe: str | None,
        tickers: list[str],
        plot_request: dict[str, Any] | None,
        direct_response: dict[str, Any] | None,
        validation_result: dict[str, Any],
        pending_action: dict[str, Any] | None,
        proxy_approved: bool,
        missing_inputs: list[str],
    ) -> dict[str, Any]:
        plot_id = plot_request.get("requested_plot_id") if plot_request else None
        user_intent, response_scope = self._intent_and_scope(router_plan, intent_lock, plot_id, direct_response)
        todos = self._build_todos(
            user_intent=user_intent,
            response_scope=response_scope,
            state=session_state,
            universe=universe,
            tickers=tickers,
            plot_id=plot_id,
            direct_response=direct_response,
            validation_result=validation_result,
            pending_action=pending_action,
            proxy_approved=proxy_approved,
            missing_inputs=missing_inputs,
        )
        plan = {
            "plan_id": f"plan_{uuid4().hex[:12]}",
            "session_id": session_state.get("session_id"),
            "user_intent": user_intent,
            "response_scope": response_scope,
            "universe": universe,
            "requested_plot_id": plot_id,
            "todos": todos,
            "blocked_step": _first_blocked_step(todos),
            "pending_user_choice": self._pending_user_choice(todos, pending_action),
            "ready_to_execute": _ready_to_execute(todos),
            "ready_to_render": _ready_to_render(todos, plot_id),
        }
        return plan

    def update_after_fallback(
        self,
        state: dict[str, Any],
        result: dict[str, Any],
        plot_request: dict[str, Any] | None,
    ) -> dict[str, Any]:
        plan = deepcopy(state.get("current_plan"))
        if not isinstance(plan, dict) or not plan:
            return state

        plot_id = (plot_request or {}).get("requested_plot_id") or plan.get("requested_plot_id")
        status = result.get("status")
        validation = result.get("validation_result") or {}
        payload = result.get("plot_payload") if isinstance(result.get("plot_payload"), dict) else {}
        weights = state.get("active_weights", {}) if isinstance(state.get("active_weights"), dict) else {}

        if status == "success":
            plan = self._mark_weight_resolution(plan, weights)
            if plot_id in LINE_PLOT_IDS:
                plan = set_todo_status(
                    plan,
                    "fetch_price_history",
                    "computed_by_fallback",
                    resolved_data={
                        "ticker_count": len(payload.get("tickers_used", [])),
                        "data_source": payload.get("data_source"),
                    },
                    fallback_method=payload.get("fallback_method") or "historical_price_lookup",
                    reason="Line chart price history was resolved through deterministic fallback computation.",
                )
                plan = set_todo_status(
                    plan,
                    "align_time_series",
                    "done",
                    resolved_data={
                        "row_count": len(payload.get("data", [])) if isinstance(payload.get("data"), list) else 0,
                        "connect_nulls": bool(payload.get("connect_nulls")),
                    },
                )
                plan = set_todo_status(
                    plan,
                    "compute_line_metric_if_needed",
                    "done",
                    resolved_data={"formula": payload.get("formula"), "y_axis": payload.get("y_axis")},
                )
            if plot_id in PIE_PLOT_IDS:
                if plot_id in PIE_PLOTS_REQUIRING_SECTOR_MAPPING:
                    plan = set_todo_status(
                        plan,
                        "resolve_sector_mapping",
                        "done",
                        resolved_data={
                            "sector_hhi": payload.get("metrics", {}).get("sector_hhi"),
                            "data_source": payload.get("data_source"),
                        },
                        reason="Sector mapping was resolved before building the donut payload.",
                    )
                plan = set_todo_status(
                    plan,
                    "compute_pie_metric",
                    "done",
                    resolved_data={
                        "category_field": payload.get("category_field"),
                        "value_field": payload.get("value_field"),
                        "total_value": payload.get("total_value"),
                    },
                )
                plan = set_todo_status(
                    plan,
                    "check_slice_count",
                    "done",
                    resolved_data={
                        "slice_count": payload.get("slice_count"),
                        "max_slices": payload.get("max_slices") or payload.get("max_outer_slices"),
                        "fallback_rendered": bool(payload.get("fallback_rendered")),
                    },
                )
            if plot_id in SCATTER_PLOT_IDS:
                plan = set_todo_status(
                    plan,
                    "fetch_price_history",
                    "computed_by_fallback",
                    resolved_data={
                        "ticker_count": len(payload.get("tickers_used", [])),
                        "data_source": payload.get("data_source"),
                    },
                    fallback_method=payload.get("fallback_method") or "historical_price_lookup",
                    reason="Scatter chart price history was resolved through deterministic fallback computation.",
                )
                plan = set_todo_status(
                    plan,
                    "compute_returns_if_needed",
                    "done",
                    resolved_data={"formula": payload.get("formula")},
                )
                plan = set_todo_status(
                    plan,
                    "compute_scatter_x_metric",
                    "done",
                    resolved_data={"x_axis": payload.get("x_axis"), "x_unit": payload.get("x_unit")},
                )
                plan = set_todo_status(
                    plan,
                    "compute_scatter_y_metric",
                    "done",
                    resolved_data={"y_axis": payload.get("y_axis"), "y_unit": payload.get("y_unit")},
                )
                if plot_id in SCATTER_PLOTS_REQUIRING_SIZE:
                    plan = set_todo_status(
                        plan,
                        "compute_optional_size_metric",
                        "done",
                        resolved_data={"size_axis": payload.get("size_axis"), "size_unit": payload.get("size_unit")},
                    )
                if plot_id in SCATTER_PLOTS_WITH_COLOR:
                    plan = set_todo_status(
                        plan,
                        "compute_optional_color_group",
                        "done",
                        resolved_data={"color_axis": payload.get("color_axis")},
                    )
                if plot_id in SCATTER_PLOTS_REQUIRING_REGRESSION:
                    plan = set_todo_status(
                        plan,
                        "compute_regression_if_requested",
                        "done",
                        resolved_data={
                            "regression_method": payload.get("regression_method"),
                            "r_squared": payload.get("r_squared"),
                        },
                    )
            if plot_id in PLOTS_REQUIRING_COVARIANCE and payload.get("covariance_matrix"):
                plan = set_todo_status(
                    plan,
                    "resolve_covariance",
                    "computed_by_fallback",
                    resolved_data={"source": "historical_returns", "ticker_count": len(payload.get("covariance_matrix", {}))},
                    fallback_method="covariance_from_returns",
                    reason="Covariance was computed from aligned ticker returns.",
                )
            plan = set_todo_status(
                plan,
                "generate_plot_payload",
                "done",
                resolved_data={
                    "plot_id": payload.get("plot_id"),
                    "row_count": len(payload.get("data", [])) if isinstance(payload.get("data"), list) else 0,
                    "data_source": payload.get("data_source"),
                },
            )
            plan = set_todo_status(
                plan,
                "validate_plot",
                "done",
                resolved_data={"can_render": bool(validation.get("can_render", True))},
            )
            plan["blocked_step"] = None
            plan["pending_user_choice"] = None
            plan["ready_to_execute"] = True
            plan["ready_to_render"] = bool(validation.get("can_render", True))
        elif status == "failed_validation":
            plan = self._mark_weight_resolution(plan, weights)
            if payload:
                plan = set_todo_status(
                    plan,
                    "generate_plot_payload",
                    "done",
                    resolved_data={"plot_id": payload.get("plot_id"), "row_count": len(payload.get("data", []))},
                )
            plan = set_todo_status(
                plan,
                "validate_plot",
                "failed_validation",
                resolved_data={"can_render": False},
                reason=validation.get("reason") or "Plot payload failed schema or context validation.",
            )
            plan["blocked_step"] = "validate_plot"
            plan["pending_user_choice"] = None
            plan["ready_to_execute"] = False
            plan["ready_to_render"] = False
        elif status == "missing_data":
            plan = self._mark_missing_inputs(plan, result.get("missing_inputs", []), result.get("reason"))
        elif status in {"unavailable", "requires_confirmation"}:
            reason = result.get("reason")
            missing_inputs = result.get("missing_inputs", [])
            if "institutional_graph_data" in missing_inputs:
                plan = set_todo_status(
                    plan,
                    "resolve_graph_data",
                    "blocked",
                    reason=reason,
                    user_options=[],
                )
                plan["blocked_step"] = "resolve_graph_data"
            elif "optimizer_permission" in missing_inputs:
                plan = set_todo_status(
                    plan,
                    "request_optimizer_permission",
                    "requires_user_input",
                    reason=reason,
                    user_options=["approve_optimizer_for_strategy_comparison"],
                )
                plan["blocked_step"] = "request_optimizer_permission"
                plan["pending_user_choice"] = "approve_optimizer_for_strategy_comparison"
            elif "sector_mapping" in missing_inputs:
                plan = set_todo_status(
                    plan,
                    "resolve_sector_mapping",
                    "blocked",
                    reason=reason,
                    user_options=[],
                )
                plan["blocked_step"] = "resolve_sector_mapping"
            elif "smaller_ticker_set" in missing_inputs:
                plan = set_todo_status(
                    plan,
                    "check_slice_count",
                    "blocked",
                    reason=reason,
                    user_options=[],
                )
                plan["blocked_step"] = "check_slice_count"
            elif "non_negative_risk_contribution" in missing_inputs:
                plan = set_todo_status(
                    plan,
                    "compute_pie_metric",
                    "blocked",
                    reason=reason,
                    user_options=[],
                )
                plan["blocked_step"] = "compute_pie_metric"
            plan["ready_to_execute"] = False
            plan["ready_to_render"] = False

        next_state = dict(state)
        next_state["current_plan"] = plan
        return next_state

    def _intent_and_scope(
        self,
        router_plan: dict[str, Any],
        intent_lock: dict[str, Any],
        plot_id: str | None,
        direct_response: dict[str, Any] | None,
    ) -> tuple[str, str]:
        if isinstance(direct_response, dict):
            if direct_response.get("type") == "universe_ticker_list":
                return "list_universe_tickers", "data_lookup"
            if direct_response.get("type") == "real_current_weights_required":
                return "explain_real_current_weights", "data_lookup"
        if plot_id in PLOT_INTENTS:
            return PLOT_INTENTS[plot_id]
        sub_intent = router_plan.get("sub_intent")
        if sub_intent in SUB_INTENT_SCOPES:
            return SUB_INTENT_SCOPES[sub_intent]
        locked = intent_lock.get("intent")
        if locked == "regime_diagnosis_only":
            return "instability_regime", "regime_only"
        return sub_intent or router_plan.get("main_intent") or "general_chat", "data_lookup"

    def _build_todos(
        self,
        *,
        user_intent: str,
        response_scope: str,
        state: dict[str, Any],
        universe: str | None,
        tickers: list[str],
        plot_id: str | None,
        direct_response: dict[str, Any] | None,
        validation_result: dict[str, Any],
        pending_action: dict[str, Any] | None,
        proxy_approved: bool,
        missing_inputs: list[str],
    ) -> list[dict[str, Any]]:
        if user_intent == "list_universe_tickers":
            return [
                _resolve_universe_todo(1, universe),
                build_todo(
                    2,
                    "return_ticker_list",
                    status="done" if tickers else "blocked",
                    required_data=["tickers"],
                    resolved_data={"ticker_count": len(tickers), "source": "universe_registry" if universe else "active_session_context"},
                    reason=None if tickers else "No tickers are available in current context.",
                ),
            ]
        if user_intent == "explain_real_current_weights":
            return [
                build_todo(
                    1,
                    "explain_holdings_required",
                    status="done",
                    required_data=["shares_or_invested_amounts_or_current_weights"],
                    resolved_data={"available": False},
                    reason="Real current allocation weights require holdings data, not price history alone.",
                )
            ]

        todos = [_resolve_universe_todo(1, universe), _resolve_tickers_todo(2, tickers)]
        step = 3
        if plot_id in PLOTS_REQUIRING_CURRENT_WEIGHTS:
            todos.append(self._resolve_weights_todo(step, state, pending_action, proxy_approved, missing_inputs))
            step += 1
        if plot_id in PLOTS_REQUIRING_ADVISORY_WEIGHTS:
            todos.append(self._resolve_advisory_weights_todo(step, state, missing_inputs))
            step += 1
        if plot_id in PLOTS_REQUIRING_COVARIANCE:
            todos.append(
                build_todo(
                    step,
                    "resolve_covariance",
                    status="pending",
                    required_data=["aligned_returns"],
                    can_fallback=True,
                    fallback_method="covariance_from_returns",
                )
            )
            step += 1
        if plot_id in PLOTS_REQUIRING_GRAPH_DATA:
            todos.append(
                build_todo(
                    step,
                    "resolve_graph_data",
                    status="blocked",
                    required_data=["institutional_graph_data"],
                    can_fallback=False,
                    reason="Institutional graph data is required; the planner will not invent centrality values.",
                )
            )
            step += 1
        if plot_id in LINE_PLOT_IDS:
            todos.extend(
                [
                    build_todo(
                        step,
                        "resolve_date_range",
                        status="done" if _has_date_range(state) else "computed_by_fallback",
                        required_data=["date_range"],
                        resolved_data=state.get("active_date_range", {}) if _has_date_range(state) else {},
                        can_fallback=True,
                        fallback_method="default_date_range" if not _has_date_range(state) else None,
                        reason=None if _has_date_range(state) else "No explicit date range was supplied; fallback computation will use the default governance test range.",
                    ),
                    build_todo(step + 1, "fetch_price_history", status="pending", required_data=["historical_adjusted_close"]),
                    build_todo(step + 2, "align_time_series", status="pending", required_data=["date", "ticker_series"]),
                    build_todo(step + 3, "compute_line_metric_if_needed", status="pending", required_data=["line_metric"]),
                ]
            )
            step += 4
        if plot_id in PIE_PLOT_IDS:
            if plot_id in PIE_PLOTS_REQUIRING_SECTOR_MAPPING:
                todos.append(
                    build_todo(
                        step,
                        "resolve_sector_mapping",
                        status="pending",
                        required_data=["sector_mapping"],
                        can_fallback=True,
                        fallback_method="default_sector_map",
                    )
                )
                step += 1
            todos.extend(
                [
                    build_todo(
                        step,
                        "compute_pie_metric",
                        status="pending",
                        required_data=["category_field", "value_field", "non_negative_values"],
                    ),
                    build_todo(
                        step + 1,
                        "check_slice_count",
                        status="pending",
                        required_data=["slice_count", "max_slices"],
                    ),
                ]
            )
            step += 2
        if plot_id in SCATTER_PLOT_IDS:
            todos.extend(
                [
                    build_todo(
                        step,
                        "resolve_date_range",
                        status="done" if _has_date_range(state) else "computed_by_fallback",
                        required_data=["date_range"],
                        resolved_data=state.get("active_date_range", {}) if _has_date_range(state) else {},
                        can_fallback=True,
                        fallback_method="default_date_range" if not _has_date_range(state) else None,
                        reason=None if _has_date_range(state) else "No explicit date range was supplied; fallback computation will use the default governance test range.",
                    ),
                    build_todo(step + 1, "fetch_price_history", status="pending", required_data=["historical_adjusted_close"]),
                    build_todo(step + 2, "compute_returns_if_needed", status="pending", required_data=["daily_log_returns"]),
                    build_todo(step + 3, "compute_scatter_x_metric", status="pending", required_data=["x_axis_metric"]),
                    build_todo(step + 4, "compute_scatter_y_metric", status="pending", required_data=["y_axis_metric"]),
                ]
            )
            step += 5
            if plot_id in SCATTER_PLOTS_REQUIRING_SIZE:
                todos.append(
                    build_todo(
                        step,
                        "compute_optional_size_metric",
                        status="pending",
                        required_data=["size_axis_metric"],
                    )
                )
                step += 1
            if plot_id in SCATTER_PLOTS_WITH_COLOR:
                todos.append(
                    build_todo(
                        step,
                        "compute_optional_color_group",
                        status="pending",
                        required_data=["color_axis_group"],
                        can_fallback=True,
                        fallback_method="default_sector_map",
                    )
                )
                step += 1
            if plot_id in SCATTER_PLOTS_REQUIRING_REGRESSION:
                todos.append(
                    build_todo(
                        step,
                        "compute_regression_if_requested",
                        status="pending",
                        required_data=["at_least_three_aligned_points", "ols_regression"],
                    )
                )
                step += 1
        if plot_id == "plot_81_strategy_performance_comparison":
            todos.append(
                build_todo(
                    step,
                    "request_optimizer_permission",
                    status="requires_user_input",
                    required_data=["optimizer_permission"],
                    reason="Strategy comparison requires explicit optimizer permission.",
                    user_options=["approve_optimizer_for_strategy_comparison"],
                )
            )
            step += 1
        if plot_id:
            render_status = "pending"
            if not validation_result.get("can_execute", True):
                render_status = "blocked"
            todos.extend(
                [
                    build_todo(step, "generate_plot_payload", status=render_status, required_data=["validated_inputs"]),
                    build_todo(step + 1, "validate_plot", status="pending", required_data=["plot_payload", "planner_metadata"]),
                ]
            )
        elif response_scope in {"data_quality_only", "diversification_only", "regime_only"}:
            todos.append(
                build_todo(
                    step,
                    "run_diagnostic_only",
                    status="done",
                    required_data=[],
                    reason="Intent lock restricts the response to diagnosis without allocation.",
                )
            )
        elif direct_response is None:
            todos.append(build_todo(step, "route_general_response", status="done"))
        return todos

    def _resolve_weights_todo(
        self,
        step: int,
        state: dict[str, Any],
        pending_action: dict[str, Any] | None,
        proxy_approved: bool,
        missing_inputs: list[str],
    ) -> dict[str, Any]:
        weights = state.get("active_weights", {}) if isinstance(state.get("active_weights"), dict) else {}
        weight_values = weights.get("weights") if isinstance(weights.get("weights"), dict) else {}
        if weight_values:
            status = "computed_by_fallback" if weights.get("type") == "equal_weight_proxy" else "done"
            return build_todo(
                step,
                "resolve_weights",
                status=status,
                required_data=["current_weights"],
                resolved_data={
                    "weight_count": len(weight_values),
                    "weight_source": weights.get("type"),
                    "approved_by_user": bool(weights.get("approved_by_user")),
                },
                can_fallback=True,
                fallback_method="equal_weight_proxy" if weights.get("type") == "equal_weight_proxy" else None,
                reason="User approved equal-weight proxy in the prompt." if proxy_approved else None,
            )
        status = "requires_user_input" if "current_weights" in missing_inputs or pending_action else "pending"
        return build_todo(
            step,
            "resolve_weights",
            status=status,
            required_data=["current_weights"],
            can_fallback=True,
            fallback_method="equal_weight_proxy",
            reason="Current weights are missing. Equal-weight proxy needs user approval before rendering.",
            user_options=["approve_equal_weight_proxy"] if status == "requires_user_input" else [],
        )

    def _resolve_advisory_weights_todo(
        self,
        step: int,
        state: dict[str, Any],
        missing_inputs: list[str],
    ) -> dict[str, Any]:
        advisory = state.get("active_advisory_weights", {}) if isinstance(state.get("active_advisory_weights"), dict) else {}
        weights = advisory.get("weights") if isinstance(advisory.get("weights"), dict) else {}
        if advisory.get("available") and weights:
            return build_todo(
                step,
                "resolve_advisory_weights",
                status="done",
                required_data=["advisory_weights"],
                resolved_data={"weight_count": len(weights), "source": advisory.get("source")},
            )
        return build_todo(
            step,
            "resolve_advisory_weights",
            status="requires_user_input" if "advisory_weights" in missing_inputs else "pending",
            required_data=["advisory_weights"],
            reason="Advisory weights are required for this chart and cannot be invented.",
            user_options=["run_advisory_allocation"] if "advisory_weights" in missing_inputs else [],
        )

    def _mark_weight_resolution(self, plan: dict[str, Any], weights: dict[str, Any]) -> dict[str, Any]:
        weight_values = weights.get("weights") if isinstance(weights.get("weights"), dict) else {}
        if not weight_values:
            return plan
        if weights.get("type") == "equal_weight_proxy":
            return set_todo_status(
                plan,
                "resolve_weights",
                "computed_by_fallback",
                resolved_data={
                    "weight_count": len(weight_values),
                    "weight_source": "equal_weight_proxy",
                    "approved_by_user": bool(weights.get("approved_by_user")),
                },
                fallback_method="equal_weight_proxy",
                reason="Equal-weight proxy was explicitly approved before rendering.",
            )
        return set_todo_status(
            plan,
            "resolve_weights",
            "done",
            resolved_data={"weight_count": len(weight_values), "weight_source": weights.get("type")},
        )

    def _mark_missing_inputs(self, plan: dict[str, Any], missing_inputs: list[str], reason: str | None) -> dict[str, Any]:
        updated = deepcopy(plan)
        if "current_weights" in missing_inputs:
            updated = set_todo_status(
                updated,
                "resolve_weights",
                "requires_user_input",
                reason=reason or "Current weights are missing.",
                user_options=["approve_equal_weight_proxy"],
            )
            updated["blocked_step"] = "resolve_weights"
            updated["pending_user_choice"] = "approve_equal_weight_proxy"
        if "advisory_weights" in missing_inputs:
            updated = set_todo_status(
                updated,
                "resolve_advisory_weights",
                "requires_user_input",
                reason=reason or "Advisory weights are missing.",
                user_options=["run_advisory_allocation"],
            )
            updated["blocked_step"] = updated.get("blocked_step") or "resolve_advisory_weights"
            updated["pending_user_choice"] = updated.get("pending_user_choice") or "run_advisory_allocation"
        updated["ready_to_execute"] = False
        updated["ready_to_render"] = False
        return updated

    def _pending_user_choice(self, todos: list[dict[str, Any]], pending_action: dict[str, Any] | None) -> str | None:
        if pending_action and pending_action.get("type") == "use_equal_weight_proxy":
            return "approve_equal_weight_proxy"
        for todo in todos:
            if todo.get("status") == "requires_user_input" and todo.get("user_options"):
                return todo["user_options"][0]
        return None


def _resolve_universe_todo(step: int, universe: str | None) -> dict[str, Any]:
    return build_todo(
        step,
        "resolve_universe",
        status="done" if universe else "blocked",
        required_data=["universe"],
        resolved_data={"universe": universe} if universe else {},
        reason=None if universe else "No universe is available in current context.",
    )


def _resolve_tickers_todo(step: int, tickers: list[str]) -> dict[str, Any]:
    return build_todo(
        step,
        "resolve_tickers",
        status="done" if tickers else "blocked",
        required_data=["tickers"],
        resolved_data={"ticker_count": len(tickers), "tickers": list(tickers)} if tickers else {},
        reason=None if tickers else "No tickers are available in current context.",
    )


LINE_PLOT_IDS = {
    "historical_adjusted_close",
    "normalized_price_comparison",
    "portfolio_value_over_time",
    "drawdown_over_time",
    "rolling_volatility_over_time",
    "rolling_correlation_over_time",
    "instability_index_over_time",
    "cvar_over_time",
    "governance_metric_threshold_over_time",
}

PIE_PLOT_IDS = {
    "sector_allocation_donut",
    "ticker_allocation_donut",
    "risk_contribution_donut",
    "sector_ticker_nested_donut",
    "portfolio_health_donut",
    "semi_donut_risk_gauge",
}

PIE_PLOTS_REQUIRING_SECTOR_MAPPING = {
    "sector_allocation_donut",
    "sector_ticker_nested_donut",
    "portfolio_health_donut",
    "semi_donut_risk_gauge",
}

SCATTER_PLOT_IDS = {
    "risk_return_scatter",
    "cvar_return_scatter",
    "drawdown_return_scatter",
    "beta_return_scatter",
    "allocation_vs_risk_contribution_scatter",
    "bubble_risk_return",
    "scatter_with_regression_line",
    "ownership_overlap_correlation_scatter",
}

SCATTER_PLOTS_REQUIRING_SIZE = {
    "bubble_risk_return",
}

SCATTER_PLOTS_WITH_COLOR = {
    "risk_return_scatter",
    "cvar_return_scatter",
    "drawdown_return_scatter",
    "beta_return_scatter",
    "allocation_vs_risk_contribution_scatter",
    "bubble_risk_return",
}

SCATTER_PLOTS_REQUIRING_REGRESSION = {
    "scatter_with_regression_line",
}


def _has_date_range(state: dict[str, Any]) -> bool:
    date_range = state.get("active_date_range", {}) if isinstance(state.get("active_date_range"), dict) else {}
    return bool(date_range.get("start") and date_range.get("end"))


def _first_blocked_step(todos: list[dict[str, Any]]) -> str | None:
    for todo in todos:
        if todo.get("status") in {"requires_user_input", "blocked", "failed_validation"}:
            return todo.get("name")
    return None


def _ready_to_execute(todos: list[dict[str, Any]]) -> bool:
    return not any(todo.get("status") in {"requires_user_input", "blocked", "failed_validation"} for todo in todos)


def _ready_to_render(todos: list[dict[str, Any]], plot_id: str | None) -> bool:
    if not plot_id:
        return False
    validate = next((todo for todo in todos if todo.get("name") == "validate_plot"), None)
    if not validate:
        return False
    return validate.get("status") == "done" and _ready_to_execute(todos)
