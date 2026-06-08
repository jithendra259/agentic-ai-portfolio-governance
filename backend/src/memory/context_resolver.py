from __future__ import annotations

import re
from typing import Any

from src.intent.intent_router import IntentRouter
from src.memory.intent_lock import build_intent_lock
from src.memory.pending_action_manager import apply_pending_action, build_equal_weight_pending_action
from src.memory.planner import GovernanceTaskPlanner
from src.memory.session_state import KNOWN_UNIVERSE_TICKERS, PROJECT_POLICY_MEMORY


PLOT_ALIASES = {
    "ticker_concentration": ("plot_42_ticker_concentration_plot", "bar", "horizontal"),
    "sector_concentration": ("plot_43_sector_concentration_plot", "bar", "horizontal"),
    "risk_contribution": ("plot_55_risk_contribution_by_ticker", "bar", "horizontal"),
    "allocation_vs_risk_contribution": ("plot_56_allocation_vs_risk_contribution_plot", "bar", "grouped"),
    "current_vs_advisory": ("plot_29_current_vs_advisory_allocation_by_ticker", "bar", "grouped"),
    "allocation_change": ("plot_32_allocation_change_by_ticker", "bar", "diverging"),
    "hhi": ("plot_39_hhi_concentration_index", "bar", "vertical"),
    "effective_holdings": ("plot_40_effective_number_of_holdings", "bar", "vertical"),
    "eigenvector_centrality": ("plot_61_eigenvector_centrality_by_ticker", "bar", "horizontal"),
    "strategy_comparison": ("plot_81_strategy_performance_comparison", "bar", "grouped"),
    "return_range_by_ticker": ("return_range_by_ticker", "rangeBar", None),
    "volatility_range_by_ticker": ("volatility_range_by_ticker", "rangeBar", None),
    "portfolio_return_waterfall": ("portfolio_return_waterfall", "rangeBar", None),
    "risk_contribution_waterfall": ("risk_contribution_waterfall", "rangeBar", None),
    "current_vs_advisory_mirrored_bar": ("current_vs_advisory_mirrored_bar", "mirroredBar", None),
    "return_distribution_histogram": ("return_distribution_histogram", "histogram", None),
    "historical_adjusted_close": ("historical_adjusted_close", "line", None),
    "normalized_price_comparison": ("normalized_price_comparison", "line", None),
    "portfolio_value_over_time": ("portfolio_value_over_time", "line", None),
    "drawdown_over_time": ("drawdown_over_time", "line_area", None),
    "rolling_volatility_over_time": ("rolling_volatility_over_time", "line", None),
    "rolling_correlation_over_time": ("rolling_correlation_over_time", "line", None),
    "instability_index_over_time": ("instability_index_over_time", "line", None),
    "cvar_over_time": ("cvar_over_time", "line", None),
    "governance_metric_threshold_over_time": ("governance_metric_threshold_over_time", "line", None),
    "sector_allocation_donut": ("sector_allocation_donut", "donut", None),
    "ticker_allocation_donut": ("ticker_allocation_donut", "donut", None),
    "risk_contribution_donut": ("risk_contribution_donut", "donut", None),
    "sector_ticker_nested_donut": ("sector_ticker_nested_donut", "nested_donut", None),
    "portfolio_health_donut": ("portfolio_health_donut", "center_label_donut", None),
    "semi_donut_risk_gauge": ("semi_donut_risk_gauge", "semi_donut", None),
    "risk_return_scatter": ("risk_return_scatter", "scatter", None),
    "cvar_return_scatter": ("cvar_return_scatter", "scatter", None),
    "drawdown_return_scatter": ("drawdown_return_scatter", "scatter", None),
    "beta_return_scatter": ("beta_return_scatter", "scatter", None),
    "allocation_vs_risk_contribution_scatter": ("allocation_vs_risk_contribution_scatter", "scatter", None),
    "bubble_risk_return": ("bubble_risk_return", "bubble_scatter", None),
    "scatter_with_regression_line": ("scatter_with_regression_line", "scatter_regression", None),
    "ownership_overlap_correlation_scatter": ("ownership_overlap_correlation_scatter", "scatter", None),
}

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

TICKER_STOPWORDS = {
    "PLOT", "SHOW", "NOW", "USE", "THEM", "SAME", "YES", "NO", "DO", "IT", "FOR", "BY",
    "AND", "OR",
    "FROM", "TO", "WITH", "IS", "MY", "YOUR",
    "RISK", "CONTRIBUTION", "TICKER", "SECTOR", "CURRENT", "ADVISORY", "ALLOCATION",
    "GENERATE", "ONLY", "AS", "A", "AN", "BAR", "CHART", "HORIZONTAL", "IF", "WEIGHT",
    "WEIGHTS", "MISSING", "ARE", "EQUAL", "PROXY", "TEST", "CLEARLY", "LABEL", "RUN",
    "OPTIMIZER", "CONCENTRATION", "ALL",
    "NOT", "USING", "RANGE", "DAILY", "RETURN", "RETURNS", "MIN", "MAX", "MINIMUM",
    "MAXIMUM", "VOL", "VOLATILITY", "WATERFALL", "HISTOGRAM", "PREMIUM", "MUI",
    "ADJUSTED", "CLOSE", "PRICE", "TREND", "TRENDS", "NORMALIZED", "NORMALISED",
    "COMPARE", "COMPARISON", "PORTFOLIO", "VALUE", "DRAWDOWN", "OVER", "TIME",
    "ROLLING", "CORRELATION", "INSTABILITY", "INDEX", "CVAR", "VAR", "LINE",
    "CALM", "ELEVATED", "CRISIS", "REGIME",
    "PIE", "DONUT", "NESTED", "SLICE", "SLICES", "HEALTH", "GAUGE", "COMPOSITION",
    "SCATTER", "BUBBLE", "REGRESSION", "POINT", "POINTS", "VARIABLE", "VARIABLES",
    "RELATIONSHIP", "RELATIONSHIPS", "CLUSTER", "CLUSTERS", "OUTLIER", "OUTLIERS",
    "BETA", "OWNERSHIP", "OVERLAP", "GRAPH", "BENCHMARK", "EXPECTED", "ANNUALIZED",
    "ANNUALISED", "AXIS", "X", "Y", "Z",
    "PLOT_ID", "CHART_TYPE", "COUNT", "FIELD", "FIELDS", "EXPLAIN", "EXPLANATION",
    "WHAT", "DOT", "DOTS", "MEAN", "HISTORY", "HISTORICAL", "PROFILE", "PROFILES",
}


class ContextResolver:
    def __init__(self, intent_router: IntentRouter | None = None) -> None:
        self.intent_router = intent_router or IntentRouter()
        self.task_planner = GovernanceTaskPlanner()

    def resolve(
        self,
        message: str,
        session_state: dict[str, Any],
        chat_history_last_25: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        state = dict(session_state or {})
        state, pending_status = apply_pending_action(state, message)

        previous_analysis = _previous_analysis_from_state(state)
        plan = self.intent_router.build_execution_plan(message, previous_analysis=previous_analysis)
        intent_lock = build_intent_lock(message, plan)
        entities = plan.get("entities", {})

        universe = _extract_universe(message) or entities.get("universe") or state.get("active_universe")
        entity_tickers = _filter_tickers(entities.get("tickers") or [])
        reused_from_cache = plan.get("cache", {}).get("reuse_previous_analysis", False)
        explicit_subset = bool(entity_tickers) and not reused_from_cache and not _requests_full_universe_scope(message, universe)
        tickers = _resolve_tickers(universe, entity_tickers, state.get("active_tickers") or [], explicit_subset)
        start_date = entities.get("start_date") or state.get("active_date_range", {}).get("start")
        end_date = entities.get("end_date") or state.get("active_date_range", {}).get("end")

        plot_request = _extract_plot_request(message, state)
        if not plot_request and pending_status == "executed" and state.get("last_plot_id"):
            plot_request = _plot_request_from_state(state)
        _apply_negative_optimizer_instructions(plan, message, plot_request)
        proxy_approved = _proxy_approved_in_message(message)
        if plot_request and proxy_approved and not _has_current_weights(state):
            state["active_weights"] = {
                "type": "equal_weight_proxy",
                "weights": _equal_weight_percentages(tickers),
                "source": "same_prompt_user_approved",
                "approved_by_user": True,
            }
            state["pending_action"] = None
            if plot_request:
                plot_request["data_source"] = "equal_weight_proxy"

        direct_response = _build_direct_response(message, universe, tickers, state)
        missing_inputs: list[str] = []
        pending_action = state.get("pending_action")
        validation = {"can_execute": True, "reason": None}

        if plot_request:
            plot_id = plot_request["requested_plot_id"]
            if plot_id in PLOTS_REQUIRING_CURRENT_WEIGHTS and not _has_current_weights(state):
                missing_inputs.append("current_weights")
                validation = {
                    "can_execute": False,
                    "reason": "Current weights are missing. Equal-weight proxy requires user approval before rendering.",
                }
                pending_action = build_equal_weight_pending_action(
                    reason="current weights missing",
                    target_plot_id=plot_id,
                    tickers=tickers,
                    universe=universe,
                    chart_type=plot_request.get("requested_chart_type"),
                )
            if plot_id in PLOTS_REQUIRING_ADVISORY_WEIGHTS and not state.get("active_advisory_weights", {}).get("available"):
                missing_inputs.append("advisory_weights")
                validation = {
                    "can_execute": False,
                    "reason": "Advisory weights are missing. Ask before running optimizer.",
                }

        current_plan = self.task_planner.build_plan(
            message=message,
            session_state=state,
            router_plan=plan,
            intent_lock=intent_lock,
            universe=universe,
            tickers=tickers,
            plot_request=plot_request,
            direct_response=direct_response,
            validation_result=validation,
            pending_action=pending_action,
            proxy_approved=proxy_approved,
            missing_inputs=missing_inputs,
        )

        updates = {
            "last_user_message": message,
            "last_intent": plan.get("main_intent"),
            "last_sub_intent": plan.get("sub_intent"),
            "last_modules_called": _modules_called(plan),
            "last_modules_skipped": plan.get("safety", {}).get("modules_skipped", []),
            "missing_inputs": missing_inputs,
            "pending_action": pending_action,
            "direct_response": direct_response,
            "last_plan": state.get("current_plan"),
            "current_plan": current_plan,
            "warnings": [],
        }
        if universe:
            updates["active_universe"] = universe
        if tickers:
            updates["active_tickers"] = tickers
            updates["active_ticker_count"] = len(tickers)
        if start_date or end_date:
            updates["active_date_range"] = {
                "start": start_date,
                "end": end_date,
            }
        if pending_status == "executed":
            updates["last_response_summary"] = "Pending action executed with approved equal-weight proxy."
        if plot_request:
            plot_request["requested_subset_explicit"] = explicit_subset
            plot_request["requested_universe"] = universe or state.get("active_universe")
            plot_request["requested_tickers"] = tickers
            updates.update(
                {
                    "last_plot_id": plot_request["requested_plot_id"],
                    "last_chart_type": plot_request["requested_chart_type"],
                    "last_bar_mode": plot_request["requested_bar_mode"],
                    "last_plot_status": "missing_data" if missing_inputs else "ready",
                }
            )

        resolved_state = _merge_state(state, updates)
        return {
            "chat_history_last_25": chat_history_last_25 or [],
            "session_state": resolved_state,
            "resolved_context": {
                "active_universe": resolved_state.get("active_universe"),
                "active_tickers": resolved_state.get("active_tickers", []),
                "active_ticker_count": resolved_state.get("active_ticker_count", 0),
                "active_weights": resolved_state.get("active_weights"),
                "active_date_range": resolved_state.get("active_date_range"),
                "plot_request": plot_request,
                "policy": PROJECT_POLICY_MEMORY,
            },
            "intent_lock": intent_lock,
            "pending_action": resolved_state.get("pending_action"),
            "memory_update": updates,
            "validation_result": validation,
            "router_plan": plan,
            "pending_status": pending_status,
            "direct_response": direct_response,
        }


def build_missing_input_response(resolved: dict[str, Any]) -> str | None:
    state = resolved.get("session_state", {})
    pending = state.get("pending_action")
    if not pending:
        return None
    if pending.get("type") == "use_equal_weight_proxy":
        tickers = pending.get("target_tickers", [])
        return (
            "I have the universe context, but current weights are missing, so I will not render the plot yet.\n\n"
            f"- universe: `{pending.get('target_universe') or state.get('active_universe')}`\n"
            f"- ticker count: {len(tickers)}\n"
            f"- requested plot: `{pending.get('target_plot_id')}`\n\n"
            "Do you want me to use an equal-weight proxy for this plot? If you say `yes`, I will label it clearly as `equal_weight_proxy`."
        )
    return None


def build_pending_execution_response(resolved: dict[str, Any]) -> str | None:
    if resolved.get("pending_status") != "executed":
        return None
    state = resolved.get("session_state", {})
    fallback_result = resolved.get("fallback_result", {})
    plot_payload = fallback_result.get("plot_payload", {})
    plot_id = state.get("last_plot_id")
    tickers = state.get("active_tickers", [])
    weights = state.get("active_weights", {})
    computed_status = fallback_result.get("status", state.get("last_plot_status"))
    data_rows = len(plot_payload.get("data", [])) if isinstance(plot_payload.get("data"), list) else 0
    return (
        "Approved. I will use the equal-weight proxy and keep it labeled as fallback/proxy data.\n\n"
        f"- plot: `{plot_id}`\n"
        f"- universe: `{state.get('active_universe')}`\n"
        f"- ticker count: {len(tickers)}\n"
        f"- weight source: `{weights.get('type')}`\n"
        f"- fallback used: yes\n"
        f"- computed status: `{computed_status}`\n"
        f"- computed rows: {data_rows}\n"
        f"- confidence: {fallback_result.get('confidence', 'Medium')}\n"
        "- limitation: Actual user current holdings were not available, so this is a test proxy."
    )


def build_direct_context_response(resolved: dict[str, Any]) -> str | None:
    direct = resolved.get("direct_response") or resolved.get("session_state", {}).get("direct_response")
    if isinstance(direct, dict) and direct.get("text"):
        return str(direct["text"])

    fallback_result = resolved.get("fallback_result", {})
    payload = fallback_result.get("plot_payload") if isinstance(fallback_result.get("plot_payload"), dict) else {}
    if fallback_result.get("status") != "success" or not payload:
        return None
    if payload.get("plot_type") == "line":
        tickers = payload.get("tickers_used") or payload.get("tickers") or []
        date_range = payload.get("date_range") or {}
        limitations = payload.get("limitations") or []
        limitation_text = "\n".join(f"- {item}" for item in limitations) if limitations else "- none"
        return (
            f"{payload.get('title') or 'Line Plot'} generated.\n\n"
            f"plot_id: {payload.get('plot_short_id') or payload.get('plot_id')}\n"
            f"chart_type: {payload.get('chart_type')}\n"
            f"ticker_count: {len(tickers)}\n"
            f"tickers_used: {', '.join(tickers)}\n"
            f"date_range: {date_range.get('start')} to {date_range.get('end')}\n"
            f"x-axis field: {payload.get('x_axis')}\n"
            f"y-axis field: {payload.get('y_axis')}\n"
            f"unit: {payload.get('unit')}\n"
            f"data_source: {payload.get('data_source')}\n"
            f"fallback_used: {str(bool(payload.get('fallback_used'))).lower()}\n"
            f"fallback_method: {payload.get('fallback_method') or 'none'}\n"
            f"connect_nulls: {str(bool(payload.get('connect_nulls'))).lower()}\n"
            f"curve: {payload.get('curve')}\n"
            f"optimizer_called: {str(bool(payload.get('optimizer_called'))).lower()}\n"
            f"advisory_allocation_generated: {str(bool(payload.get('advisory_allocation_generated'))).lower()}\n\n"
            "Limitations:\n"
            f"{limitation_text}"
        )
    if payload.get("plot_type") == "pie":
        tickers = payload.get("tickers_used") or payload.get("tickers") or []
        limitations = payload.get("limitations") or []
        limitation_text = "\n".join(f"- {item}" for item in limitations) if limitations else "- none"
        fallback_text = ""
        if payload.get("fallback_rendered"):
            fallback_text = f"fallback_rendered: true\nfallback_chart: {payload.get('fallback_chart')}\n"
        return (
            f"{payload.get('title') or 'Composition Chart'} generated.\n\n"
            f"plot_id: {payload.get('plot_short_id') or payload.get('plot_id')}\n"
            f"chart_type: {payload.get('chart_type')}\n"
            f"ticker_count: {len(tickers)}\n"
            f"slice_count: {payload.get('slice_count')}\n"
            f"category_field: {payload.get('category_field')}\n"
            f"value_field: {payload.get('value_field')}\n"
            f"total_value: {payload.get('total_value')}\n"
            f"unit: {payload.get('unit')}\n"
            f"data_source: {payload.get('data_source')}\n"
            f"fallback_used: {str(bool(payload.get('fallback_used'))).lower()}\n"
            f"{fallback_text}"
            f"optimizer_called: {str(bool(payload.get('optimizer_called'))).lower()}\n"
            f"advisory_allocation_generated: {str(bool(payload.get('advisory_allocation_generated'))).lower()}\n\n"
            "Limitations:\n"
            f"{limitation_text}"
        )
    if payload.get("plot_type") == "scatter":
        tickers = payload.get("tickers_used") or payload.get("tickers") or []
        limitations = payload.get("limitations") or []
        limitation_text = "\n".join(f"- {item}" for item in limitations) if limitations else "- none"
        regression_text = ""
        if payload.get("regression_used"):
            regression_text = (
                f"regression_method: {payload.get('regression_method')}\n"
                f"r_squared: {payload.get('r_squared')}\n"
            )
        size_axis_text = f"size_axis: {payload.get('size_axis')}\n" if payload.get("size_axis") else ""
        return (
            f"{payload.get('title') or 'Scatter Plot'} generated.\n\n"
            f"plot_id: {payload.get('plot_short_id') or payload.get('plot_id')}\n"
            f"chart_type: {payload.get('chart_type')}\n"
            f"universe: {payload.get('universe')}\n"
            f"ticker_count: {len(tickers)}\n"
            f"point_count: {payload.get('point_count')}\n"
            f"x-axis field: {payload.get('x_axis')}\n"
            f"y-axis field: {payload.get('y_axis')}\n"
            f"point_id: {payload.get('point_id')}\n"
            f"color_axis: {payload.get('color_axis')}\n"
            f"{size_axis_text}"
            f"data_source: {payload.get('data_source')}\n"
            f"fallback_used: {str(bool(payload.get('fallback_used'))).lower()}\n"
            f"fallback_method: {payload.get('fallback_method') or 'none'}\n"
            f"{regression_text}"
            f"optimizer_called: {str(bool(payload.get('optimizer_called'))).lower()}\n"
            f"advisory_allocation_generated: {str(bool(payload.get('advisory_allocation_generated'))).lower()}\n\n"
            "Limitations:\n"
            f"{limitation_text}"
        )
    if payload.get("plot_id") != "plot_42_ticker_concentration_plot":
        return None

    tickers = payload.get("tickers_used") or payload.get("tickers") or []
    short_plot_id = payload.get("plot_short_id") or payload.get("plot_id")
    return (
        "Ticker Concentration Plot generated.\n\n"
        f"plot_id: {short_plot_id}\n"
        f"chart_type: {payload.get('chart_type')}\n"
        f"bar_mode: {payload.get('bar_mode')}\n"
        f"universe: {payload.get('universe')}\n"
        f"ticker_count: {payload.get('ticker_count')}\n"
        f"tickers_used: {', '.join(tickers)}\n"
        f"x-axis field: {payload.get('x_axis')}\n"
        f"y-axis field: {payload.get('y_axis')}\n"
        f"unit: {payload.get('unit')}\n"
        f"data_source: {payload.get('data_source')}\n"
        f"fallback_used: {str(bool(payload.get('fallback_used'))).lower()}\n"
        f"optimizer_called: {str(bool(payload.get('optimizer_called'))).lower()}\n\n"
        "Note: Equal-weight proxy was used because current portfolio weights were missing "
        "and the user explicitly permitted proxy use for this test."
    )


def record_plot_success(state: dict[str, Any], plot_id: str, chart_type: str, bar_mode: str | None = None) -> dict[str, Any]:
    tickers = state.get("active_tickers", [])
    next_state = dict(state)
    next_state["last_plot_id"] = plot_id
    next_state["last_chart_type"] = chart_type
    next_state["last_bar_mode"] = bar_mode
    next_state["last_plot_status"] = "success"
    next_state["last_plot"] = {
        "plot_id": plot_id,
        "chart_type": chart_type,
        "bar_mode": bar_mode,
        "universe": state.get("active_universe"),
        "ticker_count": len(tickers),
        "status": "success",
        "data_source": state.get("active_weights", {}).get("type"),
    }
    return next_state


def _extract_universe(message: str) -> str | None:
    match = re.search(r"\bU\d{1,2}\b", str(message or "").upper())
    return match.group(0) if match else None


def _resolve_tickers(universe: str | None, explicit_tickers: list[str], previous_tickers: list[str], explicit_subset: bool) -> list[str]:
    if explicit_subset:
        return [str(t).strip().upper() for t in explicit_tickers if str(t).strip()]
    if universe and universe in KNOWN_UNIVERSE_TICKERS:
        return list(KNOWN_UNIVERSE_TICKERS[universe])
    return [str(t).strip().upper() for t in previous_tickers if str(t).strip()]


def _requests_full_universe_scope(message: str, universe: str | None) -> bool:
    if not universe:
        return False
    normalized = " ".join(str(message or "").lower().split())
    universe_lower = universe.lower()
    return (
        f"all {universe_lower} ticker" in normalized
        or f"all {universe_lower} stock" in normalized
        or f"use all {universe_lower}" in normalized
        or f"for {universe_lower} tickers" in normalized
        or f"for {universe_lower} stocks" in normalized
    )


def _equal_weight_percentages(tickers: list[str]) -> dict[str, float]:
    cleaned = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not cleaned:
        return {}
    weight = round(100.0 / len(cleaned), 10)
    return {ticker: weight for ticker in cleaned}


def _filter_tickers(tickers: list[str]) -> list[str]:
    filtered = []
    for ticker in tickers:
        normalized = str(ticker or "").strip().upper()
        if not normalized or normalized in TICKER_STOPWORDS:
            continue
        filtered.append(normalized)
    return filtered


def _extract_plot_request(message: str, state: dict[str, Any]) -> dict[str, Any] | None:
    normalized = " ".join(str(message or "").lower().split())
    key = None
    if "ownership overlap" in normalized and "correlation" in normalized and _has_plot_marker(normalized):
        key = "ownership_overlap_correlation_scatter"
    elif "beta" in normalized and "return" in normalized and _has_plot_marker(normalized):
        key = "beta_return_scatter"
    elif "cvar" in normalized and "return" in normalized and _has_plot_marker(normalized):
        key = "cvar_return_scatter"
    elif "drawdown" in normalized and "return" in normalized and _has_plot_marker(normalized):
        key = "drawdown_return_scatter"
    elif "allocation" in normalized and "risk contribution" in normalized and "scatter" in normalized:
        key = "allocation_vs_risk_contribution_scatter"
    elif ("bubble" in normalized and "risk" in normalized and "return" in normalized) or "risk-return bubble" in normalized:
        key = "bubble_risk_return"
    elif "regression" in normalized and "scatter" in normalized:
        key = "scatter_with_regression_line"
    elif (
        "risk-return" in normalized
        or "risk return scatter" in normalized
        or "risk vs return" in normalized
        or "volatility vs return" in normalized
        or "volatility-return" in normalized
    ) and _has_plot_marker(normalized):
        key = "risk_return_scatter"
    elif ("adjusted close" in normalized or "close price" in normalized or "price trend" in normalized) and _has_plot_marker(normalized):
        if _looks_like_multi_ticker_comparison(normalized, message) or _requests_multi_ticker_line(normalized, state):
            key = "normalized_price_comparison"
        else:
            key = "historical_adjusted_close"
    elif ("normalized price" in normalized or "normalised price" in normalized or "normalized comparison" in normalized) and _has_plot_marker(normalized):
        key = "normalized_price_comparison"
    elif normalized.startswith("compare ") and len(_filter_tickers(re.findall(r"\b[A-Z]{1,5}\b", str(message or "")))) >= 2:
        key = "normalized_price_comparison"
    elif "portfolio value" in normalized and ("over time" in normalized or _has_plot_marker(normalized)):
        key = "portfolio_value_over_time"
    elif "drawdown" in normalized and "return" not in normalized and ("over time" in normalized or _has_plot_marker(normalized)):
        key = "drawdown_over_time"
    elif "rolling volatility" in normalized or "volatility over time" in normalized or "volatility spike" in normalized:
        key = "rolling_volatility_over_time"
    elif "rolling correlation" in normalized or "correlation over time" in normalized:
        key = "rolling_correlation_over_time"
    elif ("instability index" in normalized or ("calm" in normalized and "crisis" in normalized) or "regime" in normalized) and _has_plot_marker(normalized):
        key = "instability_index_over_time"
    elif ("cvar over time" in normalized or "rolling cvar" in normalized) and _has_plot_marker(normalized):
        key = "cvar_over_time"
    elif "governance metric" in normalized and ("threshold" in normalized or "over time" in normalized):
        key = "governance_metric_threshold_over_time"
    elif "nested donut" in normalized or (("sector" in normalized and "ticker" in normalized) and "donut" in normalized):
        key = "sector_ticker_nested_donut"
    elif ("portfolio health" in normalized or "health donut" in normalized) and ("donut" in normalized or "pie" in normalized):
        key = "portfolio_health_donut"
    elif ("risk gauge" in normalized or "semi donut" in normalized or "semi-donut" in normalized) and ("gauge" in normalized or "donut" in normalized):
        key = "semi_donut_risk_gauge"
    elif ("sector allocation" in normalized or "sector weight" in normalized or "sector weights" in normalized) and ("donut" in normalized or "pie" in normalized):
        key = "sector_allocation_donut"
    elif ("ticker allocation" in normalized or "ticker weight" in normalized or "ticker weights" in normalized or "allocation by ticker" in normalized) and ("donut" in normalized or "pie" in normalized):
        key = "ticker_allocation_donut"
    elif "risk contribution" in normalized and ("donut" in normalized or "pie" in normalized or "share" in normalized):
        key = "risk_contribution_donut"
    elif "ownership overlap" in normalized and "correlation" in normalized and _has_plot_marker(normalized):
        key = "ownership_overlap_correlation_scatter"
    elif "beta" in normalized and "return" in normalized and _has_plot_marker(normalized):
        key = "beta_return_scatter"
    elif "cvar" in normalized and "return" in normalized and _has_plot_marker(normalized):
        key = "cvar_return_scatter"
    elif "drawdown" in normalized and "return" in normalized and _has_plot_marker(normalized):
        key = "drawdown_return_scatter"
    elif "allocation" in normalized and "risk contribution" in normalized and "scatter" in normalized:
        key = "allocation_vs_risk_contribution_scatter"
    elif ("bubble" in normalized and "risk" in normalized and "return" in normalized) or "risk-return bubble" in normalized:
        key = "bubble_risk_return"
    elif "regression" in normalized and "scatter" in normalized:
        key = "scatter_with_regression_line"
    elif (
        "risk-return" in normalized
        or "risk return scatter" in normalized
        or "risk vs return" in normalized
        or "volatility vs return" in normalized
        or "volatility-return" in normalized
    ) and _has_plot_marker(normalized):
        key = "risk_return_scatter"
    elif "ticker concentration" in normalized:
        key = "ticker_concentration"
    elif "sector concentration" in normalized:
        key = "sector_concentration"
    elif "risk contribution waterfall" in normalized:
        key = "risk_contribution_waterfall"
    elif "portfolio return waterfall" in normalized or "return contribution waterfall" in normalized:
        key = "portfolio_return_waterfall"
    elif "return range" in normalized and "ticker" in normalized:
        key = "return_range_by_ticker"
    elif "volatility range" in normalized and "ticker" in normalized:
        key = "volatility_range_by_ticker"
    elif "mirrored" in normalized and ("current" in normalized or "advisory" in normalized):
        key = "current_vs_advisory_mirrored_bar"
    elif "return distribution histogram" in normalized or ("histogram" in normalized and "return" in normalized):
        key = "return_distribution_histogram"
    elif "risk contribution" in normalized:
        key = "risk_contribution"
    elif "allocation vs risk" in normalized:
        key = "allocation_vs_risk_contribution"
    elif "current vs advisory" in normalized:
        key = "current_vs_advisory"
    elif "allocation change" in normalized:
        key = "allocation_change"
    elif "effective number" in normalized or "effective holdings" in normalized:
        key = "effective_holdings"
    elif "hhi" in normalized or "concentration index" in normalized:
        key = "hhi"
    elif "eigenvector centrality" in normalized or "centrality by ticker" in normalized:
        key = "eigenvector_centrality"
    elif "strategy comparison" in normalized or "strategy performance" in normalized:
        key = "strategy_comparison"
    elif normalized in {"plot it", "use them", "same", "do it"} and state.get("last_plot_id"):
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
    if not key:
        return None
    plot_id, chart_type, bar_mode = PLOT_ALIASES[key]
    return {
        "requested_plot_id": plot_id,
        "requested_chart_type": chart_type,
        "requested_bar_mode": bar_mode,
        "requested_universe": state.get("active_universe"),
        "requested_tickers": state.get("active_tickers", []),
        "required_fields": [],
        "data_source": state.get("active_weights", {}).get("type"),
        "status": "requested",
    }


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


def _apply_negative_optimizer_instructions(
    plan: dict[str, Any],
    message: str,
    plot_request: dict[str, Any] | None,
) -> None:
    normalized = " ".join(str(message or "").lower().split())
    forbids_optimizer = any(
        phrase in normalized
        for phrase in (
            "do not run optimizer",
            "don't run optimizer",
            "no optimizer",
            "without optimizer",
            "do not generate allocation",
            "don't generate allocation",
            "do not run optimisation",
            "do not run optimization",
        )
    )
    ticker_concentration_only = (
        plot_request is not None
        and plot_request.get("requested_plot_id") == "plot_42_ticker_concentration_plot"
        and any(phrase in normalized for phrase in ("generate only", "only the ticker concentration", "only ticker concentration"))
    )
    line_plot_only = plot_request is not None and plot_request.get("requested_chart_type") in {"line", "line_area", "stacked_area", "dual_axis_line"}
    pie_plot_only = plot_request is not None and plot_request.get("requested_chart_type") in {"pie", "donut", "center_label_donut", "nested_donut", "semi_donut"}
    scatter_plot_only = plot_request is not None and plot_request.get("requested_chart_type") in {"scatter", "bubble_scatter", "scatter_regression", "webgl_scatter"}
    if not (forbids_optimizer or ticker_concentration_only or line_plot_only or pie_plot_only or scatter_plot_only):
        return

    execution = plan.setdefault("execution", {})
    execution["needs_optimizer"] = False
    execution["needs_allocation"] = False
    execution["needs_graph_network"] = False
    safety = plan.setdefault("safety", {})
    skipped = list(safety.get("modules_skipped", []))
    for module in ("optimizer", "advisory_allocation", "graph_network"):
        if module not in skipped:
            skipped.append(module)
    safety["modules_skipped"] = skipped


def _proxy_approved_in_message(message: str) -> bool:
    normalized = " ".join(str(message or "").lower().replace("_", " ").split())
    if not normalized:
        return False
    phrases = (
        "if current weights are missing, use equal-weight proxy",
        "if current weights are missing use equal-weight proxy",
        "if current weights are missing, use equal weight proxy",
        "if current weights are missing use equal weight proxy",
        "use equal-weight current allocation proxy",
        "use equal weight current allocation proxy",
        "use equal weight for this test",
        "use equal-weight for this test",
        "use proxy for this test",
        "use equal-weight proxy for this test",
        "use equal weight proxy for this test",
    )
    return any(phrase in normalized for phrase in phrases)


def _build_direct_response(
    message: str,
    universe: str | None,
    tickers: list[str],
    state: dict[str, Any],
) -> dict[str, Any] | None:
    if _is_ticker_listing_intent(message, universe):
        if universe in KNOWN_UNIVERSE_TICKERS:
            registry_tickers = list(KNOWN_UNIVERSE_TICKERS[universe])
            return {
                "type": "universe_ticker_list",
                "text": (
                    f"universe: {universe}\n"
                    f"ticker_count: {len(registry_tickers)}\n"
                    f"ticker_list: {', '.join(registry_tickers)}\n"
                    "source: universe_registry"
                ),
            }
        if tickers:
            return {
                "type": "universe_ticker_list",
                "text": (
                    f"universe: {universe or state.get('active_universe') or 'unknown'}\n"
                    f"ticker_count: {len(tickers)}\n"
                    f"ticker_list: {', '.join(tickers)}\n"
                    "source: active_session_context"
                ),
            }

    if _is_real_values_intent(message):
        return {
            "type": "real_current_weights_required",
            "text": (
                "I cannot calculate real current allocation weights from price data alone.\n"
                "To calculate real ticker concentration, I need your holdings: shares per ticker, "
                "invested amount per ticker, or stored current portfolio weights.\n"
                "I can either:\n"
                "1. list U1 tickers,\n"
                "2. use equal-weight proxy for testing,\n"
                "3. use latest advisory weights, clearly labeled as advisory, not current."
            ),
        }
    return None


def _is_ticker_listing_intent(message: str, universe: str | None) -> bool:
    normalized = " ".join(str(message or "").lower().split())
    if not universe:
        return False
    if any(
        marker in normalized
        for marker in (
            "ticker concentration",
            "sector concentration",
            "risk contribution",
            "allocation",
            "plot",
            "chart",
            "graph",
            "bar",
        )
    ):
        return False
    has_listing_verb = any(verb in normalized for verb in ("list", "fetch", "show", "get"))
    has_ticker_noun = any(noun in normalized for noun in ("ticker", "tickers", "stocks", "constituents", "members"))
    return has_listing_verb and has_ticker_noun


def _is_real_values_intent(message: str) -> bool:
    normalized = " ".join(str(message or "").lower().split())
    return any(
        phrase in normalized
        for phrase in (
            "calculate real values",
            "calculate the real values",
            "compute real values",
            "compute the real values",
            "real current weights",
            "actual current weights",
        )
    )


def _has_plot_marker(normalized: str) -> bool:
    return any(marker in normalized for marker in ("plot", "chart", "show", "generate", "display", "visualize", "scatter", "bubble", "regression", "line"))


def _looks_like_multi_ticker_comparison(normalized: str, message: str) -> bool:
    if "compare" in normalized or "comparison" in normalized or "multi" in normalized:
        return True
    return len(_filter_tickers(re.findall(r"\b[A-Z]{1,5}\b", str(message or "")))) > 1


def _requests_multi_ticker_line(normalized: str, state: dict[str, Any]) -> bool:
    if "ticker" in normalized or "tickers" in normalized or "trends" in normalized:
        return True
    active = state.get("active_tickers") or []
    return len(active) > 1 and any(token in normalized for token in ("same", "them", "it"))


def _has_current_weights(state: dict[str, Any]) -> bool:
    weights = state.get("active_weights", {})
    return bool(weights.get("weights")) and weights.get("type") != "missing"


def _previous_analysis_from_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "analysis_id": state.get("active_analysis_id"),
        "entities": {
            "universe": state.get("active_universe"),
            "tickers": state.get("active_tickers", []),
            "current_weights": state.get("active_weights", {}).get("weights", {}),
            "start_date": state.get("active_date_range", {}).get("start"),
            "end_date": state.get("active_date_range", {}).get("end"),
        },
    }


def _modules_called(plan: dict[str, Any]) -> list[str]:
    execution = plan.get("execution", {})
    modules = ["intent_router"]
    if execution.get("needs_math"):
        modules.append("quantitative_analytics_agent")
    if execution.get("needs_optimizer"):
        modules.append("optimizer")
    if execution.get("needs_rag"):
        modules.append("rag")
    if execution.get("needs_plot"):
        modules.append("smart_plot_selector")
    if execution.get("needs_validator"):
        modules.append("validator")
    return modules


def _merge_state(state: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(state)
    for key, value in updates.items():
        merged[key] = value
    return merged
