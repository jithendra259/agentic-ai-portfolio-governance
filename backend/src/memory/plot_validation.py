from __future__ import annotations

from typing import Any
from datetime import datetime

from src.memory.chart_registry import get_chart_definition, premium_charts_enabled, premium_unavailable_message
from src.memory.session_state import KNOWN_UNIVERSE_TICKERS


PIE_CHART_TYPES = {"pie", "donut", "center_label_donut", "nested_donut", "semi_donut"}
SCATTER_CHART_TYPES = {"scatter", "bubble_scatter", "scatter_regression", "webgl_scatter"}
PERCENT_SUM_TOLERANCE = 0.5


def validate_plot_payload(request: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if not request:
        return {"can_render": True, "errors": [], "warnings": []}

    if request.get("requested_plot_id") and payload.get("plot_id") != request.get("requested_plot_id"):
        errors.append("requested_plot_id does not match returned plot_id")
    fallback_rendered = bool(payload.get("fallback_rendered"))
    if request.get("requested_chart_type") and payload.get("chart_type") != request.get("requested_chart_type") and not fallback_rendered:
        errors.append("requested_chart_type does not match returned chart_type")
    if (
        request.get("requested_bar_mode")
        and (payload.get("chart_type") == "bar" or request.get("requested_chart_type") == "bar")
        and payload.get("bar_mode") != request.get("requested_bar_mode")
    ):
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

    definition = get_chart_definition(payload.get("plot_id") or request.get("requested_plot_id")) or {}
    required_fields = request.get("required_fields") or payload.get("required_fields") or definition.get("required_fields") or []
    for field in required_fields:
        if not _field_exists(payload.get("data"), field):
            errors.append(f"required field missing: {field}")

    chart_type = payload.get("chart_type") or definition.get("chart_type")
    if chart_type == "rangeBar":
        _validate_range_bar_payload(payload, definition, errors)
    if chart_type == "mirroredBar":
        for field in ("current_weight", "advisory_weight"):
            if not _field_exists(payload.get("data"), field):
                errors.append(f"required field missing: {field}")
    if chart_type == "histogram":
        for field in ("bin_start", "bin_end", "count"):
            if not _field_exists(payload.get("data"), field):
                errors.append(f"required field missing: {field}")
    if chart_type in {"line", "line_area", "stacked_area", "dual_axis_line"} or payload.get("plot_type") == "line":
        _validate_line_payload(request, payload, definition, errors)
    if chart_type in PIE_CHART_TYPES or payload.get("plot_type") == "pie":
        _validate_pie_payload(request, payload, definition, errors)
    if chart_type in SCATTER_CHART_TYPES or payload.get("plot_type") == "scatter":
        _validate_scatter_payload(request, payload, definition, errors)

    requires_premium = bool(payload.get("requires_premium", definition.get("requires_premium", False)))
    fallback_chart = payload.get("fallback_chart") or definition.get("fallback_chart")
    premium_enabled = bool(payload.get("premium_enabled", premium_charts_enabled()))
    premium_unavailable = requires_premium and not premium_enabled
    if premium_unavailable:
        message = premium_unavailable_message(fallback_chart)
        if fallback_chart:
            warnings.append(message)
        else:
            errors.append(message)

    if not payload.get("data_source"):
        errors.append("data source is missing")
    if payload.get("proxy_used") and not payload.get("proxy_declared"):
        errors.append("proxy/default use is not declared")
    if payload.get("fallback_used") and not payload.get("data_source"):
        errors.append("fallback/proxy data source is not declared")
    if payload.get("status") in {"missing_data", "unavailable", "failed", "failed_validation"}:
        errors.append(payload.get("reason") or "plot data unavailable")
    if payload.get("optimizer_called") and _optimizer_blocked_for_plot(payload):
        errors.append("optimizer was called for a plot-only chart")
    if payload.get("advisory_allocation_generated") and _optimizer_blocked_for_plot(payload):
        errors.append("advisory allocation was generated for a plot-only chart")

    status = "pass" if not errors else "blocked"
    if not errors and premium_unavailable:
        status = "premium_unavailable"

    return {
        "can_render": not errors,
        "render": not errors,
        "status": status,
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


def _validate_range_bar_payload(payload: dict[str, Any], definition: dict[str, Any], errors: list[str]) -> None:
    start_field = payload.get("range_start_field") or payload.get("rangeStartField") or definition.get("range_start_field")
    end_field = payload.get("range_end_field") or payload.get("rangeEndField") or definition.get("range_end_field")
    if not start_field:
        start_field = "start_value" if _field_exists(payload.get("data"), "start_value") else "start"
    if not end_field:
        end_field = "end_value" if _field_exists(payload.get("data"), "end_value") else "end"
    for field in (start_field, end_field):
        if not _field_exists(payload.get("data"), field):
            errors.append(f"required field missing: {field}")


def _validate_line_payload(
    request: dict[str, Any],
    payload: dict[str, Any],
    definition: dict[str, Any],
    errors: list[str],
) -> None:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        errors.append("line chart data must contain at least one row")
        return
    if not _field_exists(data, "date"):
        errors.append("required field missing: date")
        return

    dates = [_parse_date(row.get("date")) for row in data if isinstance(row, dict)]
    if any(date is None for date in dates):
        errors.append("line chart dates must parse as dates")
    parsed = [date for date in dates if date is not None]
    if parsed and parsed != sorted(parsed):
        errors.append("line chart dates must be sorted ascending")

    y_field = payload.get("y_axis") or (definition.get("required_fields") or ["", "value"])[-1]
    if not y_field or y_field == "date":
        y_field = "value"
    if not _field_exists(data, y_field):
        errors.append(f"required field missing: {y_field}")
    else:
        numeric_values = [row.get(y_field) for row in data if isinstance(row, dict)]
        if not any(_is_finite_or_null_allowed(value) and value is not None for value in numeric_values):
            errors.append("line chart y-values are all null or non-numeric")

    tickers = payload.get("tickers_used") or payload.get("tickers") or request.get("requested_tickers") or []
    if not tickers:
        errors.append("ticker list is empty")

    if payload.get("plot_id") == "historical_adjusted_close" and len(tickers) > 1:
        errors.append("raw multi-ticker price comparison must use normalized_price_comparison")

    if (
        payload.get("plot_id") == "normalized_price_comparison"
        and len(tickers) > 12
        and not payload.get("fallback_method") == "normalize_adjusted_close_to_100"
    ):
        errors.append("multi-ticker price line must declare normalization method")

    if payload.get("plot_id") == "portfolio_value_over_time" and not payload.get("initial_capital"):
        errors.append("portfolio value plot requires initial capital")

    if payload.get("benchmark_requested") and not payload.get("benchmark_available"):
        errors.append("benchmark requested but benchmark series is unavailable")


def _validate_pie_payload(
    request: dict[str, Any],
    payload: dict[str, Any],
    definition: dict[str, Any],
    errors: list[str],
) -> None:
    if payload.get("fallback_rendered") and payload.get("plot_type") != "pie":
        return

    data = payload.get("data")
    if not isinstance(data, list) or not data:
        errors.append("pie chart data must contain at least one slice")
        return

    chart_type = payload.get("chart_type") or definition.get("chart_type")
    category_field = payload.get("category_field") or definition.get("category_field") or payload.get("x_axis")
    value_field = payload.get("value_field") or definition.get("value_field") or payload.get("y_axis")
    if not category_field:
        errors.append("pie chart category_field is missing")
    if not value_field:
        errors.append("pie chart value_field is missing")
    if category_field and not _field_exists(data, category_field):
        errors.append(f"required field missing: {category_field}")
    if value_field and not _field_exists(data, value_field):
        errors.append(f"required field missing: {value_field}")

    values = []
    for row in data:
        if not isinstance(row, dict):
            errors.append("pie chart rows must be objects")
            continue
        if value_field and row.get(value_field) is not None:
            try:
                value = float(row.get(value_field))
            except (TypeError, ValueError):
                errors.append(f"pie value must be numeric: {value_field}")
                continue
            if not _is_finite_or_null_allowed(value):
                errors.append(f"pie value must be finite: {value_field}")
            if value < 0:
                errors.append("pie/donut values cannot be negative")
            values.append(value)

    total_value = float(payload.get("total_value") if payload.get("total_value") is not None else sum(values))
    if total_value <= 0:
        errors.append("pie/donut total_value must be greater than zero")

    slice_count = int(payload.get("slice_count") or len(data))
    max_slices = definition.get("max_slices")
    if max_slices and slice_count > int(max_slices) and not payload.get("explicit_large_pie"):
        errors.append(f"pie/donut slice_count exceeds max_slices {max_slices}")

    if chart_type == "nested_donut":
        if not _field_exists(data, "sector") or not _field_exists(data, "ticker"):
            errors.append("nested donut requires sector and ticker parent-child fields")
        if not _field_exists(data, "sector_weight_percent") or not _field_exists(data, "ticker_weight_percent"):
            errors.append("nested donut requires sector and ticker weight fields")
        if not payload.get("series") or len(payload.get("series", [])) < 2:
            errors.append("nested donut requires at least two series rings")
        max_outer_slices = definition.get("max_outer_slices")
        if max_outer_slices and len(data) > int(max_outer_slices):
            errors.append(f"nested donut outer slice count exceeds max_outer_slices {max_outer_slices}")

    if payload.get("plot_id") in {"sector_allocation_donut", "sector_ticker_nested_donut", "portfolio_health_donut"}:
        if any(not row.get("sector") for row in data if isinstance(row, dict) and payload.get("plot_id") != "portfolio_health_donut"):
            errors.append("sector donut requires sector mapping for every ticker")
        if payload.get("plot_id") == "portfolio_health_donut" and not payload.get("metrics", {}).get("sector_hhi"):
            errors.append("portfolio health donut requires sector concentration metrics")

    if payload.get("time_series_requested") or (category_field and category_field == "date"):
        errors.append("pie/donut charts are blocked for time-series data")

    if _is_percent_pie(payload, value_field):
        total = sum(values)
        if abs(total - 100.0) > PERCENT_SUM_TOLERANCE:
            errors.append(f"pie/donut percentage values must sum close to 100%; got {total:.4f}")


def _validate_scatter_payload(
    request: dict[str, Any],
    payload: dict[str, Any],
    definition: dict[str, Any],
    errors: list[str],
) -> None:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        errors.append("scatter chart data must contain at least two points")
        return

    chart_type = payload.get("chart_type") or definition.get("chart_type") or "scatter"
    x_field = payload.get("x_axis") or definition.get("x_axis")
    y_field = payload.get("y_axis") or definition.get("y_axis")
    point_id = payload.get("point_id") or definition.get("point_id") or "id"
    color_axis = payload.get("color_axis") or definition.get("color_axis")
    size_axis = payload.get("size_axis") or definition.get("size_axis")

    if not x_field:
        errors.append("scatter x_axis is missing")
    if not y_field:
        errors.append("scatter y_axis is missing")
    if not point_id:
        errors.append("scatter point_id is missing")
    if x_field and not _field_exists(data, x_field):
        errors.append(f"required field missing: {x_field}")
    if y_field and not _field_exists(data, y_field):
        errors.append(f"required field missing: {y_field}")
    if point_id and not _field_exists(data, point_id):
        errors.append(f"required field missing: {point_id}")
    if color_axis and not _field_exists(data, color_axis):
        errors.append(f"required field missing: {color_axis}")
    if size_axis and not _field_exists(data, size_axis):
        errors.append(f"required field missing: {size_axis}")

    valid_points = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            errors.append("scatter rows must be objects")
            continue
        x_value = row.get(x_field) if x_field else None
        y_value = row.get(y_field) if y_field else None
        if _is_finite_or_null_allowed(x_value) and _is_finite_or_null_allowed(y_value) and x_value is not None and y_value is not None:
            valid_points.append(row)
        elif x_value is not None or y_value is not None:
            errors.append(f"scatter point {index} has non-finite x/y values")
        if size_axis and row.get(size_axis) is not None:
            try:
                size_value = float(row.get(size_axis))
            except (TypeError, ValueError):
                errors.append(f"scatter size value must be numeric: {size_axis}")
                continue
            if not _is_finite_or_null_allowed(size_value) or size_value < 0:
                errors.append("scatter bubble size values must be finite and non-negative")

    if len(valid_points) < 2:
        errors.append("scatter plot requires at least two valid points")
    if chart_type == "scatter_regression" and len(valid_points) < 3:
        errors.append("regression scatter requires at least three valid points")
    if chart_type == "scatter_regression" and payload.get("regression_used") and not payload.get("regression_line"):
        errors.append("regression scatter requires regression_line metadata")
    if chart_type == "bubble_scatter" and not size_axis:
        errors.append("bubble scatter requires size_axis")
    if chart_type == "bubble_scatter" and size_axis:
        positive_size = False
        for row in valid_points:
            try:
                positive_size = positive_size or float(row.get(size_axis) or 0.0) > 0
            except (TypeError, ValueError):
                continue
        if not positive_size:
            errors.append("bubble scatter requires at least one positive bubble size")
    if payload.get("plot_id") == "ownership_overlap_correlation_scatter" and not payload.get("graph_data_available"):
        errors.append("ownership-overlap scatter requires institutional graph data")
    if payload.get("plot_id") == "beta_return_scatter" and not payload.get("benchmark_available"):
        errors.append("beta-return scatter requires benchmark series")
    if payload.get("time_series_requested"):
        errors.append("scatter charts require two numeric relationship axes, not a time-series axis")
    if payload.get("point_count") is not None and int(payload.get("point_count") or 0) != len(valid_points):
        errors.append("scatter point_count must equal the number of valid points")
    if not isinstance(payload.get("series"), list) or not payload.get("series"):
        errors.append("scatter chart requires at least one series")


def _parse_date(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _is_finite_or_null_allowed(value: Any) -> bool:
    if value is None:
        return True
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return numeric == numeric and numeric not in {float("inf"), float("-inf")}


def _optimizer_blocked_for_plot(payload: dict[str, Any]) -> bool:
    return payload.get("plot_type") in {"line", "pie", "scatter"} or payload.get("chart_type") in {
        "line",
        "line_area",
        "stacked_area",
        "dual_axis_line",
        *PIE_CHART_TYPES,
        *SCATTER_CHART_TYPES,
    }


def _is_percent_pie(payload: dict[str, Any], value_field: str | None) -> bool:
    if payload.get("unit") in {"%", "percent"}:
        return True
    normalized_field = str(value_field or "").lower()
    if normalized_field.endswith("_percent"):
        return True
    return str(payload.get("plot_id") or "").lower() in {
        "sector_allocation_donut",
        "ticker_allocation_donut",
        "risk_contribution_donut",
        "sector_ticker_nested_donut",
    }


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
