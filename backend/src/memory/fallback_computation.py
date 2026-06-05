from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from api.analytics_router import extract_prices_from_db, generate_gbm_prices, get_portfolio_prices
from src.decision.concentration_metrics import compute_concentration_metrics
from src.memory.session_state import KNOWN_UNIVERSE_TICKERS


DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2024-12-31"
DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_ROLLING_WINDOW = 30

U1_SECTOR_MAP = {ticker: "Technology" for ticker in KNOWN_UNIVERSE_TICKERS.get("U1", [])}
DEFAULT_SECTOR_MAP = {
    **U1_SECTOR_MAP,
    "AMZN": "Consumer Discretionary",
    "JPM": "Financials",
    "XOM": "Energy",
    "UNH": "Health Care",
    "LLY": "Health Care",
    "V": "Financials",
    "MA": "Financials",
    "COST": "Consumer Staples",
    "HD": "Consumer Discretionary",
    "PG": "Consumer Staples",
    "NFLX": "Communication Services",
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "BAC": "Financials",
}


def compute_equal_weights(tickers: list[str], percent: bool = True) -> dict[str, float]:
    clean = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not clean:
        return {}
    value = (100.0 if percent else 1.0) / len(clean)
    return {ticker: value for ticker in clean}


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {str(t).upper(): float(w) for t, w in (weights or {}).items() if str(t).strip()}
    total = sum(clean.values())
    if total <= 0:
        return {}
    if total > 1.5:
        return {ticker: weight / total for ticker, weight in clean.items()}
    return clean


def compute_sector_weights(ticker_weights: dict[str, float], sector_map: dict[str, str] | None = None) -> dict[str, float]:
    sectors = sector_map or DEFAULT_SECTOR_MAP
    normalized = normalize_weights(ticker_weights)
    out: dict[str, float] = {}
    for ticker, weight in normalized.items():
        sector = sectors.get(ticker, "Other")
        out[sector] = out.get(sector, 0.0) + weight
    return out


def missing_sector_mappings(tickers: list[str], sector_map: dict[str, str] | None = None) -> list[str]:
    sectors = sector_map or DEFAULT_SECTOR_MAP
    return [str(t).upper() for t in tickers if str(t).upper() not in sectors]


def _slice_colors(count: int) -> list[str]:
    palette = ["#4f63f6", "#ffc857", "#f25467", "#38bdf8", "#4cc98a", "#e879b9", "#fb923c", "#818cf8"]
    return [palette[index % len(palette)] for index in range(count)]


def _limit_percent_slices(
    rows: list[dict[str, Any]],
    *,
    value_field: str,
    label_field: str,
    max_slices: int,
) -> list[dict[str, Any]]:
    if len(rows) <= max_slices:
        return rows
    head = rows[: max_slices - 1]
    tail = rows[max_slices - 1:]
    other_value = float(sum(float(row.get(value_field, 0.0)) for row in tail))
    if other_value > 0:
        head.append({label_field: "Other", value_field: other_value})
    return head


def compute_sector_allocation_donut(
    ticker_weights: dict[str, float],
    sector_map: dict[str, str] | None = None,
    max_slices: int = 8,
) -> dict[str, Any]:
    normalized = normalize_weights(ticker_weights)
    if not normalized:
        return {"status": "missing_data", "reason": "current weights are required", "missing_inputs": ["current_weights"]}
    missing = missing_sector_mappings(list(normalized), sector_map)
    if missing:
        return {
            "status": "unavailable",
            "reason": "sector mapping is missing for one or more tickers",
            "missing_inputs": ["sector_mapping"],
            "missing_tickers": missing,
        }
    sector_weights = compute_sector_weights(normalized, sector_map or DEFAULT_SECTOR_MAP)
    rows = [
        {"sector": sector, "weight_percent": float(weight * 100.0)}
        for sector, weight in sorted(sector_weights.items(), key=lambda item: item[1], reverse=True)
    ]
    rows = _limit_percent_slices(rows, value_field="weight_percent", label_field="sector", max_slices=max_slices)
    colors = _slice_colors(len(rows))
    for index, row in enumerate(rows):
        row["id"] = row["sector"]
        row["label"] = row["sector"]
        row["value"] = row["weight_percent"]
        row["color"] = colors[index]
    top_sector = rows[0]["sector"] if rows else None
    top_value = rows[0]["weight_percent"] if rows else 0.0
    metrics = compute_hhi_bundle(normalized, sector_map or DEFAULT_SECTOR_MAP)
    center_label = f"Top sector\n{top_sector}: {top_value:.1f}%\nEff sectors: {metrics.get('sector_effective_sectors', 0):.2f}"
    return {
        "status": "success",
        "data": rows,
        "series": [
            {
                "name": "sector_weight_percent",
                "label": "Sector allocation",
                "data": [{"id": row["sector"], "label": row["sector"], "value": row["weight_percent"], "color": row["color"]} for row in rows],
                "innerRadius": 58,
                "outerRadius": 112,
                "arcLabel": "label-percent",
                "arcLabelMinAngle": 18,
                "highlightScope": {"fade": "global", "highlight": "item"},
                "highlighted": {"additionalRadius": 4},
            }
        ],
        "total_value": float(sum(row["weight_percent"] for row in rows)),
        "slice_count": len(rows),
        "center_label": center_label,
        "metrics": metrics,
        "data_source": "sector_weights_from_ticker_weights",
        "fallback_used": False,
        "fallback_method": "compute_sector_weights_from_ticker_weights",
        "limitations": ["Sector allocation is computed from resolved ticker weights and sector mapping."],
        "confidence": "High",
    }


def compute_ticker_allocation_donut(
    ticker_weights: dict[str, float],
    max_slices: int = 8,
) -> dict[str, Any]:
    normalized = normalize_weights(ticker_weights)
    if not normalized:
        return {"status": "missing_data", "reason": "current weights are required", "missing_inputs": ["current_weights"]}
    rows = [
        {"ticker": ticker, "weight_percent": float(weight * 100.0)}
        for ticker, weight in sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    ]
    colors = _slice_colors(len(rows))
    for index, row in enumerate(rows):
        row["id"] = row["ticker"]
        row["label"] = row["ticker"]
        row["value"] = row["weight_percent"]
        row["color"] = colors[index]
    return {
        "status": "success",
        "data": rows,
        "series": [
            {
                "name": "ticker_weight_percent",
                "label": "Ticker allocation",
                "data": [{"id": row["ticker"], "label": row["ticker"], "value": row["weight_percent"], "color": row["color"]} for row in rows],
                "innerRadius": 58,
                "outerRadius": 112,
                "arcLabel": "label-percent",
                "arcLabelMinAngle": 20,
                "highlightScope": {"fade": "global", "highlight": "item"},
                "highlighted": {"additionalRadius": 4},
            }
        ],
        "total_value": float(sum(row["weight_percent"] for row in rows)),
        "slice_count": len(rows),
        "center_label": f"Tickers\n{len(rows)} slices",
        "data_source": "ticker_weights",
        "fallback_used": False,
        "fallback_method": None,
        "max_slices": max_slices,
        "limitations": ["Ticker allocation donut is intended only for small portfolios."],
        "confidence": "High",
    }


def compute_sector_ticker_nested_donut(
    ticker_weights: dict[str, float],
    sector_map: dict[str, str] | None = None,
    max_inner_slices: int = 8,
    max_outer_slices: int = 25,
) -> dict[str, Any]:
    normalized = normalize_weights(ticker_weights)
    if not normalized:
        return {"status": "missing_data", "reason": "current weights are required", "missing_inputs": ["current_weights"]}
    missing = missing_sector_mappings(list(normalized), sector_map)
    if missing:
        return {
            "status": "unavailable",
            "reason": "sector mapping is required for a nested sector-ticker donut",
            "missing_inputs": ["sector_mapping"],
            "missing_tickers": missing,
        }
    if len(normalized) > max_outer_slices:
        return {
            "status": "unavailable",
            "reason": f"nested donut supports at most {max_outer_slices} outer slices",
            "missing_inputs": ["smaller_ticker_set"],
        }
    sectors = sector_map or DEFAULT_SECTOR_MAP
    sector_weights = compute_sector_weights(normalized, sectors)
    inner_rows = [
        {"sector": sector, "sector_weight_percent": float(weight * 100.0)}
        for sector, weight in sorted(sector_weights.items(), key=lambda item: item[1], reverse=True)
    ]
    if len(inner_rows) > max_inner_slices:
        inner_rows = _limit_percent_slices(inner_rows, value_field="sector_weight_percent", label_field="sector", max_slices=max_inner_slices)
    sector_colors = {row["sector"]: color for row, color in zip(inner_rows, _slice_colors(len(inner_rows)), strict=False)}
    outer_rows = [
        {
            "sector": sectors[ticker],
            "sector_weight_percent": float(sector_weights[sectors[ticker]] * 100.0),
            "ticker": ticker,
            "ticker_weight_percent": float(weight * 100.0),
        }
        for ticker, weight in sorted(normalized.items(), key=lambda item: (sectors[item[0]], -item[1]))
    ]
    rows = outer_rows
    inner_data = [
        {
            "id": row["sector"],
            "label": row["sector"],
            "value": row["sector_weight_percent"],
            "color": sector_colors.get(row["sector"]),
        }
        for row in inner_rows
    ]
    outer_data = []
    for row in outer_rows:
        color = sector_colors.get(row["sector"])
        outer_data.append(
            {
                "id": row["ticker"],
                "label": row["ticker"],
                "value": row["ticker_weight_percent"],
                "color": color,
            }
        )
    metrics = compute_hhi_bundle(normalized, sectors)
    return {
        "status": "success",
        "data": rows,
        "series": [
            {
                "name": "sectors",
                "label": "Sectors",
                "data": inner_data,
                "innerRadius": 0,
                "outerRadius": 72,
                "arcLabel": "label-percent",
                "arcLabelMinAngle": 20,
                "highlightScope": {"fade": "global", "highlight": "item"},
            },
            {
                "name": "tickers",
                "label": "Tickers",
                "data": outer_data,
                "innerRadius": 88,
                "outerRadius": 116,
                "arcLabelMinAngle": 24,
                "highlightScope": {"fade": "global", "highlight": "item"},
                "faded": {"innerRadius": 88, "additionalRadius": -8, "color": "gray"},
            },
        ],
        "total_value": float(sum(row["ticker_weight_percent"] for row in outer_rows)),
        "slice_count": len(inner_rows) + len(outer_rows),
        "center_label": f"Sector -> ticker\nHHI {metrics.get('ticker_hhi', 0):.3f}",
        "metrics": metrics,
        "data_source": "ticker_weights_and_sector_mapping",
        "fallback_used": False,
        "fallback_method": "compute_sector_ticker_nested_donut",
        "limitations": ["Inner ring shows sector weights; outer ring shows ticker weights grouped by sector."],
        "confidence": "High",
    }


def compute_portfolio_health_donut(
    ticker_weights: dict[str, float],
    sector_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    normalized = normalize_weights(ticker_weights)
    if not normalized:
        return {"status": "missing_data", "reason": "current weights are required", "missing_inputs": ["current_weights"]}
    missing = missing_sector_mappings(list(normalized), sector_map)
    if missing:
        return {
            "status": "unavailable",
            "reason": "sector mapping is required for portfolio health donut",
            "missing_inputs": ["sector_mapping"],
            "missing_tickers": missing,
        }
    metrics = compute_hhi_bundle(normalized, sector_map or DEFAULT_SECTOR_MAP)
    ticker_score = max(0.0, min(100.0, (1.0 - float(metrics.get("ticker_hhi", 0.0))) * 100.0))
    sector_score = max(0.0, min(100.0, (1.0 - float(metrics.get("sector_hhi", 0.0))) * 100.0))
    balance_score = max(0.0, min(100.0, 100.0 - float(metrics.get("max_ticker_exposure", 0.0)) * 100.0))
    rows = [
        {"health_component": "Ticker spread", "score": ticker_score},
        {"health_component": "Sector spread", "score": sector_score},
        {"health_component": "Max exposure control", "score": balance_score},
    ]
    colors = _slice_colors(len(rows))
    for index, row in enumerate(rows):
        row["id"] = row["health_component"]
        row["label"] = row["health_component"]
        row["value"] = row["score"]
        row["color"] = colors[index]
    health_score = sum(row["score"] for row in rows) / len(rows)
    return {
        "status": "success",
        "data": rows,
        "series": [
            {
                "name": "portfolio_health",
                "label": "Portfolio health",
                "data": [{"id": row["health_component"], "label": row["health_component"], "value": row["score"], "color": row["color"]} for row in rows],
                "innerRadius": 62,
                "outerRadius": 112,
                "arcLabel": "percent",
                "arcLabelMinAngle": 18,
                "highlightScope": {"fade": "global", "highlight": "item"},
            }
        ],
        "total_value": float(sum(row["score"] for row in rows)),
        "slice_count": len(rows),
        "center_label": f"Health\n{health_score:.0f}/100\nHHI {metrics.get('ticker_hhi', 0):.3f}",
        "metrics": metrics,
        "data_source": "governance_health_from_concentration_metrics",
        "fallback_used": False,
        "fallback_method": "compute_portfolio_health_components",
        "limitations": ["Health score is a diagnostic composition summary, not an allocation recommendation."],
        "confidence": "Medium",
    }


def compute_semi_donut_risk_gauge(
    ticker_weights: dict[str, float],
    sector_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    health = compute_portfolio_health_donut(ticker_weights, sector_map)
    if health.get("status") != "success":
        return health

    health_rows = health.get("data", [])
    health_score = float(sum(float(row.get("score", 0.0)) for row in health_rows) / max(len(health_rows), 1))
    risk_pressure = max(0.0, min(100.0, 100.0 - health_score))
    buffer_score = max(0.0, min(100.0, health_score))
    rows = [
        {"health_component": "Risk pressure", "score": risk_pressure},
        {"health_component": "Stability buffer", "score": buffer_score},
    ]
    colors = ["#f25467", "#4cc98a"]
    for index, row in enumerate(rows):
        row["id"] = row["health_component"]
        row["label"] = row["health_component"]
        row["value"] = row["score"]
        row["color"] = colors[index]

    return {
        "status": "success",
        "data": rows,
        "series": [
            {
                "name": "portfolio_risk_gauge",
                "label": "Portfolio risk gauge",
                "data": [{"id": row["health_component"], "label": row["health_component"], "value": row["score"], "color": row["color"]} for row in rows],
                "innerRadius": 68,
                "outerRadius": 118,
                "startAngle": -90,
                "endAngle": 90,
                "cx": 150,
                "cy": 140,
                "arcLabel": "percent",
                "arcLabelMinAngle": 24,
                "highlightScope": {"fade": "global", "highlight": "item"},
            }
        ],
        "total_value": 100.0,
        "slice_count": len(rows),
        "center_label": f"Risk\n{risk_pressure:.0f}/100",
        "metrics": health.get("metrics", {}),
        "data_source": health.get("data_source"),
        "fallback_used": health.get("fallback_used", False),
        "fallback_method": "compute_semi_donut_risk_gauge",
        "limitations": ["Gauge score is derived from concentration diagnostics and is not an allocation recommendation."],
        "confidence": health.get("confidence", "Medium"),
    }


def compute_hhi_bundle(ticker_weights: dict[str, float], sector_map: dict[str, str] | None = None) -> dict[str, float]:
    return compute_concentration_metrics(normalize_weights(ticker_weights), sector_map or DEFAULT_SECTOR_MAP)


def compute_allocation_change(current_weights: dict[str, float], advisory_weights: dict[str, float]) -> dict[str, float]:
    current = normalize_weights(current_weights)
    advisory = normalize_weights(advisory_weights)
    tickers = sorted(set(current) | set(advisory))
    return {ticker: advisory.get(ticker, 0.0) - current.get(ticker, 0.0) for ticker in tickers}


def compute_returns_and_covariance(
    tickers: list[str],
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    clean = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not clean:
        return {"status": "unavailable", "reason": "tickers missing"}

    df_prices, is_mock = get_portfolio_prices(",".join(clean), start_date or DEFAULT_START_DATE, end_date or DEFAULT_END_DATE)
    df_returns = np.log(df_prices / df_prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if df_returns.empty or len(df_returns) < 2:
        return {"status": "unavailable", "reason": "not enough return observations"}

    covariance = df_returns.cov() * 252.0
    return {
        "status": "success",
        "tickers": list(df_returns.columns),
        "returns_df": df_returns,
        "covariance_matrix": covariance,
        "date_range": {
            "start": start_date or DEFAULT_START_DATE,
            "end": end_date or DEFAULT_END_DATE,
        },
        "data_source": "historical_returns" if not is_mock else "fallback_sample_price_simulation",
        "fallback_used": bool(is_mock),
    }


def load_adjusted_close_frame(
    tickers: list[str],
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    clean = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not clean:
        return {"status": "unavailable", "reason": "tickers missing"}

    start = start_date or DEFAULT_START_DATE
    end = end_date or DEFAULT_END_DATE
    price_dict = extract_prices_from_db(clean, start, end)
    fallback_used = False
    missing = [ticker for ticker in clean if ticker not in price_dict]
    if missing:
        price_dict = generate_gbm_prices(clean, start, end)
        fallback_used = True

    if not price_dict:
        return {"status": "unavailable", "reason": "price history unavailable"}

    frame = pd.DataFrame({ticker: price_dict.get(ticker) for ticker in clean if ticker in price_dict})
    if frame.empty:
        return {"status": "unavailable", "reason": "price history unavailable"}
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[~frame.index.isna()].sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame.dropna(how="all").empty:
        return {"status": "unavailable", "reason": "all adjusted close values are missing"}

    available_tickers = [ticker for ticker in clean if ticker in frame.columns]
    return {
        "status": "success",
        "tickers": available_tickers,
        "price_frame": frame[available_tickers],
        "date_range": {"start": start, "end": end},
        "data_source": "fallback_sample_price_simulation" if fallback_used else "historical_price_database",
        "fallback_used": fallback_used,
        "missing_tickers": [] if fallback_used else missing,
    }


def _date_str(index_value: Any) -> str:
    return pd.Timestamp(index_value).strftime("%Y-%m-%d")


def _line_rows_from_series(series: pd.Series, field: str, ticker: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for date_value, value in series.items():
        row: dict[str, Any] = {"date": _date_str(date_value), field: None if pd.isna(value) else float(value)}
        if ticker:
            row["ticker"] = ticker
        rows.append(row)
    return rows


def _series_points(series: pd.Series) -> list[dict[str, Any]]:
    return [
        {"x": _date_str(date_value), "y": None if pd.isna(value) else float(value)}
        for date_value, value in series.items()
    ]


def compute_historical_adjusted_close(
    tickers: list[str],
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    bundle = load_adjusted_close_frame(tickers[:1], start_date, end_date)
    if bundle.get("status") != "success":
        return bundle
    ticker = bundle["tickers"][0]
    series = bundle["price_frame"][ticker]
    data = _line_rows_from_series(series, "adjusted_close", ticker)
    return {
        "status": "success",
        "data": data,
        "series": [
            {
                "name": ticker,
                "label": ticker,
                "data": _series_points(series),
                "connectNulls": False,
                "showMark": "end",
            }
        ],
        "tickers": [ticker],
        "date_range": bundle["date_range"],
        "data_source": bundle["data_source"],
        "fallback_used": bundle["fallback_used"],
        "fallback_method": None if not bundle["fallback_used"] else "fallback_sample_price_simulation",
        "confidence": "Medium" if bundle["fallback_used"] else "High",
        "limitations": ["Fallback sample prices were used."] if bundle["fallback_used"] else [],
    }


def compute_normalized_price_comparison(
    tickers: list[str],
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    bundle = load_adjusted_close_frame(tickers, start_date, end_date)
    if bundle.get("status") != "success":
        return bundle
    frame: pd.DataFrame = bundle["price_frame"]
    normalized = pd.DataFrame(index=frame.index)
    for ticker in frame.columns:
        first_valid = frame[ticker].dropna().iloc[0] if not frame[ticker].dropna().empty else np.nan
        normalized[ticker] = frame[ticker] / first_valid * 100.0 if pd.notna(first_valid) and first_valid != 0 else np.nan

    rows: list[dict[str, Any]] = []
    for ticker in normalized.columns:
        rows.extend(_line_rows_from_series(normalized[ticker], "normalized_price", ticker))
    rows.sort(key=lambda row: (row["date"], row.get("ticker", "")))

    return {
        "status": "success",
        "data": rows,
        "series": [
            {
                "name": ticker,
                "label": ticker,
                "data": _series_points(normalized[ticker]),
                "connectNulls": False,
                "showMark": False,
            }
            for ticker in normalized.columns
        ],
        "tickers": list(normalized.columns),
        "date_range": bundle["date_range"],
        "data_source": bundle["data_source"],
        "fallback_used": bundle["fallback_used"],
        "fallback_method": "normalize_adjusted_close_to_100",
        "formula": "normalized_price = adjusted_close / first_valid_adjusted_close * 100",
        "confidence": "Medium" if bundle["fallback_used"] else "High",
        "limitations": ["Series are normalized to 100 at each ticker's first valid observation."],
    }


def compute_portfolio_value_over_time(
    tickers: list[str],
    weights: dict[str, float],
    start_date: str | None,
    end_date: str | None,
    initial_capital: float | None = None,
) -> dict[str, Any]:
    normalized_weights = normalize_weights(weights)
    if not normalized_weights:
        return {"status": "missing_data", "reason": "current weights are required", "missing_inputs": ["current_weights"]}
    bundle = load_adjusted_close_frame(tickers, start_date, end_date)
    if bundle.get("status") != "success":
        return bundle
    frame: pd.DataFrame = bundle["price_frame"]
    aligned = [ticker for ticker in frame.columns if ticker in normalized_weights]
    if not aligned:
        return {"status": "unavailable", "reason": "weights do not overlap price tickers"}
    w = pd.Series({ticker: normalized_weights[ticker] for ticker in aligned}, dtype=float)
    w = w / w.sum()
    first_valid = frame[aligned].apply(lambda col: col.dropna().iloc[0] if not col.dropna().empty else np.nan)
    normalized_prices = frame[aligned].divide(first_valid, axis=1)
    capital = float(initial_capital or DEFAULT_INITIAL_CAPITAL)
    portfolio_value = normalized_prices.mul(w, axis=1).sum(axis=1, min_count=len(aligned)) * capital
    return {
        "status": "success",
        "data": _line_rows_from_series(portfolio_value, "portfolio_value"),
        "series": [
            {
                "name": "portfolio_value",
                "label": "Portfolio value",
                "data": _series_points(portfolio_value),
                "connectNulls": False,
                "area": True,
                "showMark": "end",
            }
        ],
        "tickers": aligned,
        "date_range": bundle["date_range"],
        "data_source": bundle["data_source"],
        "fallback_used": bundle["fallback_used"],
        "fallback_method": "compute_portfolio_value_from_weights",
        "formula": "portfolio_value_t = initial_capital * sum_i(weight_i * adjusted_close_i_t / first_valid_adjusted_close_i)",
        "initial_capital": capital,
        "confidence": "Medium" if bundle["fallback_used"] else "High",
        "limitations": ["Portfolio value is computed from supplied/resolved weights and normalized ticker price paths."],
    }


def compute_drawdown_over_time(
    tickers: list[str],
    weights: dict[str, float],
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    portfolio = compute_portfolio_value_over_time(tickers, weights, start_date, end_date, DEFAULT_INITIAL_CAPITAL)
    if portfolio.get("status") != "success":
        return portfolio
    values = pd.Series(
        [row.get("portfolio_value") for row in portfolio["data"]],
        index=pd.to_datetime([row.get("date") for row in portfolio["data"]]),
        dtype=float,
    )
    running_peak = values.cummax()
    drawdown = (values - running_peak) / running_peak * 100.0
    return {
        "status": "success",
        "data": _line_rows_from_series(drawdown, "drawdown_percent"),
        "series": [
            {
                "name": "drawdown_percent",
                "label": "Drawdown",
                "data": _series_points(drawdown),
                "connectNulls": False,
                "area": True,
                "baseline": 0,
                "showMark": False,
            }
        ],
        "tickers": portfolio["tickers"],
        "date_range": portfolio["date_range"],
        "data_source": portfolio["data_source"],
        "fallback_used": portfolio["fallback_used"],
        "fallback_method": "compute_drawdown_from_portfolio_value",
        "formula": "drawdown = (portfolio_value - running_peak) / running_peak",
        "confidence": portfolio.get("confidence", "Medium"),
        "limitations": ["Drawdown uses running-peak logic on computed portfolio value."],
    }


def _returns_from_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return np.log(frame / frame.shift(1)).replace([np.inf, -np.inf], np.nan)


def compute_rolling_volatility_over_time(
    tickers: list[str],
    start_date: str | None,
    end_date: str | None,
    weights: dict[str, float] | None = None,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> dict[str, Any]:
    bundle = load_adjusted_close_frame(tickers, start_date, end_date)
    if bundle.get("status") != "success":
        return bundle
    returns = _returns_from_price_frame(bundle["price_frame"])
    normalized_weights = normalize_weights(weights or {})
    aligned = [ticker for ticker in returns.columns if ticker in normalized_weights]
    if aligned:
        w = pd.Series({ticker: normalized_weights[ticker] for ticker in aligned}, dtype=float)
        w = w / w.sum()
        base_returns = returns[aligned].mul(w, axis=1).sum(axis=1, min_count=len(aligned))
        label = "Portfolio rolling volatility"
    else:
        base_returns = returns.mean(axis=1, skipna=True)
        label = "Average ticker rolling volatility"
    rolling_volatility = base_returns.rolling(window).std() * np.sqrt(252.0) * 100.0
    return {
        "status": "success",
        "data": _line_rows_from_series(rolling_volatility, "rolling_volatility_percent"),
        "series": [
            {
                "name": "rolling_volatility_percent",
                "label": label,
                "data": _series_points(rolling_volatility),
                "connectNulls": False,
                "showMark": False,
            }
        ],
        "tickers": bundle["tickers"],
        "date_range": bundle["date_range"],
        "data_source": bundle["data_source"],
        "fallback_used": bundle["fallback_used"],
        "fallback_method": "compute_rolling_volatility_from_returns",
        "formula": f"rolling_volatility = rolling_std(daily_log_returns, {window}) * sqrt(252)",
        "confidence": "Medium" if bundle["fallback_used"] else "High",
        "limitations": [f"Uses a {window}-observation rolling window; threshold lines are included only when configured."],
    }


def compute_rolling_average_correlation(
    tickers: list[str],
    start_date: str | None,
    end_date: str | None,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> dict[str, Any]:
    bundle = load_adjusted_close_frame(tickers, start_date, end_date)
    if bundle.get("status") != "success":
        return bundle
    returns = _returns_from_price_frame(bundle["price_frame"])
    if len(returns.columns) < 2:
        return {"status": "unavailable", "reason": "rolling correlation requires at least two tickers"}
    values = []
    dates = []
    for index in range(window, len(returns) + 1):
        window_frame = returns.iloc[index - window:index].dropna(how="all")
        corr = window_frame.corr()
        if corr.shape[0] < 2:
            avg_corr = np.nan
        else:
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            avg_corr = float(upper.stack().mean()) if not upper.stack().empty else np.nan
        dates.append(returns.index[index - 1])
        values.append(avg_corr)
    series = pd.Series(values, index=pd.to_datetime(dates), dtype=float)
    return {
        "status": "success",
        "data": _line_rows_from_series(series, "average_correlation"),
        "series": [
            {
                "name": "average_correlation",
                "label": "Rolling average correlation",
                "data": _series_points(series),
                "connectNulls": False,
                "showMark": False,
            }
        ],
        "tickers": bundle["tickers"],
        "date_range": bundle["date_range"],
        "data_source": bundle["data_source"],
        "fallback_used": bundle["fallback_used"],
        "fallback_method": "compute_rolling_average_correlation",
        "formula": f"average pairwise correlation from a {window}-observation rolling return window",
        "confidence": "Medium" if bundle["fallback_used"] else "High",
        "limitations": ["Correlation is computed from aligned daily log returns."],
    }


def compute_rolling_var_cvar(
    tickers: list[str],
    weights: dict[str, float],
    start_date: str | None,
    end_date: str | None,
    window: int = DEFAULT_ROLLING_WINDOW,
    alpha: float = 0.95,
) -> dict[str, Any]:
    normalized_weights = normalize_weights(weights)
    if not normalized_weights:
        return {"status": "missing_data", "reason": "current weights are required", "missing_inputs": ["current_weights"]}
    bundle = load_adjusted_close_frame(tickers, start_date, end_date)
    if bundle.get("status") != "success":
        return bundle
    returns = _returns_from_price_frame(bundle["price_frame"])
    aligned = [ticker for ticker in returns.columns if ticker in normalized_weights]
    if not aligned:
        return {"status": "unavailable", "reason": "weights do not overlap return tickers"}
    w = pd.Series({ticker: normalized_weights[ticker] for ticker in aligned}, dtype=float)
    w = w / w.sum()
    portfolio_returns = returns[aligned].mul(w, axis=1).sum(axis=1, min_count=len(aligned))
    cvar_values = []
    var_values = []
    dates = []
    for index in range(window, len(portfolio_returns) + 1):
        window_returns = portfolio_returns.iloc[index - window:index].dropna().to_numpy(dtype=float)
        if window_returns.size:
            metrics = compute_var_cvar_from_returns(window_returns, alpha)
            var_values.append(metrics["var_95"] * 100.0)
            cvar_values.append(metrics["cvar_95"] * 100.0)
        else:
            var_values.append(np.nan)
            cvar_values.append(np.nan)
        dates.append(portfolio_returns.index[index - 1])
    cvar_series = pd.Series(cvar_values, index=pd.to_datetime(dates), dtype=float)
    var_series = pd.Series(var_values, index=pd.to_datetime(dates), dtype=float)
    rows = _line_rows_from_series(cvar_series, "cvar_95")
    for row, var_value in zip(rows, var_series.to_list(), strict=False):
        row["var_95"] = None if pd.isna(var_value) else float(var_value)
    return {
        "status": "success",
        "data": rows,
        "series": [
            {
                "name": "cvar_95",
                "label": "CVaR 95",
                "data": _series_points(cvar_series),
                "connectNulls": False,
                "showMark": False,
            },
            {
                "name": "var_95",
                "label": "VaR 95",
                "data": _series_points(var_series),
                "connectNulls": False,
                "showMark": False,
                "curve": "linear",
            },
        ],
        "tickers": aligned,
        "date_range": bundle["date_range"],
        "data_source": bundle["data_source"],
        "fallback_used": bundle["fallback_used"],
        "fallback_method": "compute_rolling_var_cvar",
        "formula": "CVaR 95 is the mean loss beyond the rolling 95% VaR threshold, computed from portfolio returns.",
        "confidence": "Medium" if bundle["fallback_used"] else "High",
        "limitations": [f"VaR/CVaR use a {window}-observation rolling window and resolved current weights."],
    }


def compute_instability_index_over_time(
    tickers: list[str],
    start_date: str | None,
    end_date: str | None,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> dict[str, Any]:
    bundle = load_adjusted_close_frame(tickers, start_date, end_date)
    if bundle.get("status") != "success":
        return bundle
    frame: pd.DataFrame = bundle["price_frame"]
    returns = _returns_from_price_frame(frame)
    mean_returns = returns.mean(axis=1, skipna=True)
    rolling_vol = mean_returns.rolling(window).std() * np.sqrt(252.0)
    equity = np.exp(mean_returns.fillna(0).cumsum())
    drawdown = 1.0 - equity / equity.cummax()
    corr_result = compute_rolling_average_correlation(tickers, start_date, end_date, window)
    if corr_result.get("status") == "success":
        corr_series = pd.Series(
            [row.get("average_correlation") for row in corr_result["data"]],
            index=pd.to_datetime([row.get("date") for row in corr_result["data"]]),
            dtype=float,
        ).reindex(rolling_vol.index)
    else:
        corr_series = pd.Series(np.nan, index=rolling_vol.index)

    vol_component = _normalize_01(rolling_vol)
    corr_component = _normalize_01(corr_series.clip(lower=0))
    drawdown_component = _normalize_01(drawdown)
    instability = (0.4 * vol_component + 0.3 * corr_component + 0.3 * drawdown_component).clip(0, 1)
    rows = _line_rows_from_series(instability, "instability_index")
    for row in rows:
        row["calm_threshold"] = 0.50
        row["crisis_threshold"] = 0.85
    return {
        "status": "success",
        "data": rows,
        "series": [
            {
                "name": "instability_index",
                "label": "Instability index",
                "data": _series_points(instability),
                "connectNulls": False,
                "showMark": "end",
            },
            {
                "name": "calm_threshold",
                "label": "Calm threshold",
                "data": [{"x": row["date"], "y": 0.50} for row in rows],
                "connectNulls": False,
                "showMark": False,
                "curve": "linear",
            },
            {
                "name": "crisis_threshold",
                "label": "Crisis threshold",
                "data": [{"x": row["date"], "y": 0.85} for row in rows],
                "connectNulls": False,
                "showMark": False,
                "curve": "linear",
            },
        ],
        "tickers": bundle["tickers"],
        "date_range": bundle["date_range"],
        "data_source": bundle["data_source"],
        "fallback_used": bundle["fallback_used"],
        "fallback_method": "compute_instability_index_if_components_available",
        "formula": "instability_index = 0.4 * volatility_component + 0.3 * correlation_component + 0.3 * drawdown_component",
        "confidence": "Medium" if bundle["fallback_used"] else "High",
        "limitations": ["Instability is a computed diagnostic series; no allocation or optimizer output is generated."],
    }


def _normalize_01(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    min_value = clean.min(skipna=True)
    max_value = clean.max(skipna=True)
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(0.0, index=clean.index)
    return ((clean - min_value) / (max_value - min_value)).fillna(0.0)


def compute_risk_contribution(
    tickers: list[str],
    weights: dict[str, float],
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    returns_bundle = compute_returns_and_covariance(tickers, start_date, end_date)
    if returns_bundle.get("status") != "success":
        return returns_bundle

    cov_df: pd.DataFrame = returns_bundle["covariance_matrix"]
    normalized_weights = normalize_weights(weights)
    aligned_tickers = [ticker for ticker in cov_df.columns if ticker in normalized_weights]
    if not aligned_tickers:
        return {"status": "unavailable", "reason": "weights do not overlap covariance tickers"}

    cov = cov_df.loc[aligned_tickers, aligned_tickers].to_numpy(dtype=float)
    w = np.array([normalized_weights[ticker] for ticker in aligned_tickers], dtype=float)
    w = w / w.sum()
    denominator = float(w.T @ cov @ w)
    if denominator <= 0 or not np.isfinite(denominator):
        return {"status": "unavailable", "reason": "portfolio variance is not finite"}

    marginal = cov @ w
    contributions = w * marginal / denominator
    data = [
        {
            "ticker": ticker,
            "risk_contribution_percent": float(contributions[index] * 100.0),
            "allocation_percent": float(w[index] * 100.0),
        }
        for index, ticker in enumerate(aligned_tickers)
    ]
    data.sort(key=lambda row: row["risk_contribution_percent"], reverse=True)
    return {
        "status": "success",
        "plot_id": "plot_55_risk_contribution_by_ticker",
        "chart_type": "bar",
        "bar_mode": "horizontal",
        "data": data,
        "covariance_matrix": cov_df.to_dict(),
        "data_source": returns_bundle["data_source"],
        "fallback_used": returns_bundle["fallback_used"],
        "formula": "risk_contribution_i = w_i * (Sigma w)_i / (w^T Sigma w)",
        "confidence": "Medium" if returns_bundle["fallback_used"] else "High",
        "limitations": [
            "Actual current holdings were not available; proxy weights are used when active_weights.type is equal_weight_proxy."
        ],
    }


def compute_risk_contribution_donut(
    tickers: list[str],
    weights: dict[str, float],
    start_date: str | None,
    end_date: str | None,
    max_slices: int = 10,
) -> dict[str, Any]:
    result = compute_risk_contribution(tickers, weights, start_date, end_date)
    if result.get("status") != "success":
        return result

    raw_rows = [
        {
            "name": row["ticker"],
            "ticker": row["ticker"],
            "risk_contribution_percent": float(row["risk_contribution_percent"]),
            "allocation_percent": float(row.get("allocation_percent", 0.0)),
        }
        for row in result.get("data", [])
    ]
    if any(row["risk_contribution_percent"] < -1e-8 for row in raw_rows):
        return {
            "status": "unavailable",
            "reason": "risk contribution contains negative values; use the signed risk contribution bar chart instead",
            "missing_inputs": ["non_negative_risk_contribution"],
            "fallback_used": False,
        }

    for row in raw_rows:
        if abs(row["risk_contribution_percent"]) < 1e-8:
            row["risk_contribution_percent"] = 0.0
    rows = _limit_percent_slices(
        raw_rows,
        value_field="risk_contribution_percent",
        label_field="name",
        max_slices=max_slices,
    )
    colors = _slice_colors(len(rows))
    for index, row in enumerate(rows):
        row["id"] = row["name"]
        row["label"] = row["name"]
        row["value"] = row["risk_contribution_percent"]
        row["color"] = colors[index]
    return {
        "status": "success",
        "data": rows,
        "series": [
            {
                "name": "risk_contribution_percent",
                "label": "Risk contribution",
                "data": [{"id": row["name"], "label": row["name"], "value": row["risk_contribution_percent"], "color": row["color"]} for row in rows],
                "innerRadius": 58,
                "outerRadius": 112,
                "arcLabel": "label-percent",
                "arcLabelMinAngle": 20,
                "highlightScope": {"fade": "global", "highlight": "item"},
                "highlighted": {"additionalRadius": 4},
            }
        ],
        "total_value": float(sum(row["risk_contribution_percent"] for row in rows)),
        "slice_count": len(rows),
        "center_label": f"Risk share\n{len(rows)} slices",
        "covariance_matrix": result.get("covariance_matrix"),
        "data_source": result.get("data_source"),
        "fallback_used": bool(result.get("fallback_used")),
        "fallback_method": "compute_risk_contribution_from_weights_and_covariance",
        "formula": result.get("formula"),
        "limitations": result.get("limitations", []),
        "confidence": result.get("confidence", "Medium"),
    }


def compute_return_range_by_ticker(
    tickers: list[str],
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    returns_bundle = compute_returns_and_covariance(tickers, start_date, end_date)
    if returns_bundle.get("status") != "success":
        return returns_bundle
    returns_df: pd.DataFrame = returns_bundle["returns_df"]
    data = [
        {
            "ticker": ticker,
            "min_return": float(returns_df[ticker].min() * 100.0),
            "max_return": float(returns_df[ticker].max() * 100.0),
        }
        for ticker in returns_df.columns
    ]
    data.sort(key=lambda row: row["max_return"] - row["min_return"], reverse=True)
    return {
        "status": "success",
        "data": data,
        "data_source": returns_bundle["data_source"],
        "fallback_used": returns_bundle["fallback_used"],
        "confidence": "Medium" if returns_bundle["fallback_used"] else "High",
        "limitations": ["Return ranges are based on daily log returns over the resolved date range."],
    }


def compute_volatility_range_by_ticker(
    tickers: list[str],
    start_date: str | None,
    end_date: str | None,
    window: int = 20,
) -> dict[str, Any]:
    returns_bundle = compute_returns_and_covariance(tickers, start_date, end_date)
    if returns_bundle.get("status") != "success":
        return returns_bundle
    returns_df: pd.DataFrame = returns_bundle["returns_df"]
    rolling_vol = returns_df.rolling(window).std().dropna(how="all") * np.sqrt(252.0) * 100.0
    if rolling_vol.empty:
        rolling_vol = returns_df.std().to_frame().T * np.sqrt(252.0) * 100.0
    data = [
        {
            "ticker": ticker,
            "min_volatility": float(rolling_vol[ticker].min()),
            "max_volatility": float(rolling_vol[ticker].max()),
        }
        for ticker in rolling_vol.columns
    ]
    data.sort(key=lambda row: row["max_volatility"] - row["min_volatility"], reverse=True)
    return {
        "status": "success",
        "data": data,
        "data_source": returns_bundle["data_source"],
        "fallback_used": returns_bundle["fallback_used"],
        "confidence": "Medium" if returns_bundle["fallback_used"] else "High",
        "limitations": [f"Volatility ranges use {window}-day rolling annualized volatility."],
    }


def compute_return_distribution_histogram(
    tickers: list[str],
    weights: dict[str, float],
    start_date: str | None,
    end_date: str | None,
    bins: int = 20,
) -> dict[str, Any]:
    returns_bundle = compute_returns_and_covariance(tickers, start_date, end_date)
    if returns_bundle.get("status") != "success":
        return returns_bundle
    returns_df: pd.DataFrame = returns_bundle["returns_df"]
    normalized_weights = normalize_weights(weights)
    aligned_tickers = [ticker for ticker in returns_df.columns if ticker in normalized_weights]
    if not aligned_tickers:
        return {"status": "unavailable", "reason": "weights do not overlap return tickers"}
    w = np.array([normalized_weights[ticker] for ticker in aligned_tickers], dtype=float)
    w = w / w.sum()
    portfolio_returns = returns_df[aligned_tickers].to_numpy(dtype=float) @ w * 100.0
    counts, edges = np.histogram(portfolio_returns, bins=bins)
    data = [
        {
            "bin_start": float(edges[index]),
            "bin_end": float(edges[index + 1]),
            "count": int(counts[index]),
        }
        for index in range(len(counts))
    ]
    return {
        "status": "success",
        "data": data,
        "data_source": returns_bundle["data_source"],
        "fallback_used": returns_bundle["fallback_used"],
        "confidence": "Medium" if returns_bundle["fallback_used"] else "High",
        "limitations": ["Histogram uses weighted daily portfolio log returns expressed in percentage points."],
    }


def compute_portfolio_return_waterfall(
    tickers: list[str],
    weights: dict[str, float],
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    returns_bundle = compute_returns_and_covariance(tickers, start_date, end_date)
    if returns_bundle.get("status") != "success":
        return returns_bundle
    returns_df: pd.DataFrame = returns_bundle["returns_df"]
    normalized_weights = normalize_weights(weights)
    aligned_tickers = [ticker for ticker in returns_df.columns if ticker in normalized_weights]
    if not aligned_tickers:
        return {"status": "unavailable", "reason": "weights do not overlap return tickers"}

    contributions = []
    for ticker in aligned_tickers:
        ticker_total_return = float((np.exp(returns_df[ticker].sum()) - 1.0) * 100.0)
        contributions.append((ticker, ticker_total_return * normalized_weights[ticker]))
    contributions.sort(key=lambda item: abs(item[1]), reverse=True)

    running = 0.0
    data = []
    for ticker, contribution in contributions:
        start_value = running
        running += contribution
        data.append({"component": ticker, "start_value": float(start_value), "end_value": float(running)})
    data.append({"component": "Total", "start_value": 0.0, "end_value": float(running)})
    return {
        "status": "success",
        "data": data,
        "data_source": returns_bundle["data_source"],
        "fallback_used": returns_bundle["fallback_used"],
        "confidence": "Medium" if returns_bundle["fallback_used"] else "High",
        "limitations": ["Contribution waterfall uses current weights and realized ticker log returns."],
    }


def compute_risk_contribution_waterfall(
    tickers: list[str],
    weights: dict[str, float],
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    result = compute_risk_contribution(tickers, weights, start_date, end_date)
    if result.get("status") != "success":
        return result
    running = 0.0
    data = []
    for row in sorted(result["data"], key=lambda item: abs(item["risk_contribution_percent"]), reverse=True):
        start_value = running
        running += float(row["risk_contribution_percent"])
        data.append({"component": row["ticker"], "start_value": float(start_value), "end_value": float(running)})
    data.append({"component": "Total", "start_value": 0.0, "end_value": float(running)})
    return {
        "status": "success",
        "data": data,
        "data_source": result["data_source"],
        "fallback_used": result["fallback_used"],
        "confidence": result.get("confidence", "Medium"),
        "limitations": result.get("limitations", []),
    }


def compute_var_cvar_from_returns(portfolio_returns: np.ndarray, alpha: float = 0.95) -> dict[str, float]:
    if portfolio_returns.size == 0:
        return {"var_95": 0.0, "cvar_95": 0.0}
    losses = -portfolio_returns
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size else var
    return {"var_95": var, "cvar_95": cvar}


def compute_drawdown_from_returns(portfolio_returns: np.ndarray) -> dict[str, float]:
    if portfolio_returns.size == 0:
        return {"maximum_drawdown": 0.0}
    equity = np.exp(np.cumsum(portfolio_returns))
    drawdown = 1.0 - equity / np.maximum.accumulate(equity)
    return {"maximum_drawdown": float(np.max(drawdown))}


def _annualized_return_percent(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    return float((np.exp(clean.mean() * 252.0) - 1.0) * 100.0)


def _annualized_volatility_percent(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.size < 2:
        return 0.0
    return float(clean.std(ddof=1) * np.sqrt(252.0) * 100.0)


def _ticker_max_drawdown_percent(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    return float(compute_drawdown_from_returns(clean)["maximum_drawdown"] * 100.0)


def _ticker_cvar_percent(returns: pd.Series, alpha: float = 0.95) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    return float(compute_var_cvar_from_returns(clean, alpha)["cvar_95"] * 100.0)


def _sector_for_ticker(ticker: str, sector_map: dict[str, str] | None = None) -> str:
    return (sector_map or DEFAULT_SECTOR_MAP).get(str(ticker).upper(), "Unknown")


def _scatter_rows_to_series(
    rows: list[dict[str, Any]],
    *,
    x_field: str,
    y_field: str,
    id_field: str,
    color_field: str | None = None,
    size_field: str | None = None,
    marker_size: int = 8,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    if color_field:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            group = str(row.get(color_field) or "Other")
            groups.setdefault(group, []).append(row)
    else:
        groups = {"Series": rows}

    colors = _slice_colors(len(groups))
    series = []
    for index, (group, group_rows) in enumerate(sorted(groups.items(), key=lambda item: item[0])):
        data = []
        for row in group_rows:
            point = {
                "x": float(row[x_field]),
                "y": float(row[y_field]),
                "id": str(row.get(id_field)),
                "label": str(row.get(id_field)),
            }
            if size_field:
                point["sizeValue"] = float(row.get(size_field, 0.0))
            if color_field:
                point["colorValue"] = group
            data.append(point)
        entry = {
            "name": group,
            "label": group,
            "color": colors[index],
            "markerSize": marker_size,
            "data": data,
            "highlightScope": {"highlight": "series", "fade": "global"},
        }
        if size_field:
            entry["sizeAxisId"] = "point_size"
        if color_field:
            entry["colorAxisId"] = "point_color"
        series.append(entry)
    return series


def _scatter_common_payload(
    *,
    rows: list[dict[str, Any]],
    bundle: dict[str, Any],
    x_field: str,
    y_field: str,
    id_field: str,
    title: str,
    x_label: str,
    y_label: str,
    fallback_method: str,
    formula: str,
    color_field: str | None = "sector",
    size_field: str | None = None,
    chart_type: str = "scatter",
    component: str = "ScatterChart",
    chart_tier: str = "free",
    requires_premium: bool = False,
    fallback_chart: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    valid_rows = [
        row for row in rows
        if row.get(id_field) is not None
        and np.isfinite(float(row.get(x_field, np.nan)))
        and np.isfinite(float(row.get(y_field, np.nan)))
    ]
    z_axis = []
    if size_field and valid_rows:
        sizes = [float(row.get(size_field, 0.0)) for row in valid_rows]
        min_size = float(min(sizes))
        max_size = float(max(sizes))
        if min_size == max_size:
            min_size = 0.0
            max_size = max(max_size, 1.0)
        z_axis.append(
            {
                "id": "point_size",
                "sizeMap": {"type": "continuous", "min": min_size, "max": max_size, "size": [5, 22]},
            }
        )
    if color_field and valid_rows:
        values = sorted({str(row.get(color_field) or "Other") for row in valid_rows})
        z_axis.append(
            {
                "id": "point_color",
                "data": values,
                "colorMap": {"type": "ordinal", "values": values, "colors": _slice_colors(len(values))},
            }
        )

    payload: dict[str, Any] = {
        "status": "success",
        "chart_type": chart_type,
        "chart_tier": chart_tier,
        "component": component,
        "requires_premium": requires_premium,
        "fallback_chart": fallback_chart,
        "data": valid_rows,
        "series": _scatter_rows_to_series(
            valid_rows,
            x_field=x_field,
            y_field=y_field,
            id_field=id_field,
            color_field=color_field,
            size_field=size_field,
        ),
        "xAxis": [{"label": x_label, "domainLimit": "nice"}],
        "yAxis": [{"label": y_label, "domainLimit": "nice", "width": 68}],
        "zAxis": z_axis or None,
        "x_axis": x_field,
        "y_axis": y_field,
        "x_unit": "%",
        "y_unit": "%",
        "point_id": id_field,
        "point_count": len(valid_rows),
        "color_axis": color_field,
        "size_axis": size_field,
        "title": title,
        "grid": {"horizontal": True, "vertical": True},
        "hitAreaRadius": 24,
        "data_source": bundle.get("data_source"),
        "fallback_used": bool(bundle.get("fallback_used")),
        "fallback_method": fallback_method,
        "formula": formula,
        "confidence": "Medium" if bundle.get("fallback_used") else "High",
        "limitations": ["Scatter metrics are computed from aligned adjusted close price history."],
    }
    if not z_axis:
        payload.pop("zAxis", None)
    if extra:
        payload.update(extra)
    return payload


def compute_risk_return_scatter(
    tickers: list[str],
    start_date: str | None,
    end_date: str | None,
    weights: dict[str, float] | None = None,
    sector_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    bundle = load_adjusted_close_frame(tickers, start_date, end_date)
    if bundle.get("status") != "success":
        return bundle
    returns = _returns_from_price_frame(bundle["price_frame"])
    normalized_weights = normalize_weights(weights or {})
    rows = []
    for ticker in bundle["tickers"]:
        series = returns[ticker].dropna()
        if len(series) < 2:
            continue
        rows.append(
            {
                "ticker": ticker,
                "annualized_volatility_percent": _annualized_volatility_percent(series),
                "annualized_return_percent": _annualized_return_percent(series),
                "allocation_percent": float(normalized_weights.get(ticker, 0.0) * 100.0),
                "sector": _sector_for_ticker(ticker, sector_map),
            }
        )
    rows.sort(key=lambda row: row["annualized_volatility_percent"], reverse=True)
    return _scatter_common_payload(
        rows=rows,
        bundle=bundle,
        x_field="annualized_volatility_percent",
        y_field="annualized_return_percent",
        id_field="ticker",
        title="Risk-Return Scatter by Ticker",
        x_label="Annualized Volatility (%)",
        y_label="Annualized Return (%)",
        fallback_method="compute_return_and_volatility_from_adjusted_close",
        formula="x = daily log-return standard deviation * sqrt(252); y = exp(mean daily log return * 252) - 1",
    )


def compute_cvar_return_scatter(
    tickers: list[str],
    start_date: str | None,
    end_date: str | None,
    sector_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    bundle = load_adjusted_close_frame(tickers, start_date, end_date)
    if bundle.get("status") != "success":
        return bundle
    returns = _returns_from_price_frame(bundle["price_frame"])
    rows = []
    for ticker in bundle["tickers"]:
        series = returns[ticker].dropna()
        if len(series) < 2:
            continue
        rows.append(
            {
                "ticker": ticker,
                "cvar_95_percent": _ticker_cvar_percent(series),
                "annualized_return_percent": _annualized_return_percent(series),
                "sector": _sector_for_ticker(ticker, sector_map),
            }
        )
    rows.sort(key=lambda row: row["cvar_95_percent"], reverse=True)
    return _scatter_common_payload(
        rows=rows,
        bundle=bundle,
        x_field="cvar_95_percent",
        y_field="annualized_return_percent",
        id_field="ticker",
        title="CVaR-Return Scatter by Ticker",
        x_label="CVaR 95 (%)",
        y_label="Annualized Return (%)",
        fallback_method="compute_cvar_and_return_from_returns",
        formula="x = mean loss beyond daily 95% VaR; y = exp(mean daily log return * 252) - 1",
    )


def compute_drawdown_return_scatter(
    tickers: list[str],
    start_date: str | None,
    end_date: str | None,
    sector_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    bundle = load_adjusted_close_frame(tickers, start_date, end_date)
    if bundle.get("status") != "success":
        return bundle
    returns = _returns_from_price_frame(bundle["price_frame"])
    rows = []
    for ticker in bundle["tickers"]:
        series = returns[ticker].dropna()
        if len(series) < 2:
            continue
        rows.append(
            {
                "ticker": ticker,
                "max_drawdown_percent": _ticker_max_drawdown_percent(series),
                "annualized_return_percent": _annualized_return_percent(series),
                "sector": _sector_for_ticker(ticker, sector_map),
            }
        )
    rows.sort(key=lambda row: row["max_drawdown_percent"], reverse=True)
    return _scatter_common_payload(
        rows=rows,
        bundle=bundle,
        x_field="max_drawdown_percent",
        y_field="annualized_return_percent",
        id_field="ticker",
        title="Drawdown-Return Scatter by Ticker",
        x_label="Maximum Drawdown (%)",
        y_label="Annualized Return (%)",
        fallback_method="compute_drawdown_and_return_from_price_history",
        formula="x = maximum running-peak drawdown from cumulative returns; y = exp(mean daily log return * 252) - 1",
    )


def compute_allocation_vs_risk_contribution_scatter(
    tickers: list[str],
    weights: dict[str, float],
    start_date: str | None,
    end_date: str | None,
    sector_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    computed = compute_risk_contribution(tickers, weights, start_date, end_date)
    if computed.get("status") != "success":
        return computed
    rows = []
    for row in computed.get("data", []):
        ticker = row["ticker"]
        allocation = float(row.get("allocation_percent", 0.0))
        risk_contribution = float(row.get("risk_contribution_percent", 0.0))
        rows.append(
            {
                "ticker": ticker,
                "allocation_percent": allocation,
                "risk_contribution_percent": risk_contribution,
                "allocation_risk_gap_percent": risk_contribution - allocation,
                "bubble_size_value": abs(risk_contribution - allocation),
                "sector": _sector_for_ticker(ticker, sector_map),
            }
        )
    return _scatter_common_payload(
        rows=rows,
        bundle=computed,
        x_field="allocation_percent",
        y_field="risk_contribution_percent",
        id_field="ticker",
        title="Allocation vs Risk Contribution Scatter",
        x_label="Allocation Share (%)",
        y_label="Risk Contribution (%)",
        fallback_method="compute_risk_contribution_from_weights_and_covariance",
        formula="risk_contribution_i = w_i * (Sigma w)_i / (w^T Sigma w)",
        extra={
            "covariance_matrix": computed.get("covariance_matrix"),
            "limitations": computed.get("limitations", []),
        },
    )


def compute_bubble_risk_return_scatter(
    tickers: list[str],
    weights: dict[str, float],
    start_date: str | None,
    end_date: str | None,
    sector_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    normalized_weights = normalize_weights(weights)
    if not normalized_weights:
        return {"status": "missing_data", "reason": "current weights are required for bubble size", "missing_inputs": ["current_weights"]}
    computed = compute_risk_return_scatter(tickers, start_date, end_date, normalized_weights, sector_map)
    if computed.get("status") != "success":
        return computed
    rows = []
    for row in computed.get("data", []):
        ticker = row["ticker"]
        allocation = float(normalized_weights.get(ticker, 0.0) * 100.0)
        next_row = dict(row)
        next_row["bubble_size_value"] = allocation
        next_row["allocation_percent"] = allocation
        rows.append(next_row)
    return _scatter_common_payload(
        rows=rows,
        bundle=computed,
        x_field="annualized_volatility_percent",
        y_field="annualized_return_percent",
        id_field="ticker",
        title="Risk-Return Bubble by Ticker",
        x_label="Annualized Volatility (%)",
        y_label="Annualized Return (%)",
        fallback_method="compute_bubble_size_from_weight_or_risk_contribution",
        formula="x = annualized volatility; y = annualized return; bubble size = current allocation weight",
        size_field="bubble_size_value",
        chart_type="bubble_scatter",
        component="ScatterChartPro",
        chart_tier="pro",
        fallback_chart="risk_return_scatter",
        extra={
            "size_unit": "%",
            "limitations": [
                "Bubble size uses resolved current weights. It is not an advisory allocation recommendation."
            ],
        },
    )


def compute_pairwise_return_regression_scatter(
    tickers: list[str],
    start_date: str | None,
    end_date: str | None,
    max_points: int = 500,
) -> dict[str, Any]:
    clean = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if len(clean) < 2:
        return {"status": "unavailable", "reason": "at least two tickers are required for regression scatter", "missing_inputs": ["second_ticker"]}
    pair = clean[:2]
    bundle = load_adjusted_close_frame(pair, start_date, end_date)
    if bundle.get("status") != "success":
        return bundle
    returns = _returns_from_price_frame(bundle["price_frame"])[pair].dropna(how="any") * 100.0
    if len(returns) < 3:
        return {"status": "unavailable", "reason": "regression scatter requires at least three aligned return observations", "missing_inputs": ["aligned_returns"]}
    sampled = returns
    sampled_points = False
    if len(returns) > max_points:
        stride = int(np.ceil(len(returns) / max_points))
        sampled = returns.iloc[::stride].head(max_points)
        sampled_points = True
    rows = [
        {
            "date": _date_str(date_value),
            "ticker_x": pair[0],
            "ticker_y": pair[1],
            "x_return_percent": float(values[pair[0]]),
            "y_return_percent": float(values[pair[1]]),
        }
        for date_value, values in sampled.iterrows()
    ]
    x = returns[pair[0]].to_numpy(dtype=float)
    y = returns[pair[1]].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 0.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    regression_line = {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": float(slope * x_min + intercept),
        "y_max": float(slope * x_max + intercept),
        "method": "ordinary_least_squares",
    }
    payload = _scatter_common_payload(
        rows=rows,
        bundle=bundle,
        x_field="x_return_percent",
        y_field="y_return_percent",
        id_field="date",
        title=f"{pair[0]} vs {pair[1]} Return Regression Scatter",
        x_label=f"{pair[0]} Daily Return (%)",
        y_label=f"{pair[1]} Daily Return (%)",
        fallback_method="compute_ols_regression",
        formula="ordinary least squares regression on aligned daily log returns",
        color_field=None,
        chart_type="scatter_regression",
        extra={
            "tickers": pair,
            "regression_used": True,
            "regression_method": "ordinary_least_squares",
            "regression_line": regression_line,
            "r_squared": r_squared,
            "sampled_points": sampled_points,
            "raw_point_count": len(returns),
            "limitations": [
                "Regression is descriptive and uses aligned daily log returns; it is not a forecast."
            ],
        },
    )
    return payload
