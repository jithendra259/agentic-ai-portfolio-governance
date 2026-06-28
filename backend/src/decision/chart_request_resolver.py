import re
from typing import Any

from src.agents.derived_plot_tools import run_data_analysis_plot


TICKER_RE = re.compile(r"(?<![A-Z0-9.])([A-Z0-9]{1,10}(?:[.-][A-Z0-9]{1,6})?)(?![A-Z0-9.])")
DATE_RE = re.compile(r"\b(?:19|20)\d{2}(?:-\d{2})?(?:-\d{2})?\b")
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2025-01-01"
KNOWN_SECTORS = {
    "technology": "Technology",
    "information technology": "Technology",
    "financials": "Financials",
    "financial services": "Financial Services",
    "healthcare": "Healthcare",
    "health care": "Healthcare",
    "energy": "Energy",
    "utilities": "Utilities",
    "industrials": "Industrials",
    "industrial": "Industrials",
    "real estate": "Real Estate",
    "consumer discretionary": "Consumer Discretionary",
    "consumer staples": "Consumer Staples",
    "communication services": "Communication Services",
    "materials": "Materials",
}

STOP_TICKERS = {
    "AI",
    "API",
    "BOX",
    "CV",
    "EDA",
    "ETF",
    "GDP",
    "HI",
    "I",
    "LLM",
    "MA",
    "OHLC",
    "SMA",
    "UI",
    "USD",
}


def _normalize_date(raw: str, end: bool = False) -> str:
    value = str(raw or "").strip()
    if len(value) == 4:
        return f"{value}-12-31" if end else f"{value}-01-01"
    if len(value) == 7:
        return f"{value}-12" if end else f"{value}-01"
    return value


def _extract_dates(message: str) -> tuple[str, str]:
    dates = DATE_RE.findall(message or "")
    if len(dates) >= 2:
        return _normalize_date(dates[0]), _normalize_date(dates[1], end=True)
    if len(dates) == 1:
        return _normalize_date(dates[0]), DEFAULT_END_DATE
    return DEFAULT_START_DATE, DEFAULT_END_DATE


def _extract_tickers(message: str) -> list[str]:
    seen = set()
    tickers = []
    for raw_ticker in TICKER_RE.findall(message or ""):
        if raw_ticker != raw_ticker.upper() or raw_ticker.isdigit():
            continue
        ticker = raw_ticker.upper()
        if len(ticker) == 1 and "." not in ticker and "-" not in ticker:
            continue
        if ticker in STOP_TICKERS or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)
    return tickers


def _extract_sector(message: str) -> str | None:
    normalized = re.sub(r"\s+", " ", str(message or "").lower())
    for alias, canonical in sorted(KNOWN_SECTORS.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return canonical
    return None


def _looks_like_chart_request(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "plot",
            "chart",
            "graph",
            "heatmap",
            "heat map",
            "boxplot",
            "box plot",
            "candlestick",
            "candle",
            "area",
            "scatter",
            "bar",
            "line",
        )
    )


def _choose_analysis_task(text: str) -> str | None:
    if ("missing" in text or "null" in text or "coverage" in text or "availability" in text) and (
        "heatmap" in text or "heat map" in text
    ):
        return "missing_data_heatmap"
    if ("ohlc" in text or ("open" in text and "high" in text and "low" in text and "close" in text)) and (
        "corr" in text or "correlation" in text
    ):
        return "ohlc_correlation_heatmap"
    if ("corr" in text or "correlation" in text or "covariance" in text) and (
        "heatmap" in text or "heat map" in text
    ):
        return "returns_correlation_heatmap"
    if "boxplot" in text or "box plot" in text or "box-and-whisker" in text or "whisker" in text:
        return "returns_box_plot"
    if "spread" in text and ("area" in text or "line" in text or "plot" in text or "chart" in text):
        return "price_spread_area"
    if "area" in text and ("take any" in text or "any metric" in text or "default" in text or "use any" in text):
        return "price_spread_area"
    if "price line" in text or (
        ("line" in text or "plot" in text or "chart" in text)
        and ("price" in text or "close" in text or "closing" in text)
    ):
        return "price_line"
    return None


def resolve_deterministic_chart_request(message: str, session_id: str) -> dict[str, Any] | None:
    """
    Resolve obvious chart requests without relying on LLM tool-choice.

    This is intentionally conservative: it only handles chart requests where a
    safe approved analysis_task exists. The LLM still handles exploratory
    conversation and chart-type explanations.
    """
    raw_message = str(message or "").strip()
    text = re.sub(r"\s+", " ", raw_message.lower())
    if not _looks_like_chart_request(text):
        return None

    analysis_task = _choose_analysis_task(text)
    if not analysis_task:
        return None

    tickers = _extract_tickers(raw_message)
    sector = _extract_sector(raw_message)
    start_date, end_date = _extract_dates(raw_message)

    if analysis_task == "price_spread_area" and len(tickers) < 2 and not sector:
        tickers = ["AAPL", "MSFT"]
    elif analysis_task == "returns_box_plot" and not tickers and not sector:
        tickers = ["AAPL", "MSFT", "NVDA"]
    elif analysis_task == "returns_correlation_heatmap" and len(tickers) < 2 and not sector:
        tickers = ["AAPL", "MSFT", "NVDA"]
    elif analysis_task == "ohlc_correlation_heatmap" and not tickers and not sector:
        tickers = ["AAPL"]

    result = run_data_analysis_plot.func(
        analysis_task=analysis_task,
        tickers=tickers,
        ticker=tickers[0] if tickers else None,
        sector=sector,
        start_date=start_date,
        end_date=end_date,
        config={"configurable": {"thread_id": session_id}},
    )
    if not isinstance(result, dict) or result.get("status") != "success" or not result.get("plot_id"):
        return None

    return {
        "analysis_task": analysis_task,
        "plot_id": result["plot_id"],
        "tickers": result.get("tickers") or tickers,
        "sector": sector,
        "start_date": start_date,
        "end_date": end_date,
        "summary": result,
    }


def build_chart_response(resolved: dict[str, Any]) -> str:
    task = resolved.get("analysis_task", "chart")
    tickers = ", ".join(resolved.get("tickers") or [])
    sector = resolved.get("sector")
    start_date = resolved.get("start_date")
    end_date = resolved.get("end_date")
    labels = {
        "missing_data_heatmap": "missing-data heatmap",
        "ohlc_correlation_heatmap": "OHLC correlation heatmap",
        "returns_correlation_heatmap": "returns correlation heatmap",
        "returns_box_plot": "daily returns box plot",
        "price_line": "close-price line chart",
        "price_spread_area": "close-price spread area chart",
    }
    chart_name = labels.get(task, task.replace("_", " "))
    scope = f" for {sector}" if sector else f" for {tickers}" if tickers else ""
    window = f" from {start_date} to {end_date}" if start_date and end_date else ""
    return f"Done — I generated the {chart_name}{scope}{window}."
