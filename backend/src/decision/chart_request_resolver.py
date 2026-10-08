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
    "INR",
    "THE",
    "AND",
    "FOR",
    "OF",
    "FROM",
    "TO",
    "IN",
    "ON",
    "AT",
    "BY",
    "WITH",
    "PLOT",
    "CHART",
    "LINE",
    "CLOSE",
    "CLOSING",
    "PRICE",
    "PRICES",
    "DATA",
    "STOCK",
    "STOCKS",
    "DAILY",
    "HISTORICAL",
    "SHOW",
    "GET",
    "DRAW",
    "GRAPH",
    "DISPLAY",
    "GIVE",
    "ME",
    "PLEASE",
    "CAN",
    "YOU",
    "WHAT",
    "IS",
    "ARE",
    "SEE",
    "VIEW",
    "ANY",
    "TAKE",
    "METRIC",
    "REAL",
    "ESTATE",
    "NOW",
    "SO",
    "NOT",
    "ALL",
}

COMPANY_ALIASES = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "nvdia": "NVDA",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "infy": "INFY.NS",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "berkshire": "BRK-B",
    "berkshire hathaway": "BRK-B",
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "walmart": "WMT",
    "broadcom": "AVGO",
    "eli lilly": "LLY",
    "lilly": "LLY",
    "exxon": "XOM",
    "exxon mobil": "XOM",
    "amd": "AMD",
    "intel": "INTC",
    "cisco": "CSCO",
    "qualcomm": "QCOM",
    "adobe": "ADBE",
    "oracle": "ORCL",
    "costco": "COST",
    "salesforce": "CRM",
}

KNOWN_TICKERS = {
    "AA", "AAL", "AAPL", "ABBV", "ADBE", "ADI", "AEP", "AES", "AKR", "ALB", "ALCO", "ALKS", "AMAT", "AMD",
    "AMGN", "AMT", "AMZN", "APD", "ARE", "ASML", "ASX", "AVGO", "AWK", "AXP", "BA", "BAC", "BLK", "CAR",
    "CAT", "CCI", "CCL", "CDE", "CF", "CFG", "CHD", "CHTR", "CLX", "CMCSA", "CMPR", "CMS", "COF", "COLD",
    "COP", "COST", "CPRT", "CRM", "CRSP", "CSCO", "CVLT", "CVX", "DAL", "DASH", "DD", "DHI", "DIS", "DLR",
    "DOV", "DOW", "DUK", "DXCM", "EA", "EGP", "EMN", "EMR", "ENB", "EOG", "EQIX", "EQNR", "EQT", "ES",
    "ETN", "ETR", "EVR", "EVRG", "EXAS", "EXC", "EXR", "FANG", "FCX", "FIS", "FITB", "FLEX", "FLR", "GE",
    "GILD", "GIS", "GOOG", "GOOGL", "GS", "HAL", "HON", "HSY", "IBM", "IEX", "IFF", "INTU", "IR", "IRM",
    "ITT", "JCI", "JEF", "JNJ", "JPM", "KDP", "KLAC", "KMB", "KO", "LIN", "LLY", "LMT", "LNT", "LRCX",
    "LTC", "LVS", "LYB", "MCD", "MDLZ", "META", "MGM", "MKSI", "MNST", "MO", "MOS", "MPC", "MRK", "MSFT",
    "MU", "NEE", "NFLX", "NHI", "NKE", "NOW", "NRG", "NUE", "NVDA", "OKE", "ORCL", "OXY", "PANW", "PBA",
    "PEG", "PEP", "PFE", "PG", "PH", "PK", "PLD", "PM", "PNC", "POOL", "PPG", "PSA", "PSX", "QCOM",
    "QTRX", "RBLX", "RCL", "REG", "REGN", "REPYY", "RF", "RJF", "ROP", "RSG", "RTX", "S", "SBUX", "SCCO",
    "SCHW", "SJM", "SLB", "SNA", "SO", "SPG", "SQM", "STAG", "STZ", "T", "TAP", "TD", "TMUS", "TRGP",
    "TROW", "TRP", "TSLA", "TSN", "TTWO", "TXN", "UAL", "UL", "UNH", "USB", "VEEV", "VLO", "VRTX", "VZ",
    "WEC", "WEE", "WELL", "WFC", "WLK", "WMT", "WYNN", "XEL", "XOM", "ZM",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
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
    raw_message = str(message or "").strip()
    if not raw_message:
        return []

    found: list[tuple[int, str]] = []
    seen: set[str] = set()

    # 1. Company aliases in lowercase message (e.g. nvdia, apple, microsoft)
    lower_msg = raw_message.lower()
    for alias, sym in sorted(COMPANY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = r"\b" + re.escape(alias) + r"\b"
        for m in re.finditer(pattern, lower_msg):
            found.append((m.start(), sym))

    # 2. Uppercase tickers matching TICKER_RE (e.g. AAPL, RELIANCE.NS, BRK-B)
    for m in TICKER_RE.finditer(raw_message):
        raw = m.group(1)
        if raw == raw.upper() and not raw.isdigit():
            if len(raw) > 1 or "." in raw or "-" in raw:
                if raw not in STOP_TICKERS:
                    found.append((m.start(), raw))

    # 3. Lowercase/mixed tokens matching KNOWN_TICKERS (e.g. msft, aapl, nvda)
    for m in re.finditer(r"\b[a-zA-Z0-9.-]+\b", raw_message):
        u_tok = m.group(0).upper()
        if u_tok in KNOWN_TICKERS and u_tok not in STOP_TICKERS:
            found.append((m.start(), u_tok))

    # Sort by character position in message to preserve user request order
    found.sort(key=lambda item: item[0])
    ordered_tickers: list[str] = []
    for _, sym in found:
        if sym not in seen:
            seen.add(sym)
            ordered_tickers.append(sym)

    return ordered_tickers


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
    elif analysis_task == "price_line" and not tickers and not sector:
        return None

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
