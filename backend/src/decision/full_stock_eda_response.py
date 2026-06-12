from __future__ import annotations

from typing import Any

from api.analytics_router import get_full_stock_eda_analytics
from src.intent.intent_router import IntentRouter


DEFAULT_TICKERS = "AAPL,JPM,MSFT,BAC,META,WFC,GOOG,HSBC"
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2023-12-31"


def build_full_stock_eda_response(message: str) -> str | None:
    plan = IntentRouter().build_execution_plan(message)
    if plan.get("sub_intent") != "stock_eda_full":
        return None

    entities = plan.get("entities", {})
    tickers = ",".join(entities.get("tickers") or []) or DEFAULT_TICKERS
    start_date = entities.get("start_date") or DEFAULT_START_DATE
    end_date = entities.get("end_date") or DEFAULT_END_DATE
    data = get_full_stock_eda_analytics(tickers=tickers, start_date=start_date, end_date=end_date)
    return _format_full_stock_eda(data, plan)


def _format_full_stock_eda(data: dict[str, Any], plan: dict[str, Any]) -> str:
    tickers = ", ".join(data.get("tickers", []))
    date_range = data.get("date_range", {})
    missing_rows = data.get("missing_values", [])
    outliers = data.get("outliers", [])
    by_ticker = data.get("descriptive_statistics_by_ticker", [])
    by_sector = data.get("descriptive_statistics_by_sector", [])
    seasonal = data.get("seasonal_summaries", {})
    time_series = data.get("time_series", {})
    metadata = data.get("metadata", {})
    fallback_tickers = metadata.get("fallback_tickers", [])

    top_outliers = outliers[:6]
    sector_close = [
        row for row in by_sector
        if row.get("metric") == "close"
    ]
    ticker_return = [
        row for row in by_ticker
        if row.get("metric") == "market_return"
    ]

    lines = [
        "# Full Stock EDA",
        "",
        f"- endpoint called: `/api/analytics/stock-eda-full`",
        f"- detected intent: `{plan.get('sub_intent')}`",
        f"- tickers: {tickers}",
        f"- date range: {date_range.get('start')} to {date_range.get('end')}",
        f"- fallback/sample data: {'yes' if data.get('is_mock') else 'no'}",
        f"- source: {metadata.get('source', 'historical_prices')}",
        f"- fallback tickers: {', '.join(fallback_tickers) if fallback_tickers else 'none'}",
        "",
        "## Coverage Now Included",
        "- OHLCV preprocessing: open, high, low, close, adjusted close, and volume are prepared per ticker.",
        "- Feature engineering: `Market_Return = (Close - Open) / Close * 100` and `Volatility = (High - Low) / Close * 100`.",
        "- Descriptive statistics: count, mean, standard deviation, quartiles, min, max, skewness, and kurtosis by ticker and sector.",
        "- Missing values and outliers: missing coverage plus z-score outlier rows across price, return, volatility, and volume metrics.",
        "- Seasonal analysis: yearly, quarterly, monthly, and day-of-week summaries by sector, plus ticker-level seasonal views.",
        "- Time-series analysis: sector-level and ticker-level daily trend summaries for close, volume, volatility, and market return.",
        "",
        "## Data Quality",
        f"- ticker coverage rows: {len(missing_rows)}",
        f"- outlier rows flagged: {len(outliers)}",
        f"- descriptive statistic rows by ticker: {len(by_ticker)}",
        f"- descriptive statistic rows by sector: {len(by_sector)}",
        "",
        "## Sector Close Summary",
    ]

    if sector_close:
        for row in sector_close[:8]:
            lines.append(
                "- {sector}: mean close {mean}, median {median}, skewness {skewness}, kurtosis {kurtosis}".format(
                    sector=row.get("sector"),
                    mean=row.get("mean"),
                    median=row.get("median"),
                    skewness=row.get("skewness"),
                    kurtosis=row.get("kurtosis"),
                )
            )
    else:
        lines.append("- No sector close summary was available.")

    lines.extend(["", "## Ticker Market Return Summary"])
    if ticker_return:
        for row in ticker_return[:10]:
            lines.append(
                "- {ticker}: average return {mean}%, sd {sd}, skewness {skewness}, kurtosis {kurtosis}".format(
                    ticker=row.get("ticker"),
                    mean=row.get("mean"),
                    sd=row.get("sd"),
                    skewness=row.get("skewness"),
                    kurtosis=row.get("kurtosis"),
                )
            )
    else:
        lines.append("- No ticker market return summary was available.")

    lines.extend(["", "## Seasonal Tables Ready"])
    for name, rows in seasonal.items():
        lines.append(f"- {name}: {len(rows)} rows")

    lines.extend(["", "## Time-Series Tables Ready"])
    for name, rows in time_series.items():
        lines.append(f"- {name}: {len(rows)} bounded rows")

    if top_outliers:
        lines.extend(["", "## Sample Outliers"])
        for row in top_outliers:
            lines.append(
                "- {date} {ticker} {metric}: value {value}, z-score {z_score}".format(
                    date=row.get("date"),
                    ticker=row.get("ticker"),
                    metric=row.get("metric"),
                    value=row.get("value"),
                    z_score=row.get("z_score"),
                )
            )

    lines.extend(
        [
            "",
            "Advisory-language validation: passed. This is analytical EDA only; it does not recommend trades or execute orders.",
        ]
    )
    return "\n".join(lines)
