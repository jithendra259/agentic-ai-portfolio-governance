from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from src.analytics.stock_eda.metadata import DEFAULT_TICKER_METADATA
from src.analytics.stock_eda.serialization import first_present, safe_float

logger = logging.getLogger(__name__)


def extract_ohlcv_from_db(
    tickers: list[str],
    start_date: str,
    end_date: str,
    get_collection: Callable[[], object | None],
) -> tuple[pd.DataFrame, bool, dict]:
    col = get_collection()
    if col is None:
        return pd.DataFrame(), True, {"source": "mongo_unavailable"}

    try:
        date_expr = {"$ifNull": ["$$hp.Date", "$$hp.date"]}
        docs = list(
            col.find(
                {"ticker": {"$in": tickers}},
                {
                    "_id": 0,
                    "ticker": 1,
                    "sector": 1,
                    "company": 1,
                    "longName": 1,
                    "shortName": 1,
                    "info.sector": 1,
                    "info.longName": 1,
                    "info.shortName": 1,
                    "historical_prices": {
                        "$filter": {
                            "input": "$historical_prices",
                            "as": "hp",
                            "cond": {
                                "$and": [
                                    {"$gte": [date_expr, str(start_date)]},
                                    {"$lte": [date_expr, str(end_date)]},
                                ]
                            },
                        }
                    },
                },
            ).batch_size(50).max_time_ms(20000)
        )
    except Exception as exc:
        logger.warning("Could not load OHLCV data from MongoDB: %s", exc)
        return pd.DataFrame(), True, {"source": "mongo_error", "error": str(exc)}

    rows = []
    missing_ohlcv_fields: dict[str, list[str]] = {}
    for doc in docs:
        ticker = str(doc.get("ticker", "")).upper()
        metadata = DEFAULT_TICKER_METADATA.get(ticker, {})
        info = doc.get("info") if isinstance(doc.get("info"), dict) else {}
        sector = doc.get("sector") or info.get("sector") or metadata.get("sector") or "Unknown"
        company = (
            doc.get("company")
            or doc.get("longName")
            or doc.get("shortName")
            or info.get("longName")
            or info.get("shortName")
            or metadata.get("company")
            or ticker
        )
        hist = doc.get("historical_prices", [])
        if not hist:
            continue
        frame = pd.DataFrame(hist)
        if frame.empty:
            continue
        date_col = "Date" if "Date" in frame.columns else "date" if "date" in frame.columns else None
        close_col = "Close" if "Close" in frame.columns else "close" if "close" in frame.columns else None
        if not date_col or not close_col:
            continue
        missing_fields = []
        for required in ("Open", "High", "Low", "Volume"):
            if required not in frame.columns and required.lower() not in frame.columns:
                missing_fields.append(required)
        if missing_fields:
            missing_ohlcv_fields[ticker] = missing_fields
        for _, row in frame.iterrows():
            close = safe_float(first_present(row, ["Close", "close", "Adj Close", "Adj_Close", "Adjusted"]))
            open_value = safe_float(first_present(row, ["Open", "open"], close), close)
            high = safe_float(first_present(row, ["High", "high"], max(open_value or close or 0, close or 0)), close)
            low = safe_float(first_present(row, ["Low", "low"], min(open_value or close or 0, close or 0)), close)
            volume = safe_float(first_present(row, ["Volume", "volume"], 0), 0)
            adjusted = safe_float(first_present(row, ["Adj Close", "Adj_Close", "Adjusted", "adjusted", "Close", "close"], close), close)
            rows.append(
                {
                    "date": pd.to_datetime(row[date_col], errors="coerce"),
                    "ticker": ticker,
                    "sector": sector,
                    "company": company,
                    "open": open_value,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adjusted_close": adjusted,
                    "volume": volume,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, True, {"source": "historical_prices", "missing_tickers": tickers}
    df = df.dropna(subset=["date", "close"]).sort_values(["ticker", "date"])
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    return df, False, {
        "source": "historical_prices",
        "missing_tickers": [ticker for ticker in tickers if ticker not in set(df["ticker"])],
        "missing_ohlcv_fields": missing_ohlcv_fields,
    }


def generate_fallback_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    generate_price_series: Callable[[list[str], str, str], dict[str, pd.Series]],
) -> pd.DataFrame:
    price_dict = generate_price_series(tickers, start_date, end_date)
    rows = []
    rng = np.random.default_rng(42)
    for ticker, close_series in price_dict.items():
        metadata = DEFAULT_TICKER_METADATA.get(ticker, {"sector": "Unknown", "company": ticker})
        previous_close = None
        for date_dt, close_value in close_series.items():
            close = float(close_value)
            open_value = float(previous_close if previous_close is not None else close * (1 + rng.normal(0, 0.004)))
            high = max(open_value, close) * (1 + abs(rng.normal(0.004, 0.003)))
            low = min(open_value, close) * (1 - abs(rng.normal(0.004, 0.003)))
            volume = int(max(100000, rng.normal(35_000_000, 8_000_000)))
            rows.append(
                {
                    "date": pd.to_datetime(date_dt),
                    "ticker": ticker,
                    "sector": metadata.get("sector", "Unknown"),
                    "company": metadata.get("company", ticker),
                    "open": open_value,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adjusted_close": close,
                    "volume": volume,
                }
            )
            previous_close = close
    return pd.DataFrame(rows)
