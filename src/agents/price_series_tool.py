from __future__ import annotations

import logging
import time
import uuid

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from src.agents.live_data_tools import (
    _extract_price_frame,
    _find_documents_with_retry,
    _normalize_tickers,
)


logger = logging.getLogger(__name__)

_MAX_TICKERS = 30
_MAX_OBSERVATIONS = 1500
_ANALYSIS_CACHE_TTL_SECONDS = 900
_ANALYSIS_CACHE_MAX_ENTRIES = 32
_ANALYSIS_CACHE: dict[str, tuple[float, dict]] = {}


def _prune_analysis_cache() -> None:
    now = time.monotonic()
    expired = [key for key, (expires_at, _) in _ANALYSIS_CACHE.items() if expires_at <= now]
    for key in expired:
        _ANALYSIS_CACHE.pop(key, None)

    if len(_ANALYSIS_CACHE) <= _ANALYSIS_CACHE_MAX_ENTRIES:
        return

    for key, _ in sorted(_ANALYSIS_CACHE.items(), key=lambda item: item[1][0]):
        _ANALYSIS_CACHE.pop(key, None)
        if len(_ANALYSIS_CACHE) <= _ANALYSIS_CACHE_MAX_ENTRIES:
            break


def _store_analysis_dataset(dataset: dict) -> str:
    _prune_analysis_cache()
    cache_key = f"analysis_{uuid.uuid4().hex}"
    _ANALYSIS_CACHE[cache_key] = (time.monotonic() + _ANALYSIS_CACHE_TTL_SECONDS, dataset)
    return cache_key


def load_cached_analysis_dataset(cache_key: str) -> dict | None:
    _prune_analysis_cache()
    cached = _ANALYSIS_CACHE.get(str(cache_key).strip())
    if not cached:
        return None
    expires_at, dataset = cached
    if expires_at <= time.monotonic():
        _ANALYSIS_CACHE.pop(str(cache_key).strip(), None)
        return None
    return dataset


@tool
def get_price_series_for_analysis(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> dict:
    """
    Fetch daily closing prices from MongoDB and return a structured payload for
    downstream statistical analysis or custom plotting.

    The returned dictionary is intentionally compact for chat-model efficiency and contains:
    - analysis_cache_key: reference to the cached full dataset
    - available_fields: fields available in the cached dataset
    - stats: ticker -> summary stats
    - tickers_included / tickers_missing
    - observations_by_ticker
    - start_date / end_date

    Use this tool whenever the user asks for correlations, return
    distributions, volatility comparisons, drawdowns, rolling volatility,
    Sharpe-style comparisons, box plots, violin plots, or other analyses that
    need the underlying time series rather than a pre-rendered line chart.
    """

    start_time = time.perf_counter()

    if not tickers:
        return {"error": "No tickers provided."}

    cleaned = _normalize_tickers(tickers)[:_MAX_TICKERS]
    if not cleaned:
        return {"error": "No valid tickers provided."}

    def _coerce_date(raw: str, end_of_period: bool = False) -> pd.Timestamp:
        """Accept YYYY, YYYY-MM, or YYYY-MM-DD. Expand short forms gracefully."""
        raw = str(raw).strip()
        if len(raw) == 4:  # just a year
            return pd.Timestamp(f"{raw}-12-31") if end_of_period else pd.Timestamp(f"{raw}-01-01")
        if len(raw) == 7:  # YYYY-MM
            if end_of_period:
                month_end = pd.Timestamp(raw).to_period("M").to_timestamp("M")
                return month_end
            return pd.Timestamp(f"{raw}-01")
        return pd.to_datetime(raw)  # full YYYY-MM-DD

    try:
        start_dt = _coerce_date(start_date, end_of_period=False)
        end_dt = _coerce_date(end_date, end_of_period=True)
    except Exception as exc:
        return {"error": f"Invalid date: {exc}. Use YYYY, YYYY-MM, or YYYY-MM-DD."}

    if start_dt >= end_dt:
        return {"error": "start_date must be before end_date."}

    try:
        docs = _find_documents_with_retry(
            {"ticker": {"$in": cleaned}},
            {
                "ticker": 1, 
                "historical_prices.Date": 1, 
                "historical_prices.date": 1, 
                "historical_prices.Close": 1, 
                "historical_prices.close": 1
            },
        )
    except Exception as exc:
        logger.warning("Price series analysis lookup failed: %s", exc)
        return {"error": f"Database error: {exc}"}

    found = {str(doc.get("ticker", "")).upper(): doc for doc in docs}

    prices_out: dict[str, list[dict[str, float | str]]] = {}
    returns_out: dict[str, list[float]] = {}
    stats_out: dict[str, dict[str, float | int]] = {}
    missing: list[str] = []

    for ticker in cleaned:
        doc = found.get(ticker)
        if not doc:
            missing.append(ticker)
            continue

        df = _extract_price_frame(doc)
        if df.empty:
            missing.append(ticker)
            continue

        filtered = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()
        if len(filtered) < 5:
            missing.append(ticker)
            continue

        if len(filtered) > _MAX_OBSERVATIONS:
            step = len(filtered) // _MAX_OBSERVATIONS + 1
            filtered = filtered.iloc[::step].copy()

        prices_out[ticker] = [
            {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "close": round(float(row["Close"]), 4),
            }
            for _, row in filtered.iterrows()
        ]

        close_arr = filtered["Close"].astype(float).to_numpy()
        log_returns = np.diff(np.log(close_arr)).tolist()
        returns_out[ticker] = [round(float(value), 6) for value in log_returns]
        
        # Capture strictly aligned dates
        p_dates = [ (row["Date"] if isinstance(row["Date"], pd.Timestamp) else pd.to_datetime(row["Date"])).strftime("%Y-%m-%d") for _, row in filtered.iterrows() ]
        # returns has N-1 points, shifted forward (start at index 1)
        r_dates = p_dates[1:]

        arr = np.array(log_returns, dtype=float)
        stats_out[ticker] = {
            "mean_return": round(float(arr.mean()), 6) if len(arr) else 0.0,
            "std_return": round(float(arr.std()), 6) if len(arr) else 0.0,
            "annualised_vol": round(float(arr.std() * np.sqrt(252)), 6) if len(arr) else 0.0,
            "total_return_pct": round((float(close_arr[-1]) / float(close_arr[0]) - 1.0) * 100.0, 4),
            "observations": len(filtered),
        }

    if not prices_out:
        return {
            "error": "No price data found for any requested ticker in the given date range.",
            "tickers_missing": missing,
        }

    full_dataset = {
        "prices": prices_out,
        "returns": returns_out,
        "stats": stats_out,
        "tickers_included": sorted(prices_out.keys()),
        "tickers_missing": missing,
        "start_date": start_date,
        "end_date": end_date,
        "price_dates": p_dates if 'p_dates' in locals() else [],
        "return_dates": r_dates if 'r_dates' in locals() else [],
    }
    cache_key = _store_analysis_dataset(full_dataset)
    elapsed_seconds = time.perf_counter() - start_time
    logger.info(
        "Prepared cached price-series analysis for %s tickers over %s to %s in %.2fs",
        len(full_dataset["tickers_included"]),
        start_date,
        end_date,
        elapsed_seconds,
    )

    return {
        "analysis_cache_key": cache_key,
        "available_fields": {
            "prices": "ticker -> [{date, close}, ...]",
            "returns": "ticker -> [daily log returns]",
            "stats": "ticker -> {mean_return, std_return, annualised_vol, total_return_pct, observations}",
        },
        "stats": stats_out,
        "tickers_included": full_dataset["tickers_included"],
        "tickers_missing": missing,
        "observations_by_ticker": {
            ticker: details.get("observations", 0)
            for ticker, details in stats_out.items()
        },
        "start_date": start_date,
        "end_date": end_date,
        "price_dates": full_dataset["price_dates"],
        "return_dates": full_dataset["return_dates"],
    }
