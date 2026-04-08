from __future__ import annotations

import logging

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


@tool
def get_price_series_for_analysis(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> dict:
    """
    Fetch daily closing prices from MongoDB and return a structured payload for
    downstream statistical analysis or custom plotting.

    The returned dictionary contains:
    - prices: ticker -> [{date, close}, ...]
    - returns: ticker -> [daily log returns]
    - stats: ticker -> summary stats
    - tickers_included / tickers_missing
    - start_date / end_date

    Use this tool whenever the user asks for correlations, return
    distributions, volatility comparisons, drawdowns, rolling volatility,
    Sharpe-style comparisons, box plots, violin plots, or other analyses that
    need the underlying time series rather than a pre-rendered line chart.
    """

    if not tickers:
        return {"error": "No tickers provided."}

    cleaned = _normalize_tickers(tickers)[:_MAX_TICKERS]
    if not cleaned:
        return {"error": "No valid tickers provided."}

    try:
        start_dt = pd.to_datetime(start_date, format="%Y-%m-%d")
        end_dt = pd.to_datetime(end_date, format="%Y-%m-%d")
    except Exception as exc:
        return {"error": f"Invalid date format: {exc}. Use YYYY-MM-DD."}

    if start_dt >= end_dt:
        return {"error": "start_date must be before end_date."}

    try:
        docs = _find_documents_with_retry(
            {"ticker": {"$in": cleaned}},
            {"ticker": 1, "historical_prices": 1},
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

    return {
        "prices": prices_out,
        "returns": returns_out,
        "stats": stats_out,
        "tickers_included": sorted(prices_out.keys()),
        "tickers_missing": missing,
        "start_date": start_date,
        "end_date": end_date,
    }
