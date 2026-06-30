from __future__ import annotations

import logging
import time
import uuid

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from src.agents.live_data_tools import (
    _cached_yfinance_history_frame,
    _extract_price_frame,
    _find_price_documents_with_retry,
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
    Fetch daily OHLCV prices from MongoDB and return a compact structured
    payload for downstream statistical analysis or custom plotting.

    Stats are computed from the full filtered daily history. Long price series
    are downsampled only for cached plotting payloads and never for return math.
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
        if len(raw) == 4:
            return pd.Timestamp(f"{raw}-12-31") if end_of_period else pd.Timestamp(f"{raw}-01-01")
        if len(raw) == 7:
            if end_of_period:
                return pd.Timestamp(raw).to_period("M").to_timestamp("M")
            return pd.Timestamp(f"{raw}-01")
        return pd.to_datetime(raw)

    try:
        start_dt = _coerce_date(start_date, end_of_period=False)
        end_dt = _coerce_date(end_date, end_of_period=True)
    except Exception as exc:
        return {"error": f"Invalid date: {exc}. Use YYYY, YYYY-MM, or YYYY-MM-DD."}

    if start_dt >= end_dt:
        return {"error": "start_date must be before end_date."}

    try:
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")
        docs = _find_price_documents_with_retry(
            cleaned,
            start_date=start_str,
            end_date=end_str,
            keep_ohlcv=True,
        )
    except Exception as exc:
        logger.warning("Price series analysis lookup failed: %s", exc)
        return {"error": f"Database error: {exc}"}

    found = {str(doc.get("ticker", "")).upper(): doc for doc in docs}

    prices_out: dict[str, list[dict[str, float | str]]] = {}
    returns_out: dict[str, list[float]] = {}
    price_dates_by_ticker: dict[str, list[str]] = {}
    return_dates_by_ticker: dict[str, list[str]] = {}
    stats_out: dict[str, dict[str, float | int | str]] = {}
    missing: list[str] = []

    for ticker in cleaned:
        doc = found.get(ticker)
        if not doc:
            df = _cached_yfinance_history_frame(ticker, start_str, end_str)
        else:
            df = _extract_price_frame(doc, keep_ohlcv=True)
        if df.empty:
            missing.append(ticker)
            continue

        full_filtered = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()
        if len(full_filtered) < 5:
            missing.append(ticker)
            continue

        full_filtered = full_filtered.sort_values("Date")
        full_close_arr = full_filtered["Close"].astype(float).to_numpy()
        full_log_returns = np.diff(np.log(full_close_arr)).tolist()

        sampled = full_filtered
        # Full data requested - bypassing downsampling

        prices_out[ticker] = [
            {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 4) if "Open" in row and not pd.isna(row["Open"]) else (round(float(row["open"]), 4) if "open" in row and not pd.isna(row["open"]) else None),
                "high": round(float(row["High"]), 4) if "High" in row and not pd.isna(row["High"]) else (round(float(row["high"]), 4) if "high" in row and not pd.isna(row["high"]) else None),
                "low": round(float(row["Low"]), 4) if "Low" in row and not pd.isna(row["Low"]) else (round(float(row["low"]), 4) if "low" in row and not pd.isna(row["low"]) else None),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]) if "Volume" in row and not pd.isna(row["Volume"]) else (int(row["volume"]) if "volume" in row and not pd.isna(row["volume"]) else None),
            }
            for _, row in sampled.iterrows()
        ]
        returns_out[ticker] = [round(float(value), 6) for value in full_log_returns]

        full_price_dates = [
            (row["Date"] if isinstance(row["Date"], pd.Timestamp) else pd.to_datetime(row["Date"], utc=True).tz_localize(None)).strftime("%Y-%m-%d")
            for _, row in full_filtered.iterrows()
        ]
        sampled_price_dates = [
            (row["Date"] if isinstance(row["Date"], pd.Timestamp) else pd.to_datetime(row["Date"], utc=True).tz_localize(None)).strftime("%Y-%m-%d")
            for _, row in sampled.iterrows()
        ]
        price_dates_by_ticker[ticker] = sampled_price_dates
        return_dates_by_ticker[ticker] = full_price_dates[1:]

        arr = np.array(full_log_returns, dtype=float)
        running_max = np.maximum.accumulate(full_close_arr)
        drawdowns = (full_close_arr / running_max) - 1.0
        max_drawdown_pct = float(drawdowns.min() * 100.0) if len(drawdowns) else 0.0
        calendar_days = max(
            (
                full_filtered["Date"].iloc[-1].to_pydatetime()
                - full_filtered["Date"].iloc[0].to_pydatetime()
            ).days,
            1,
        )
        calendar_years = max(calendar_days / 365.25, 1 / 365.25)
        cagr_pct = ((float(full_close_arr[-1]) / float(full_close_arr[0])) ** (1.0 / calendar_years) - 1.0) * 100.0
        stats_out[ticker] = {
            "mean_return": round(float(arr.mean()), 6) if len(arr) else 0.0,
            "std_return": round(float(arr.std()), 6) if len(arr) else 0.0,
            "annualised_vol": round(float(arr.std() * np.sqrt(252)), 6) if len(arr) else 0.0,
            "total_return_pct": round((float(full_close_arr[-1]) / float(full_close_arr[0]) - 1.0) * 100.0, 4),
            "cagr_pct": round(cagr_pct, 4),
            "max_drawdown_pct": round(max_drawdown_pct, 4),
            "trading_days": len(full_filtered),
            "observations": len(full_filtered),
            "sampled_observations": len(sampled),
            "first_price_date": full_price_dates[0],
            "last_price_date": full_price_dates[-1],
            "first_close": round(float(full_close_arr[0]), 6),
            "last_close": round(float(full_close_arr[-1]), 6),
        }

    if not prices_out:
        return {
            "error": "No price data found for any requested ticker in the given date range.",
            "tickers_missing": missing,
        }

    aligned_price_dates: list[str] = []
    aligned_return_dates: list[str] = []
    if prices_out:
        aligned_series = [
            pd.Series(
                [row["close"] for row in rows],
                index=pd.to_datetime([row["date"] for row in rows]),
                name=ticker,
            )
            for ticker, rows in prices_out.items()
            if rows
        ]
        common_start = max(series.index.min() for series in aligned_series)
        common_end = min(series.index.max() for series in aligned_series)
        aligned_prices = pd.concat(aligned_series, axis=1).sort_index()
        aligned_prices = aligned_prices[
            (aligned_prices.index >= common_start) & (aligned_prices.index <= common_end)
        ]
        aligned_prices = aligned_prices.ffill().dropna(how="any")
        aligned_price_dates = [date.strftime("%Y-%m-%d") for date in aligned_prices.index]
        aligned_return_dates = aligned_price_dates[1:]

    full_dataset = {
        "prices": prices_out,
        "returns": returns_out,
        "stats": stats_out,
        "tickers_included": sorted(prices_out.keys()),
        "tickers_missing": missing,
        "start_date": start_date,
        "end_date": end_date,
        "price_dates": aligned_price_dates,
        "return_dates": aligned_return_dates,
        "price_dates_by_ticker": price_dates_by_ticker,
        "return_dates_by_ticker": return_dates_by_ticker,
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
            "prices": "ticker -> [{date, open, high, low, close, volume}, ...] (downsampled for long ranges)",
            "returns": "ticker -> [daily log returns] (full series in cache)",
            "stats": "ticker -> summary stats computed from full daily history",
        },
        "stats": stats_out,
        "tickers_included": full_dataset["tickers_included"],
        "tickers_missing": missing,
        "observations_by_ticker": {
            ticker: details.get("observations", 0)
            for ticker, details in stats_out.items()
        },
        "sampled_observations_by_ticker": {
            ticker: details.get("sampled_observations", 0)
            for ticker, details in stats_out.items()
        },
        "start_date": start_date,
        "end_date": end_date,
        "date_range_used": {
            "start": aligned_price_dates[0] if aligned_price_dates else None,
            "end": aligned_price_dates[-1] if aligned_price_dates else None,
        },
    }
