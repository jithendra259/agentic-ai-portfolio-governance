"""Dynamic Ticker Data Service.

Provides modular, robust data fetching for any stock or asset ticker.
Dynamically retrieves live/historical data from yfinance and blends with MongoDB
cache so users are never restricted to hardcoded universes.
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

_YFINANCE_MODULE = None


def get_yfinance():
    """Lazy loader for yfinance."""
    global _YFINANCE_MODULE
    if _YFINANCE_MODULE is None:
        try:
            import yfinance as yf
            _YFINANCE_MODULE = yf
        except ImportError:
            logger.warning("yfinance is not installed in the python environment.")
            return None
    return _YFINANCE_MODULE


def normalize_ticker(ticker: Any) -> str:
    """Clean and normalize a ticker symbol."""
    if not ticker:
        return ""
    token = str(ticker).strip().upper()
    token = re.sub(r"[\$'\"]", "", token)
    return token


def normalize_ticker_list(tickers: Any) -> List[str]:
    """Parse and normalize a collection or delimited string of tickers."""
    if not tickers:
        return []
    if isinstance(tickers, str):
        raw_items = re.split(r"[,\s;]+", tickers)
    elif isinstance(tickers, (list, tuple, set)):
        raw_items = list(tickers)
    else:
        raw_items = [str(tickers)]

    seen = set()
    cleaned = []
    for item in raw_items:
        norm = normalize_ticker(item)
        if norm and norm not in seen and len(norm) <= 12:
            seen.add(norm)
            cleaned.append(norm)
    return cleaned


def history_frame_to_records(history: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a pandas OHLCV DataFrame into clean dictionary records."""
    if history is None or history.empty or "Close" not in history.columns:
        return []

    frame = history.reset_index().copy()
    if "Date" not in frame.columns:
        index_name = history.index.name or "Date"
        if index_name in frame.columns:
            frame = frame.rename(columns={index_name: "Date"})

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    for col in ["Open", "High", "Low", "Volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date")
    records = []
    has_open = "Open" in frame.columns
    has_high = "High" in frame.columns
    has_low = "Low" in frame.columns
    has_vol = "Volume" in frame.columns

    for _, row in frame.iterrows():
        rec = {
            "Date": row["Date"].strftime("%Y-%m-%d"),
            "Close": round(float(row["Close"]), 6),
        }
        if has_open and pd.notna(row["Open"]):
            rec["Open"] = round(float(row["Open"]), 6)
        if has_high and pd.notna(row["High"]):
            rec["High"] = round(float(row["High"]), 6)
        if has_low and pd.notna(row["Low"]):
            rec["Low"] = round(float(row["Low"]), 6)
        if has_vol and pd.notna(row["Volume"]):
            rec["Volume"] = int(row["Volume"])
        records.append(rec)

    return records


def fetch_yfinance_snapshot(ticker: str) -> Optional[Dict[str, Any]]:
    """Fetch live company info, fundamentals, institutional holders, and history for any ticker."""
    yf = get_yfinance()
    if yf is None:
        return None

    clean_t = normalize_ticker(ticker)
    if not clean_t:
        return None

    try:
        ticker_obj = yf.Ticker(clean_t)
        try:
            info = ticker_obj.get_info() if hasattr(ticker_obj, "get_info") else getattr(ticker_obj, "info", {})
        except Exception:
            info = {}
        if not isinstance(info, dict):
            info = {}

        history = ticker_obj.history(period="max", auto_adjust=False)
        if history is None or history.empty:
            # Retry with 5y if max returned empty
            history = ticker_obj.history(period="5y", auto_adjust=False)
    except Exception as exc:
        logger.warning("yfinance fetch failed for %s: %s", clean_t, exc)
        return None

    if history is None or history.empty:
        return None

    inst_holders = []
    try:
        ih_df = getattr(ticker_obj, "institutional_holders", None)
        if ih_df is not None and not ih_df.empty:
            for _, r in ih_df.iterrows():
                h_name = str(r.get("Holder") or r.get("holder") or "")
                pct_val = r.get("pctHeld") or r.get("% Out") or 0.0
                if h_name:
                    inst_holders.append({
                        "Holder": h_name,
                        "pctHeld": f"{float(pct_val) * 100:.2f}%" if isinstance(pct_val, (int, float)) and pct_val <= 1.0 else str(pct_val),
                    })
    except Exception:
        inst_holders = []

    key_stats = {
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "profit_margin": info.get("profitMargins"),
        "return_on_equity": info.get("returnOnEquity"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
    }
    key_stats = {k: v for k, v in key_stats.items() if v not in (None, "", "N/A")}

    info_payload = {
        "company_name": info.get("longName") or info.get("shortName") or clean_t,
        "shortName": info.get("shortName") or info.get("longName") or clean_t,
        "longName": info.get("longName") or info.get("shortName") or clean_t,
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "country": info.get("country", "Unknown"),
        "website": info.get("website", "N/A"),
        "summary": info.get("longBusinessSummary") or info.get("longSummary") or "",
    }

    return {
        "ticker": clean_t,
        "shortName": info_payload["shortName"],
        "longName": info_payload["longName"],
        "historical_prices": history_frame_to_records(history),
        "info": info_payload,
        "key_stats": key_stats,
        "financials": {},
        "graph_relationships": {
            "institutional_holders": inst_holders,
        },
        "analysis_and_estimates": {},
        "_source": "yfinance_live",
    }


def get_dynamic_price_documents(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    keep_ohlcv: bool = False,
    mongo_docs_finder: Optional[Callable[..., List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Fetch price documents for any tickers, checking MongoDB first, then dynamically using yfinance."""
    cleaned = normalize_ticker_list(tickers)
    if not cleaned:
        return []

    docs: List[Dict[str, Any]] = []

    # 1. Try MongoDB if finder callable provided
    if mongo_docs_finder is not None:
        try:
            raw_docs = mongo_docs_finder(cleaned, start_date=start_date, end_date=end_date, keep_ohlcv=keep_ohlcv)
            if raw_docs:
                docs = list(raw_docs)
        except Exception as exc:
            logger.warning("MongoDB price lookup failed for %s: %s; falling back to yfinance", cleaned, exc)
            docs = []

    # 2. Check which tickers have usable historical price data
    found_tickers = set()
    valid_docs: List[Dict[str, Any]] = []
    for d in docs:
        t = str(d.get("ticker", "")).upper()
        hp = d.get("historical_prices", [])
        if hp:
            found_tickers.add(t)
            valid_docs.append(d)

    # 3. For any missing tickers, dynamically fetch via yfinance
    missing = [t for t in cleaned if t not in found_tickers]
    for missing_ticker in missing:
        yf_doc = fetch_yfinance_snapshot(missing_ticker)
        if yf_doc and yf_doc.get("historical_prices"):
            if start_date or end_date:
                hp = yf_doc["historical_prices"]
                filtered_hp = []
                for p in hp:
                    p_date = str(p.get("Date") or p.get("date") or "")
                    if start_date and p_date < str(start_date):
                        continue
                    if end_date and p_date > str(end_date):
                        continue
                    filtered_hp.append(p)
                yf_doc["historical_prices"] = filtered_hp

            if yf_doc["historical_prices"]:
                valid_docs.append(yf_doc)
                found_tickers.add(missing_ticker)
                logger.info("Successfully fetched dynamic stock data for %s via yfinance (%d bars)", missing_ticker, len(yf_doc["historical_prices"]))

    return valid_docs
