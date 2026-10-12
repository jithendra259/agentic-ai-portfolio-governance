"""
macro_benchmark_service.py - Real-world benchmarks and dynamic macro indicators.

Provides:
- Real S&P 500 (^GSPC), Nasdaq 100 (^NDX), CBOE Volatility (^VIX) data
- Dynamic risk-free rate derived from US 10Y Treasury Yield (^TNX)
- Ingestion of benchmark assets into MongoDB for persistent offline governance
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from pymongo import MongoClient

from dotenv import load_dotenv
from config import CONFIG

load_dotenv()
logger = logging.getLogger(__name__)

BENCHMARK_SYMBOLS = {
    "SP500": "^GSPC",
    "NASDAQ100": "^NDX",
    "VIX": "^VIX",
    "TNX_10Y": "^TNX",
}


def get_dynamic_risk_free_rate(default_rate: float | None = None) -> float:
    """
    Fetch the latest US 10-Year Treasury Yield (^TNX) from yfinance or cache.
    Returns annualized decimal rate (e.g. 0.0523 for 5.23%), or fallback default.
    """
    fallback = default_rate if default_rate is not None else float(getattr(CONFIG, "RISK_FREE", 0.03))
    try:
        import yfinance as yf
        tnx = yf.Ticker("^TNX").history(period="5d")
        if tnx is not None and not tnx.empty and "Close" in tnx.columns:
            latest_yield = float(tnx["Close"].dropna().iloc[-1])
            if np.isfinite(latest_yield) and 0.0 < latest_yield < 25.0:
                return round(latest_yield / 100.0, 4)
    except Exception as exc:
        logger.warning("Could not fetch live ^TNX yield; using fallback %.4f: %s", fallback, exc)
    return fallback


def get_benchmark_prices(
    benchmark_symbol: str = "^GSPC",
    start_date: str = "2024-01-01",
    end_date: str = "2026-10-08",
) -> pd.Series:
    """
    Retrieve clean historical daily closing prices for a benchmark symbol.
    Checks MongoDB first, then yfinance.
    """
    clean_sym = benchmark_symbol.strip().upper()
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # 1. Check MongoDB
    mongo_uri = (os.getenv("MONGO_URI") or "").strip()
    if mongo_uri:
        try:
            client = MongoClient(
                mongo_uri,
                tls=True,
                tlsAllowInvalidCertificates=True,
                serverSelectionTimeoutMS=3000,
            )
            col = client["Stock_data"]["ticker"]
            doc = col.find_one({"ticker": clean_sym}, {"historical_prices": 1})
            if doc and doc.get("historical_prices"):
                df = pd.DataFrame(doc["historical_prices"]).copy()
                date_col = next((c for c in df.columns if str(c).lower() in {"date", "datetime"}), None)
                close_col = next((c for c in df.columns if str(c).lower() in {"close", "adj close"}), None)
                if date_col and close_col:
                    s_date = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
                    s_close = pd.to_numeric(df[close_col], errors="coerce")
                    df = df.assign(**{date_col: s_date, close_col: s_close})
                    df = df.dropna(subset=[date_col, close_col]).sort_values(date_col)
                    df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]
                    if not df.empty and df[date_col].max() >= (end_dt - pd.Timedelta(days=7)):
                        return df.set_index(date_col)[close_col]
        except Exception as exc:
            logger.debug("Mongo benchmark lookup error for %s: %s", clean_sym, exc)

    # 2. Fetch from yfinance
    try:
        import yfinance as yf
        df_yf = yf.Ticker(clean_sym).history(start=start_date, end=end_date, auto_adjust=False)
        if df_yf is not None and not df_yf.empty:
            df_yf = df_yf.reset_index()
            date_col = next((c for c in df_yf.columns if str(c).lower() in {"date", "datetime"}), df_yf.columns[0])
            close_col = next((c for c in df_yf.columns if str(c).lower() in {"close", "adj close"}), "Close")
            df_yf[date_col] = pd.to_datetime(df_yf[date_col], errors="coerce").dt.tz_localize(None)
            df_yf[close_col] = pd.to_numeric(df_yf[close_col], errors="coerce")
            df_yf = df_yf.dropna(subset=[date_col, close_col]).sort_values(date_col)
            df_yf = df_yf.drop_duplicates(subset=[date_col], keep="last")
            df_yf = df_yf[(df_yf[date_col] >= start_dt) & (df_yf[date_col] <= end_dt)]
            if not df_yf.empty:
                return df_yf.set_index(date_col)[close_col]
    except Exception as exc:
        logger.warning("yfinance benchmark fetch failed for %s: %s", clean_sym, exc)

    return pd.Series(dtype=float)


def sync_benchmarks_to_mongo() -> dict[str, int]:
    """Ingest/update benchmark and macro symbols into MongoDB ticker collection."""
    mongo_uri = (os.getenv("MONGO_URI") or "").strip()
    if not mongo_uri:
        return {}

    import yfinance as yf
    client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True, serverSelectionTimeoutMS=10000)
    col = client["Stock_data"]["ticker"]
    results = {}

    for name, sym in BENCHMARK_SYMBOLS.items():
        try:
            t_obj = yf.Ticker(sym)
            hist = t_obj.history(period="10y", auto_adjust=False)
            if hist is None or hist.empty:
                continue

            hist = hist.reset_index()
            date_col = next((c for c in hist.columns if str(c).lower() in {"date", "datetime"}), hist.columns[0])
            hist["Date"] = pd.to_datetime(hist[date_col], errors="coerce").dt.tz_localize(None)
            hist = hist.dropna(subset=["Date"]).sort_values("Date")

            records = []
            for _, r in hist.iterrows():
                records.append({
                    "Date": r["Date"].strftime("%Y-%m-%d"),
                    "Open": round(float(r.get("Open", 0.0)), 4) if pd.notna(r.get("Open")) else 0.0,
                    "High": round(float(r.get("High", 0.0)), 4) if pd.notna(r.get("High")) else 0.0,
                    "Low": round(float(r.get("Low", 0.0)), 4) if pd.notna(r.get("Low")) else 0.0,
                    "Close": round(float(r.get("Close", 0.0)), 4) if pd.notna(r.get("Close")) else 0.0,
                    "Volume": int(r.get("Volume", 0)) if pd.notna(r.get("Volume")) else 0,
                })

            col.update_one(
                {"ticker": sym},
                {
                    "$set": {
                        "ticker": sym,
                        "symbol": sym,
                        "name": f"Benchmark: {name}",
                        "universes": ["BENCHMARK", "MACRO"],
                        "historical_prices": records,
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
                upsert=True,
            )
            results[sym] = len(records)
            logger.info("Synchronized benchmark %s (%d daily bars)", sym, len(records))
        except Exception as exc:
            logger.error("Failed to sync benchmark %s: %s", sym, exc)

    return results


# Semantic alias
get_macro_indicator_series = get_benchmark_prices
