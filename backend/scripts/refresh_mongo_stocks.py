"""
refresh_mongo_stocks.py - Sync MongoDB Stock Tickers with Latest Market Data from yfinance.

Pulls recent daily OHLCV trading bars from yfinance and appends them to the
'historical_prices' array in the MongoDB 'ticker' collection, ensuring that
stocks have complete and up-to-date market history.

Usage:
    python refresh_mongo_stocks.py --universes U1 U2 U3 U4 U5
    python refresh_mongo_stocks.py --all
    python refresh_mongo_stocks.py --tickers AAPL MSFT NVDA
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pymongo
from dotenv import load_dotenv

# Ensure backend root is in python path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

load_dotenv(backend_dir.parent / ".env")
load_dotenv(backend_dir / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("refresh_stocks")


def get_mongo_collection() -> pymongo.collection.Collection:
    mongo_uri = (os.getenv("MONGO_URI") or "").strip()
    if not mongo_uri:
        raise ValueError("MONGO_URI is missing from environment.")
    client = pymongo.MongoClient(
        mongo_uri,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=10000,
    )
    return client["Stock_data"]["ticker"]


def get_tickers_to_update(
    collection: pymongo.collection.Collection,
    universes: list[str] | None = None,
    tickers: list[str] | None = None,
    update_all: bool = False,
) -> list[str]:
    if tickers:
        return [t.strip().upper() for t in tickers if t.strip()]

    query = {}
    if universes and not update_all:
        norm_univ = [u.strip().upper() for u in universes if u.strip()]
        query = {"universes": {"$in": norm_univ}}

    docs = collection.find(query, {"ticker": 1, "symbol": 1}).sort("ticker", 1)
    result = []
    for d in docs:
        t = (d.get("ticker") or d.get("symbol") or "").strip().upper()
        if t and t not in result:
            result.append(t)
    return result


def refresh_ticker(
    ticker: str,
    collection: pymongo.collection.Collection,
    yf_module,
) -> tuple[int, str]:
    """
    Fetch missing history for a single ticker and append to MongoDB.
    Returns (num_bars_added, status_message).
    """
    doc = collection.find_one({"ticker": ticker}, {"historical_prices": 1})
    if not doc:
        return 0, f"Ticker {ticker} not found in MongoDB."

    hp = doc.get("historical_prices", [])
    last_date_str = None
    if hp:
        last_rec = hp[-1]
        last_date_str = last_rec.get("Date") or last_rec.get("date")

    if last_date_str:
        try:
            last_dt = pd.to_datetime(last_date_str)
            start_fetch_dt = last_dt + pd.Timedelta(days=1)
            start_str = start_fetch_dt.strftime("%Y-%m-%d")
        except Exception:
            start_str = "2026-01-01"
    else:
        start_str = "2015-01-01"

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if start_str >= today_str:
        return 0, "Already up to date."

    try:
        t_obj = yf_module.Ticker(ticker)
        df = t_obj.history(start=start_str, auto_adjust=False)
        if df is None or df.empty:
            return 0, "No new trading bars available on yfinance."

        df = df.reset_index()
        date_col = next((c for c in df.columns if str(c).lower() in {"date", "datetime"}), df.columns[0])
        df["Date"] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["Date"]).sort_values("Date")

        new_records = []
        for _, row in df.iterrows():
            d_str = row["Date"].strftime("%Y-%m-%d")
            # Don't duplicate if already exists
            if last_date_str and d_str <= last_date_str:
                continue
            rec = {
                "Date": d_str,
                "Open": round(float(row.get("Open", 0.0)), 6) if pd.notna(row.get("Open")) else 0.0,
                "High": round(float(row.get("High", 0.0)), 6) if pd.notna(row.get("High")) else 0.0,
                "Low": round(float(row.get("Low", 0.0)), 6) if pd.notna(row.get("Low")) else 0.0,
                "Close": round(float(row.get("Close", 0.0)), 6) if pd.notna(row.get("Close")) else 0.0,
                "Volume": int(row.get("Volume", 0)) if pd.notna(row.get("Volume")) else 0,
            }
            new_records.append(rec)

        if not new_records:
            return 0, "Up to date after alignment."

        collection.update_one(
            {"ticker": ticker},
            {
                "$push": {"historical_prices": {"$each": new_records}},
                "$set": {
                    "updated_at": datetime.now(timezone.utc),
                    "last_market_date": new_records[-1]["Date"],
                },
            },
        )
        return len(new_records), f"Added {len(new_records)} bars (up to {new_records[-1]['Date']})"

    except Exception as exc:
        return 0, f"Error: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Refresh MongoDB stocks with latest yfinance data.")
    parser.add_argument("--universes", nargs="+", default=["U1", "U2", "U3", "U4", "U5"], help="Universes to update (e.g. U1 U2). Default: U1..U5")
    parser.add_argument("--all", action="store_true", help="Update all tickers in database.")
    parser.add_argument("--tickers", nargs="+", default=None, help="Explicit list of tickers to update.")
    args = parser.parse_args()

    import yfinance as yf

    logger.info("Connecting to MongoDB...")
    col = get_mongo_collection()
    logger.info("Connected successfully.")

    target_tickers = get_tickers_to_update(
        col,
        universes=args.universes,
        tickers=args.tickers,
        update_all=args.all,
    )
    logger.info("Found %d tickers to process.", len(target_tickers))

    success_count = 0
    total_bars_added = 0
    start_time = time.time()

    for idx, ticker in enumerate(target_tickers, 1):
        bars_added, msg = refresh_ticker(ticker, col, yf)
        total_bars_added += bars_added
        if bars_added > 0:
            success_count += 1
            logger.info("[%d/%d] %-6s: %s", idx, len(target_tickers), ticker, msg)
        else:
            logger.debug("[%d/%d] %-6s: %s", idx, len(target_tickers), ticker, msg)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Update Complete in %.2fs!", elapsed)
    logger.info("Tickers processed: %d | Updated: %d | Total new bars added: %d", len(target_tickers), success_count, total_bars_added)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
