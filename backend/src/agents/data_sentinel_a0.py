"""
Agent 0: Data Sentinel
Pre-fetches and caches rolling window price matrices for all universes into the
MongoDB blackboard_mpi collection, enabling deterministic offline pipeline runs.
Designed to run ONCE before the 51-window x 11-universe pipeline execution.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Resolve root path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from config import CONFIG
from src.blackboard.memory_store import BlackboardMemoryStore

load_dotenv()
logger = logging.getLogger(__name__)


class DataSentinelAgent:
    """
    Agent 0: Data Sentinel.

    Iterates over all 11 universes and 51 rolling windows, computing log-return
    matrices and persisting them to the MongoDB blackboard_mpi collection.
    Subsequent agents (1–4) read from this store rather than fetching live data.
    """

    ALL_UNIVERSES = [f"U{i}" for i in range(1, 12)]  # U1 through U11

    def __init__(
        self,
        mongo_uri: str | None = None,
        db_name: str = CONFIG.DB_NAME,
    ):
        self.mongo_uri = (mongo_uri or os.getenv("MONGO_URI") or "").strip()
        self.db_name = db_name
        self.blackboard = BlackboardMemoryStore(
            mongo_uri=self.mongo_uri,
            db_name=self.db_name,
        )
        self._price_cache: dict[str, pd.DataFrame] = {}

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        universes: list[str] | None = None,
        n_windows: int = CONFIG.N_WINDOWS,
        window_size: int = CONFIG.WINDOW_SIZE,
        step_size: int = CONFIG.STEP_SIZE,
        start_date: str = CONFIG.START_DATE,
        end_date: str = CONFIG.END_DATE,
    ) -> dict[str, int]:
        """
        Pre-compute all rolling windows and store in the blackboard.
        Returns a summary dict of stored window counts per universe.
        """
        target_universes = universes or self.ALL_UNIVERSES
        summary: dict[str, int] = {}

        for universe_id in target_universes:
            logger.info("[Agent 0] Processing universe %s ...", universe_id)
            try:
                count = self._process_universe(
                    universe_id, n_windows, window_size, step_size, start_date, end_date
                )
                summary[universe_id] = count
                logger.info("[Agent 0] %s: stored %d windows.", universe_id, count)
            except Exception as exc:
                logger.error("[Agent 0] Failed for %s: %s", universe_id, exc)
                summary[universe_id] = 0

        return summary

    # ── Universe Processing ───────────────────────────────────────────────────

    def _process_universe(
        self,
        universe_id: str,
        n_windows: int,
        window_size: int,
        step_size: int,
        start_date: str,
        end_date: str,
    ) -> int:
        """Fetch full universe price history and slice into rolling windows."""
        tickers = self._fetch_universe_tickers(universe_id)
        if not tickers:
            logger.warning("[Agent 0] No tickers found for %s.", universe_id)
            return 0

        prices = self._fetch_price_matrix(tickers, start_date, end_date)
        if prices.empty:
            logger.warning("[Agent 0] Empty price matrix for %s.", universe_id)
            return 0

        stored = 0
        for w_idx in range(n_windows):
            start_idx = w_idx * step_size
            end_idx = start_idx + window_size
            if end_idx > len(prices):
                break

            window_prices = prices.iloc[start_idx:end_idx].copy()
            if len(window_prices) < 20:
                continue

            log_returns = np.log(window_prices / window_prices.shift(1))
            log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")

            if log_returns.empty or log_returns.shape[1] < 2:
                continue

            window_start_date = str(window_prices.index[0].date())
            window_end_date = str(window_prices.index[-1].date())
            window_id = f"W{w_idx:03d}"

            payload: dict[str, Any] = {
                "tickers": list(log_returns.columns),
                "window_start": window_start_date,
                "window_end": window_end_date,
                "n_periods": len(log_returns),
                "returns_matrix": log_returns.to_json(),  # Serialised for MongoDB
                "prices_matrix": window_prices.to_json(),
            }

            self.blackboard.store_window(
                universe_id=universe_id,
                window_id=window_id,
                window_number=w_idx,
                data=payload,
            )
            stored += 1

        return stored

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fetch_universe_tickers(self, universe_id: str) -> list[str]:
        """Queries MongoDB for tickers in a given universe."""
        try:
            from pymongo import MongoClient
            client = MongoClient(
                self.mongo_uri,
                tls=True,
                tlsAllowInvalidCertificates=True,
                serverSelectionTimeoutMS=10000,
            )
            db = client[self.db_name]
            docs = list(db["ticker"].find(
                {"universes": universe_id},
                {"_id": 0, "ticker": 1},
            ))
            return [d["ticker"] for d in docs if d.get("ticker")]
        except Exception as exc:
            logger.error("Ticker fetch failed for %s: %s", universe_id, exc)
            return []

    def _fetch_price_matrix(
        self, tickers: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetches price data per ticker from MongoDB and assembles a price matrix."""
        series_map: dict[str, pd.Series] = {}
        try:
            from pymongo import MongoClient
            client = MongoClient(
                self.mongo_uri,
                tls=True,
                tlsAllowInvalidCertificates=True,
                serverSelectionTimeoutMS=10000,
                socketTimeoutMS=30000,
            )
            db = client[self.db_name]
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)

            for ticker in tickers:
                if ticker in self._price_cache:
                    series_map[ticker] = self._price_cache[ticker]
                    continue
                doc = db["ticker"].find_one(
                    {"ticker": ticker},
                    {
                        "_id": 0,
                        "historical_prices.Date": 1,
                        "historical_prices.Close": 1,
                    },
                )
                if not doc or not doc.get("historical_prices"):
                    continue
                rows = doc["historical_prices"]
                df = pd.DataFrame(rows)[["Date", "Close"]]
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date").sort_index()
                df = df.loc[start_ts:end_ts]
                if df.empty:
                    continue
                s = df["Close"].rename(ticker)
                self._price_cache[ticker] = s
                series_map[ticker] = s

        except Exception as exc:
            logger.error("Price matrix fetch failed: %s", exc)
            return pd.DataFrame()

        if len(series_map) < 2:
            return pd.DataFrame()

        prices = pd.concat(series_map.values(), axis=1).sort_index()
        prices = prices.ffill().dropna(how="any")
        return prices
