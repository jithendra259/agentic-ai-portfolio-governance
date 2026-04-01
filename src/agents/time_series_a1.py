import logging
from datetime import timedelta

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class TimeSeriesAgent:
    """
    Agent 1: Ingest historical price data, compute the covariance matrix,
    and derive the composite instability index from the dominant correlation eigenvalue.
    """

    def __init__(self, db_collection):
        self.collection = db_collection

    def execute(self, universe_id, target_date_str, lookback_days=90):
        print(f"[Agent 1] Analyzing {universe_id} for T={target_date_str}")

        target_date = pd.to_datetime(target_date_str)
        start_date = target_date - timedelta(days=lookback_days + 45)

        prices_df = self._fetch_price_matrix(universe_id, start_date, target_date)
        if prices_df.empty:
            raise ValueError(f"No price data found for {universe_id} ending on {target_date_str}")

        prices_df = prices_df.ffill().dropna(axis=1)
        returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
        returns_df = returns_df.tail(lookback_days)

        covariance_matrix = returns_df.cov()
        correlation_matrix = returns_df.corr()
        eigenvalues, _ = np.linalg.eigh(correlation_matrix)

        max_eigenvalue = float(np.max(eigenvalues))
        num_assets = len(returns_df.columns)
        raw_instability_index = max_eigenvalue / num_assets

        if num_assets > 1:
            lower_bound = 1.0 / num_assets
            instability_index = (raw_instability_index - lower_bound) / (1.0 - lower_bound)
        else:
            instability_index = 0.0
        instability_index = float(np.clip(instability_index, 0.0, 1.0))

        print(f"   -> Extracted {num_assets} valid assets.")
        print("   -> Computed covariance matrix (Sigma_t).")
        print(
            f"   -> Systemic instability index (I_t) = {instability_index:.4f} "
            f"(raw={raw_instability_index:.4f})"
        )
        logger.info(
            "Agent 1 instability metrics for %s at %s: raw=%0.4f normalized=%0.4f",
            universe_id,
            target_date_str,
            raw_instability_index,
            instability_index,
        )

        return {
            "returns_df": returns_df,
            "covariance_matrix": covariance_matrix,
            "instability_index": instability_index,
            "raw_instability_index": raw_instability_index,
        }

    def _fetch_price_matrix(self, universe_id, start_date, end_date):
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        cursor = self.collection.find({"universes": universe_id}, {"ticker": 1, "historical_prices": 1})

        all_prices = {}
        for doc in cursor:
            ticker = doc.get("ticker")
            history = doc.get("historical_prices", [])
            if not history:
                continue

            filtered_history = [
                row for row in history if row.get("Date") and start_str <= row["Date"] <= end_str
            ]
            if not filtered_history:
                continue

            df = pd.DataFrame(filtered_history)
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            all_prices[ticker] = df["Close"].astype(float)

        if not all_prices:
            return pd.DataFrame()

        return pd.DataFrame(all_prices)
