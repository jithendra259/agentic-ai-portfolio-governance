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

    def __init__(
        self,
        db_collection,
        missing_threshold=0.05,
        ffill_limit=3,
        winsorize_limits=(0.01, 0.01),
        min_assets=2,
        min_observations=20,
    ):
        self.collection = db_collection
        self.missing_threshold = float(missing_threshold)
        self.ffill_limit = int(ffill_limit)
        self.winsorize_limits = winsorize_limits
        self.min_assets = int(min_assets)
        self.min_observations = int(min_observations)

    def execute(self, universe_id, target_date_str, lookback_days=90):
        print(f"[Agent 1] Analyzing {universe_id} for T={target_date_str}")

        target_date = pd.to_datetime(target_date_str)
        start_date = target_date - timedelta(days=lookback_days + 45)

        prices_df = self._fetch_price_matrix(universe_id, start_date, target_date)
        if prices_df.empty:
            raise ValueError(f"No price data found for {universe_id} ending on {target_date_str}")

        prepared_prices_df, dropped_assets = self._prepare_price_matrix(prices_df)
        returns_df = self._compute_returns(prepared_prices_df, lookback_days=lookback_days)

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
        mean_volatility = float(returns_df.std().mean() * np.sqrt(252.0))
        mean_correlation = self._mean_pairwise_correlation(correlation_matrix)
        mean_drawdown = self._mean_drawdown(returns_df)
        retained_assets = returns_df.columns.tolist()

        print(f"   -> Extracted {num_assets} valid assets.")
        print(f"   -> Dropped assets during quality control: {dropped_assets or 'None'}")
        print("   -> Computed covariance matrix (Sigma_t).")
        print(
            f"   -> Systemic instability index (I_t) = {instability_index:.4f} "
            f"(raw={raw_instability_index:.4f})"
        )
        print(
            f"   -> Diagnostics: vol={mean_volatility:.4f}, "
            f"corr={mean_correlation:.4f}, drawdown={mean_drawdown:.4f}"
        )
        logger.info(
            "Agent 1 metrics for %s at %s: raw=%0.4f normalized=%0.4f vol=%0.4f corr=%0.4f dd=%0.4f dropped=%s",
            universe_id,
            target_date_str,
            raw_instability_index,
            instability_index,
            mean_volatility,
            mean_correlation,
            mean_drawdown,
            dropped_assets,
        )

        return {
            "returns_df": returns_df,
            "covariance_matrix": covariance_matrix,
            "instability_index": instability_index,
            "raw_instability_index": raw_instability_index,
            "retained_assets": retained_assets,
            "dropped_assets": dropped_assets,
            "mean_volatility": mean_volatility,
            "mean_correlation": mean_correlation,
            "mean_drawdown": mean_drawdown,
        }

    def _prepare_price_matrix(self, prices_df):
        working_df = prices_df.sort_index().copy()
        dropped_assets = []

        for ticker in list(working_df.columns):
            missing_pct = float(working_df[ticker].isna().mean())
            if missing_pct > self.missing_threshold:
                dropped_assets.append(ticker)

        if dropped_assets:
            working_df = working_df.drop(columns=dropped_assets, errors="ignore")

        if working_df.shape[1] < self.min_assets:
            raise ValueError(
                "Not enough assets remained after missing-data filtering to continue the analysis."
            )

        working_df = working_df.ffill(limit=self.ffill_limit)
        working_df = working_df.dropna(axis=1)

        if working_df.shape[1] < self.min_assets:
            raise ValueError(
                "Not enough assets remained after forward-filling and residual gap removal."
            )

        return working_df, dropped_assets

    def _compute_returns(self, prices_df, lookback_days):
        returns_df = np.log(prices_df / prices_df.shift(1))
        returns_df = returns_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        returns_df = returns_df.tail(lookback_days)

        if len(returns_df) < self.min_observations:
            raise ValueError(
                f"Only {len(returns_df)} clean observations remained after preprocessing; "
                f"at least {self.min_observations} are required."
            )

        lower_limit, upper_limit = self.winsorize_limits
        for column in returns_df.columns:
            series = returns_df[column]
            lower = float(series.quantile(lower_limit))
            upper = float(series.quantile(1.0 - upper_limit))
            returns_df[column] = series.clip(lower=lower, upper=upper)

        return returns_df

    def _mean_pairwise_correlation(self, correlation_matrix):
        if correlation_matrix.shape[0] < 2:
            return 0.0

        tri = correlation_matrix.to_numpy()[np.triu_indices(correlation_matrix.shape[0], k=1)]
        tri = tri[np.isfinite(tri)]
        if len(tri) == 0:
            return 0.0
        return float(np.mean(tri))

    def _mean_drawdown(self, returns_df):
        cumulative = np.exp(returns_df.cumsum())
        running_peak = cumulative.cummax()
        drawdown = (cumulative - running_peak) / running_peak.replace(0.0, np.nan)
        drawdown = drawdown.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return float(np.abs(drawdown.min()).mean())

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
