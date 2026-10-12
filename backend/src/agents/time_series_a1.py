import logging
from datetime import timedelta

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from config import CONFIG
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
        num_assets = len(returns_df.columns)
        mean_volatility = float(returns_df.std().mean() * np.sqrt(252.0))
        mean_correlation = self._mean_pairwise_correlation(correlation_matrix)
        mean_drawdown = self._mean_drawdown(returns_df)

        vol_norm = float(np.clip((mean_volatility - 0.10) / (0.60 - 0.10), 0.0, 1.0))
        corr_norm = float(np.clip((mean_correlation - 0.10) / (0.80 - 0.10), 0.0, 1.0))
        dd_norm = float(np.clip((mean_drawdown - 0.05) / (0.40 - 0.05), 0.0, 1.0))

        vix_norm = None
        latest_vix = None
        try:
            from src.analytics.macro_benchmark_service import get_benchmark_prices
            vix_series = get_benchmark_prices("^VIX", start_date.strftime("%Y-%m-%d"), target_date_str)
            if vix_series is not None and not vix_series.empty:
                latest_vix = float(vix_series.dropna().iloc[-1])
                vix_norm = float(np.clip((latest_vix - 12.0) / (35.0 - 12.0), 0.0, 1.0))
        except Exception:
            pass

        if vix_norm is not None:
            raw_instability_index = 0.35 * vol_norm + 0.25 * corr_norm + 0.25 * dd_norm + 0.15 * vix_norm
        else:
            raw_instability_index = 0.4 * vol_norm + 0.3 * corr_norm + 0.3 * dd_norm
        instability_index = float(np.clip(raw_instability_index, 0.0, 1.0))
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
            + (f", vix={latest_vix:.2f}" if latest_vix is not None else "")
        )
        logger.info(
            "Agent 1 metrics for %s at %s: raw=%0.4f normalized=%0.4f vol=%0.4f corr=%0.4f dd=%0.4f vix=%s dropped=%s",
            universe_id,
            target_date_str,
            raw_instability_index,
            instability_index,
            mean_volatility,
            mean_correlation,
            mean_drawdown,
            str(latest_vix),
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
            "latest_vix": latest_vix,
            "vix_norm": vix_norm,
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

        from src.agents.ticker_data_service import normalize_ticker_list, get_dynamic_price_documents

        tickers = normalize_ticker_list(universe_id) if isinstance(universe_id, (list, tuple, set)) or "," in str(universe_id) else []
        if tickers:
            docs = get_dynamic_price_documents(tickers, start_date=start_str, end_date=end_str)
        else:
            docs = list(self.collection.find({"universes": universe_id}, {"ticker": 1, "historical_prices": 1}))
            if not docs:
                docs = get_dynamic_price_documents([str(universe_id)], start_date=start_str, end_date=end_str)

        all_prices = {}
        for doc in docs:
            ticker = doc.get("ticker")
            history = doc.get("historical_prices", [])
            if not history:
                continue

            filtered_history = [
                row for row in history
                if (row.get("Date") or row.get("date")) and start_str <= str(row.get("Date") or row.get("date")) <= end_str
            ]
            if not filtered_history:
                continue

            df = pd.DataFrame(filtered_history).copy()
            date_col = next((c for c in df.columns if str(c).lower() in {"date", "datetime"}), None)
            close_col = next((c for c in df.columns if str(c).lower() in {"adj close", "adj_close", "close"}), None)
            if not date_col or not close_col:
                continue

            s_date = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
            s_close = pd.to_numeric(df[close_col], errors="coerce")
            df = df.assign(**{date_col: s_date, close_col: s_close}).dropna(subset=[date_col, close_col])
            df = df.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
            if not df.empty:
                all_prices[ticker] = df.set_index(date_col)[close_col]

        if not all_prices:
            return pd.DataFrame()

        return pd.DataFrame(all_prices)
