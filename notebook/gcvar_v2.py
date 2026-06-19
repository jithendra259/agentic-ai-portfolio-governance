"""Supplemental Linear-Centrality Adaptive G-CVaR V2 utilities.

This module is intentionally separate from ``gcvar_protocol.py`` so the
completed quadratic G-CVaR thesis protocol remains frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdaptiveGateSignal:
    instability: float
    threshold: float
    steepness: float
    multiplier: float
    effective_lambda: float
    active: bool


def _normalize_unit_interval(values: pd.Series) -> pd.Series:
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    series = series.fillna(0.0)
    if series.empty:
        return series
    low = float(series.min())
    high = float(series.max())
    if high > low:
        return (series - low) / (high - low)
    series.loc[:] = 0.0
    return series


def load_clean_13f_holdings(
    path: str | Path,
    publication_lag_days: int = 45,
) -> pd.DataFrame:
    """Load a local 13F-style holdings file without downloading anything."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(
            columns=["filing_date", "manager_id", "ticker", "market_value"]
        )

    holdings = pd.read_csv(path)
    required = {"manager_id", "ticker", "market_value"}
    missing = required.difference(holdings.columns)
    if missing:
        raise ValueError(f"Missing required 13F columns: {sorted(missing)}")

    if "filing_date" not in holdings.columns:
        if "report_date" not in holdings.columns:
            raise ValueError("Need either filing_date or report_date in holdings CSV.")
        holdings["report_date"] = pd.to_datetime(holdings["report_date"])
        holdings["filing_date"] = holdings["report_date"] + pd.Timedelta(
            days=publication_lag_days
        )

    holdings["filing_date"] = pd.to_datetime(holdings["filing_date"])
    holdings["manager_id"] = holdings["manager_id"].astype(str).str.strip()
    holdings["ticker"] = holdings["ticker"].astype(str).str.upper().str.strip()
    holdings["market_value"] = pd.to_numeric(
        holdings["market_value"], errors="coerce"
    )
    holdings = holdings.dropna(
        subset=["filing_date", "manager_id", "ticker", "market_value"]
    )
    holdings = holdings.loc[holdings["market_value"] > 0].copy()
    return holdings.sort_values(
        ["filing_date", "manager_id", "ticker"]
    ).reset_index(drop=True)


def build_holdings_matrix(
    holdings: pd.DataFrame,
    tickers: Iterable[str],
    asof_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build manager-by-ticker exposures using only public filings."""
    target = [str(ticker).upper().strip() for ticker in tickers]
    if holdings is None or pd.DataFrame(holdings).empty or not target:
        return pd.DataFrame(columns=target)

    frame = pd.DataFrame(holdings).copy()
    frame["filing_date"] = pd.to_datetime(frame["filing_date"])
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["manager_id"] = frame["manager_id"].astype(str).str.strip()
    frame["market_value"] = pd.to_numeric(frame["market_value"], errors="coerce")

    asof_date = pd.to_datetime(asof_date)
    frame = frame.loc[
        (frame["filing_date"] <= asof_date)
        & (frame["ticker"].isin(target))
        & (frame["market_value"] > 0)
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=target)

    latest = frame["filing_date"].max()
    frame = frame.loc[frame["filing_date"].eq(latest)]
    matrix = frame.pivot_table(
        index="manager_id",
        columns="ticker",
        values="market_value",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(columns=target, fill_value=0.0)
    row_sum = matrix.sum(axis=1).replace(0, np.nan)
    matrix = matrix.div(row_sum, axis=0).fillna(0.0)
    return matrix.loc[matrix.sum(axis=1) > 0]


def _centrality_from_adjacency(
    adjacency: pd.DataFrame,
    target_tickers: Iterable[str],
    name: str,
) -> pd.Series:
    tickers = list(target_tickers)
    adjacency = pd.DataFrame(adjacency).reindex(
        index=tickers, columns=tickers, fill_value=0.0
    )
    if adjacency.empty or float(np.abs(adjacency.to_numpy()).sum()) <= 0:
        return pd.Series(0.0, index=tickers, name=name)
    graph = nx.from_pandas_adjacency(adjacency)
    try:
        centrality = nx.eigenvector_centrality_numpy(graph, weight="weight")
    except Exception:
        centrality = dict(graph.degree(weight="weight"))
    penalty = pd.Series(centrality, dtype=float).reindex(tickers).fillna(0.0)
    penalty = _normalize_unit_interval(penalty)
    penalty.name = name
    return penalty


def institutional_centrality_penalty(
    holdings_matrix: pd.DataFrame,
    tickers: Iterable[str],
) -> pd.Series:
    tickers = list(tickers)
    matrix = pd.DataFrame(holdings_matrix).reindex(columns=tickers, fill_value=0.0)
    if matrix.empty:
        return pd.Series(0.0, index=tickers, name="institutional_graph_penalty")
    adjacency_values = matrix.T.to_numpy() @ matrix.to_numpy()
    np.fill_diagonal(adjacency_values, 0.0)
    adjacency = pd.DataFrame(adjacency_values, index=tickers, columns=tickers)
    return _centrality_from_adjacency(
        adjacency, tickers, "institutional_graph_penalty"
    )


def correlation_centrality_penalty(
    returns: pd.DataFrame,
    threshold: float = 0.30,
) -> pd.Series:
    clean = pd.DataFrame(returns).dropna(how="any")
    tickers = list(clean.columns)
    if clean.empty:
        return pd.Series(dtype=float, name="correlation_graph_penalty")
    adjacency = clean.corr().abs().fillna(0.0)
    adjacency = adjacency.where(adjacency >= float(threshold), 0.0)
    values = adjacency.to_numpy(copy=True)
    np.fill_diagonal(values, 0.0)
    adjacency = pd.DataFrame(values, index=tickers, columns=tickers)
    return _centrality_from_adjacency(adjacency, tickers, "correlation_graph_penalty")


def get_linear_graph_penalty(
    returns: pd.DataFrame,
    tickers: Iterable[str],
    asof_date: pd.Timestamp,
    holdings: pd.DataFrame | None,
    threshold: float = 0.30,
) -> tuple[pd.Series, str]:
    target = [str(ticker).upper().strip() for ticker in tickers]
    if holdings is not None and not pd.DataFrame(holdings).empty:
        matrix = build_holdings_matrix(holdings, target, asof_date)
        penalty = institutional_centrality_penalty(matrix, target)
        if not penalty.empty and float(penalty.abs().sum()) > 0:
            return penalty.reindex(target).fillna(0.0), "sec_13f_institutional_coownership"

    penalty = correlation_centrality_penalty(returns, threshold=threshold)
    penalty = penalty.reindex(target).fillna(0.0)
    return penalty, "correlation_proxy"


def expanding_zscore(values: pd.Series, min_periods: int = 60) -> pd.Series:
    series = pd.Series(values, dtype=float)
    mean = series.expanding(min_periods=min_periods).mean().shift(1)
    std = series.expanding(min_periods=min_periods).std().shift(1)
    zscore = (series - mean) / std.replace(0, np.nan)
    return zscore.replace([np.inf, -np.inf], np.nan)


def compute_instability_series(
    returns: pd.DataFrame,
    window: int = 126,
    alpha_drift: float = 0.40,
    alpha_vol: float = 0.30,
    alpha_corr: float = 0.30,
) -> pd.DataFrame:
    clean = pd.DataFrame(returns).dropna(how="any")
    if len(clean) < window:
        return pd.DataFrame(index=clean.index)
    dates: list[pd.Timestamp] = []
    covariance_drift: list[float] = []
    correlation_stress: list[float] = []
    previous_covariance: np.ndarray | None = None

    for end_position in range(window, len(clean) + 1):
        block = clean.iloc[end_position - window:end_position]
        dates.append(block.index[-1])
        covariance = block.cov().to_numpy()
        covariance_drift.append(
            np.nan
            if previous_covariance is None
            else float(np.linalg.norm(covariance - previous_covariance, ord="fro"))
        )
        previous_covariance = covariance
        correlation = block.corr().to_numpy()
        mask = ~np.eye(correlation.shape[0], dtype=bool)
        correlation_stress.append(float(np.nanmean(correlation[mask])))

    drift = pd.Series(covariance_drift, index=dates, name="covariance_drift")
    corr = pd.Series(correlation_stress, index=dates, name="correlation_stress")
    volatility = clean.rolling(window).std().mean(axis=1).reindex(drift.index)
    volatility.name = "rolling_volatility"
    z_drift = expanding_zscore(drift).rename("z_drift")
    z_vol = expanding_zscore(volatility).rename("z_vol")
    z_corr = expanding_zscore(corr).rename("z_corr")
    instability = (
        alpha_drift * z_drift + alpha_vol * z_vol + alpha_corr * z_corr
    ).rename("instability_index")
    return pd.concat(
        [drift, volatility, corr, z_drift, z_vol, z_corr, instability], axis=1
    )


def calibrate_gate(
    instability: pd.Series,
    validation_start: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    target_frequency: float = 0.10,
) -> tuple[float, float]:
    series = pd.Series(instability, dtype=float).dropna().sort_index()
    validation = series.loc[pd.Timestamp(validation_start):pd.Timestamp(validation_end)]
    source = validation if len(validation) >= 30 else series
    if source.empty:
        return 0.0, 1.0
    theta = float(source.quantile(1.0 - float(target_frequency)))
    iqr = float(source.quantile(0.75) - source.quantile(0.25))
    steepness = float(np.log(9.0) / max(iqr, 1e-6))
    return theta, steepness


def adaptive_gate_signal(
    instability: float,
    theta: float,
    steepness: float,
    graph_lambda: float,
) -> AdaptiveGateSignal:
    if not np.isfinite(instability):
        multiplier = 0.0
    else:
        multiplier = float(
            1.0 / (1.0 + np.exp(-float(steepness) * (float(instability) - float(theta))))
        )
    multiplier = float(np.clip(multiplier, 0.0, 1.0))
    effective = float(graph_lambda) * multiplier
    return AdaptiveGateSignal(
        instability=float(instability) if np.isfinite(instability) else np.nan,
        threshold=float(theta),
        steepness=float(steepness),
        multiplier=multiplier,
        effective_lambda=effective,
        active=bool(multiplier > 0.5),
    )
