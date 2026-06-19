"""Supplemental Linear-Centrality Adaptive G-CVaR V2 utilities.

This module is intentionally separate from ``gcvar_protocol.py`` so the
completed quadratic G-CVaR thesis protocol remains frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cvxpy as cp
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


@dataclass(frozen=True)
class LinearGCVarParams:
    alpha: float = 0.95
    max_weight: float = 0.30
    return_tradeoff: float = 0.25
    graph_lambda: float = 0.0025
    lookback_days: int = 756
    rebalance_frequency: str = "QE"
    graph_threshold: float = 0.30
    minimum_training_observations: int = 200
    target_active_frequency: float = 0.10


@dataclass
class LinearGCVarWalkForwardResult:
    returns: pd.Series
    weights: pd.DataFrame
    audit: pd.DataFrame
    instability: pd.DataFrame


CORE_RANKING_STRATEGIES = {
    "equal_weight",
    "buy_hold_equal_weight",
    "inverse_volatility",
    "minimum_variance",
    "mean_variance",
    "risk_parity",
    "hierarchical_risk_parity",
    "cvar_optimized",
    "standard_cvar",
    "graph_cvar_optimized",
    "static_graph_cvar",
    "adaptive_graph_cvar",
    "fixed_quarterly_graph_cvar",
}

SUPPLEMENTAL_RANKING_STRATEGIES = {"adaptive_graph_cvar_v2"}
HITL_RANKING_STRATEGIES = {"sample_hitl_governed_adaptive_gcvar"}

REQUIRED_GOVERNANCE_METRICS = (
    "annual_return",
    "annual_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "historical_cvar_loss_95",
    "max_drawdown_magnitude",
    "turnover",
    "hhi",
    "effective_n",
    "graph_exposure",
)

GOVERNANCE_RANKING_WEIGHTS = {
    "annual_return": 0.10,
    "annual_volatility": 0.10,
    "sharpe_ratio": 0.15,
    "sortino_ratio": 0.10,
    "historical_cvar_loss_95": 0.20,
    "max_drawdown_magnitude": 0.15,
    "turnover": 0.05,
    "hhi": 0.05,
    "effective_n": 0.05,
    "graph_exposure": 0.05,
}

HIGHER_IS_BETTER = {
    "annual_return": True,
    "annual_volatility": False,
    "sharpe_ratio": True,
    "sortino_ratio": True,
    "historical_cvar_loss_95": False,
    "max_drawdown_magnitude": False,
    "turnover": False,
    "hhi": False,
    "effective_n": True,
    "graph_exposure": False,
}


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


def optimize_linear_centrality_cvar(
    returns: pd.DataFrame,
    centrality: pd.Series,
    effective_lambda: float,
    params: LinearGCVarParams,
) -> tuple[pd.Series, dict[str, object]]:
    """Solve CVaR with a linear centrality penalty c_t^T w."""
    clean = pd.DataFrame(returns).dropna(how="any")
    columns = list(clean.columns)
    observations, assets = clean.shape
    if observations == 0 or assets == 0:
        raise ValueError("V2 optimizer requires nonempty aligned returns")

    matrix = clean.to_numpy()
    mean = clean.mean().reindex(columns).to_numpy()
    penalty = pd.Series(centrality, dtype=float).reindex(columns).fillna(0.0).to_numpy()
    weight = cp.Variable(assets)
    threshold = cp.Variable()
    excess = cp.Variable(observations)

    losses = -matrix @ weight
    cvar = threshold + cp.sum(excess) / ((1.0 - params.alpha) * observations)
    expected_return = mean @ weight
    graph_penalty = penalty @ weight
    objective = cp.Minimize(
        cvar
        - float(params.return_tradeoff) * expected_return
        + float(effective_lambda) * graph_penalty
    )
    max_weight = max(float(params.max_weight), 1.0 / assets + 1e-6)
    constraints = [
        excess >= losses - threshold,
        excess >= 0,
        cp.sum(weight) == 1,
        weight >= 0,
        weight <= max_weight,
    ]
    problem = cp.Problem(objective, constraints)

    solver_used = None
    for solver in ("CLARABEL", "ECOS", "SCS"):
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver, verbose=False)
            if weight.value is not None:
                solver_used = solver
                break
        except Exception:
            continue

    if weight.value is None:
        fallback = np.ones(assets) / assets
        weights = pd.Series(fallback, index=columns)
        return weights, {
            "solver": "equal_weight_fallback",
            "status": "fallback",
            "fallback": True,
            "graph_objective_type": "linear_centrality",
            "weight_sum": 1.0,
            "maximum_weight": float(weights.max()),
        }

    raw = np.maximum(np.asarray(weight.value).ravel(), 0.0)
    if float(raw.sum()) <= 0:
        raw = np.ones(assets) / assets
    weights = pd.Series(raw / raw.sum(), index=columns)
    return weights, {
        "solver": solver_used,
        "status": problem.status,
        "fallback": False,
        "graph_objective_type": "linear_centrality",
        "weight_sum": float(weights.sum()),
        "maximum_weight": float(weights.max()),
        "objective_value": float(problem.value) if problem.value is not None else np.nan,
    }


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> list[pd.Timestamp]:
    if len(index) == 0:
        return []
    dates = (
        pd.Series(index, index=index)
        .resample(frequency)
        .first()
        .dropna()
        .tolist()
    )
    first = pd.Timestamp(index[0])
    if first not in dates:
        dates.insert(0, first)
    return sorted(pd.to_datetime(dates))


def run_linear_gcvar_walk_forward(
    returns: pd.DataFrame,
    universe: str,
    holdings: pd.DataFrame | None,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    params: LinearGCVarParams | None = None,
) -> LinearGCVarWalkForwardResult:
    """Run supplemental V2 over the untouched test lane."""
    params = params or LinearGCVarParams()
    aligned = pd.DataFrame(returns).dropna(how="any").sort_index()
    instability = compute_instability_series(aligned)
    if "instability_index" in instability.columns:
        theta, steepness = calibrate_gate(
            instability["instability_index"],
            validation_start,
            validation_end,
            params.target_active_frequency,
        )
    else:
        theta, steepness = 0.0, 1.0

    evaluation = aligned.loc[pd.Timestamp(test_start):pd.Timestamp(test_end)]
    decision_dates = _rebalance_dates(evaluation.index, params.rebalance_frequency)
    realized: list[pd.Series] = []
    weight_rows: list[pd.Series] = []
    audit_rows: list[dict[str, object]] = []
    previous_weights: pd.Series | None = None

    instability_series = (
        instability["instability_index"].dropna()
        if "instability_index" in instability.columns
        else pd.Series(dtype=float)
    )
    for position, decision_date in enumerate(decision_dates):
        next_date = (
            decision_dates[position + 1]
            if position + 1 < len(decision_dates)
            else pd.Timestamp(test_end) + pd.Timedelta(days=1)
        )
        history = aligned.loc[aligned.index < decision_date].tail(params.lookback_days)
        period = aligned.loc[
            (aligned.index >= decision_date) & (aligned.index < next_date)
        ]
        if len(history) < params.minimum_training_observations or period.empty:
            continue

        available_signal = instability_series.loc[instability_series.index < decision_date]
        instability_value = float(available_signal.iloc[-1]) if len(available_signal) else np.nan
        gate = adaptive_gate_signal(
            instability_value, theta, steepness, params.graph_lambda
        )
        centrality, graph_source = get_linear_graph_penalty(
            history,
            history.columns,
            decision_date,
            holdings,
            threshold=params.graph_threshold,
        )
        weights, solver_audit = optimize_linear_centrality_cvar(
            history, centrality, gate.effective_lambda, params
        )
        realized.append((period @ weights).rename("return"))
        turnover = (
            float((weights - previous_weights.reindex(weights.index).fillna(0.0)).abs().sum())
            if previous_weights is not None
            else 0.0
        )
        hhi = float(np.square(weights).sum())
        graph_exposure = float(
            (weights.reindex(centrality.index).fillna(0.0) * centrality).sum()
        )
        weight_rows.append(weights.rename(decision_date))
        audit_rows.append(
            {
                "universe": universe,
                "decision_date": decision_date,
                "period_end": next_date,
                "training_start": history.index.min(),
                "training_end": history.index.max(),
                "instability_index": gate.instability,
                "theta": gate.threshold,
                "steepness_k": gate.steepness,
                "lambda_multiplier": gate.multiplier,
                "lambda_effective": gate.effective_lambda,
                "active": gate.active,
                "graph_source": graph_source,
                "graph_exposure": graph_exposure,
                "turnover": turnover,
                "hhi": hhi,
                "effective_n": float(1.0 / hhi) if hhi > 0 else np.nan,
                "asset_count": len(weights),
                **solver_audit,
            }
        )
        previous_weights = weights

    return LinearGCVarWalkForwardResult(
        returns=pd.concat(realized).sort_index() if realized else pd.Series(dtype=float),
        weights=pd.DataFrame(weight_rows),
        audit=pd.DataFrame(audit_rows),
        instability=instability,
    )


def _classify_ranking_family(strategy: object) -> str:
    name = str(strategy)
    if name in HITL_RANKING_STRATEGIES or "hitl" in name.lower():
        return "hitl_simulation"
    if name in SUPPLEMENTAL_RANKING_STRATEGIES or name.endswith("_v2"):
        return "supplemental"
    if name in CORE_RANKING_STRATEGIES:
        return "core"
    return "supplemental"


def _normalize_metric_aliases(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(metrics).copy()
    if "annual_volatility" not in frame.columns and "volatility" in frame.columns:
        frame["annual_volatility"] = frame["volatility"]
    if "turnover" not in frame.columns and "mean_turnover" in frame.columns:
        frame["turnover"] = frame["mean_turnover"]
    elif "turnover" in frame.columns and "mean_turnover" in frame.columns:
        frame["turnover"] = frame["turnover"].fillna(frame["mean_turnover"])
    if "max_drawdown_magnitude" not in frame.columns and "max_drawdown" in frame.columns:
        frame["max_drawdown_magnitude"] = pd.to_numeric(
            frame["max_drawdown"], errors="coerce"
        ).abs()
    elif "max_drawdown_magnitude" in frame.columns and "max_drawdown" in frame.columns:
        frame["max_drawdown_magnitude"] = pd.to_numeric(
            frame["max_drawdown_magnitude"], errors="coerce"
        ).fillna(
            pd.to_numeric(frame["max_drawdown"], errors="coerce").abs()
        )
    return frame


def _rank_score(values: pd.Series, higher_is_better: bool) -> pd.Series:
    series = pd.Series(values, dtype=float)
    count = int(series.notna().sum())
    if count <= 1:
        return pd.Series(1.0, index=series.index)
    rank = series.rank(pct=True, method="average")
    if higher_is_better:
        return rank
    return 1.0 + (1.0 / count) - rank


def compute_family_rankings(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank complete rows only, separately by ranking family and universe."""
    frame = _normalize_metric_aliases(metrics)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame["ranking_family"] = frame["strategy"].map(_classify_ranking_family)

    missing_columns = [
        column for column in REQUIRED_GOVERNANCE_METRICS if column not in frame.columns
    ]
    for column in missing_columns:
        frame[column] = np.nan

    numeric = frame.loc[:, REQUIRED_GOVERNANCE_METRICS].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = np.isfinite(numeric).all(axis=1)
    eligible = frame.loc[finite].copy()
    rejected = frame.loc[~finite].copy()

    if not rejected.empty:
        reasons: list[str] = []
        for idx, row in numeric.loc[~finite].iterrows():
            missing = [
                column
                for column in REQUIRED_GOVERNANCE_METRICS
                if not np.isfinite(row[column])
            ]
            reasons.append(
                "missing_required_governance_metrics:" + ",".join(missing)
            )
        rejected["rejection_reason"] = reasons
        rejected["governance_composite_score"] = np.nan
        rejected["governance_rank"] = np.nan
    else:
        rejected = pd.DataFrame(columns=list(frame.columns) + ["rejection_reason"])

    ranked_groups: list[pd.DataFrame] = []
    for (_, _), group in eligible.groupby(["universe", "ranking_family"], dropna=False):
        scored = group.copy()
        for metric, weight in GOVERNANCE_RANKING_WEIGHTS.items():
            scored[f"score_{metric}"] = _rank_score(
                pd.to_numeric(scored[metric], errors="coerce"),
                HIGHER_IS_BETTER[metric],
            )
        scored["governance_composite_score"] = sum(
            GOVERNANCE_RANKING_WEIGHTS[metric] * scored[f"score_{metric}"]
            for metric in GOVERNANCE_RANKING_WEIGHTS
        )
        scored["governance_rank"] = scored["governance_composite_score"].rank(
            ascending=False, method="dense"
        )
        ranked_groups.append(scored)

    rankings = pd.concat(ranked_groups, ignore_index=True) if ranked_groups else pd.DataFrame()
    return rankings, rejected.reset_index(drop=True)
