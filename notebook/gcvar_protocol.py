"""Leakage-safe helpers for the Walk-Forward Governance G-CVaR protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProtocolDates:
    download_start: pd.Timestamp
    download_end_exclusive: pd.Timestamp
    training_start: pd.Timestamp
    training_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @classmethod
    def default(cls) -> "ProtocolDates":
        return cls(
            download_start=pd.Timestamp("2014-01-01"),
            download_end_exclusive=pd.Timestamp("2026-01-01"),
            training_start=pd.Timestamp("2014-01-01"),
            training_end=pd.Timestamp("2019-12-31"),
            validation_start=pd.Timestamp("2020-01-01"),
            validation_end=pd.Timestamp("2022-12-31"),
            test_start=pd.Timestamp("2023-01-01"),
            test_end=pd.Timestamp("2025-12-31"),
        )

    def validate(self) -> None:
        ordered = [
            self.download_start,
            self.training_start,
            self.training_end,
            self.validation_start,
            self.validation_end,
            self.test_start,
            self.test_end,
        ]
        if ordered != sorted(ordered):
            raise ValueError("Protocol dates must be monotonically ordered")
        if self.download_end_exclusive <= self.test_end:
            raise ValueError("Exclusive download end must be after the test end")


def stable_parameter_hash(candidate_grid: Iterable[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(candidate_grid), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_boundary_audit(
    universe: str,
    raw_index: pd.Index,
    selected_params: Mapping[str, Any],
    parameter_grid_hash: str,
    dates: ProtocolDates,
) -> dict[str, Any]:
    dates.validate()
    index = pd.DatetimeIndex(raw_index)
    if index.empty:
        raise ValueError("Boundary audit requires at least one raw observation")
    return {
        "universe": universe,
        "raw_data_start": str(index.min().date()),
        "raw_data_end": str(index.max().date()),
        "training_start": str(dates.training_start.date()),
        "training_end": str(dates.training_end.date()),
        "calibration_start": str(dates.validation_start.date()),
        "calibration_end": str(dates.validation_end.date()),
        "test_start": str(dates.test_start.date()),
        "test_end": str(dates.test_end.date()),
        "selected_params": json.dumps(dict(selected_params), sort_keys=True),
        "parameter_grid_hash": parameter_grid_hash,
        "whether_test_used_in_calibration": False,
    }


def make_tail_graph_psd_matrix(
    returns: pd.DataFrame,
    tail_quantile: float = 0.20,
    threshold: float = 0.25,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    clean = pd.DataFrame(returns).dropna(how="any")
    if clean.empty:
        raise ValueError("Tail graph requires nonempty aligned returns")
    market = clean.mean(axis=1)
    tail = clean.loc[market <= market.quantile(tail_quantile)]
    if len(tail) < 30:
        tail = clean
    corr = tail.corr().abs().fillna(0.0)
    values = corr.to_numpy(copy=True)
    np.fill_diagonal(values, 0.0)
    values[values < threshold] = 0.0
    values = (values + values.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    clipped = np.maximum(eigenvalues, 1e-8)
    psd = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    psd = (psd + psd.T) / 2.0
    graph = pd.DataFrame(psd, index=clean.columns, columns=clean.columns)
    diagnostics = {
        "observations": len(clean),
        "tail_observations": len(tail),
        "tail_quantile": tail_quantile,
        "threshold": threshold,
        "minimum_eigenvalue_before_projection": float(eigenvalues.min()),
        "minimum_eigenvalue_after_projection": float(np.linalg.eigvalsh(psd).min()),
    }
    return graph, diagnostics


def graph_exposure(weights: pd.Series, graph: pd.DataFrame) -> float:
    aligned = pd.Series(weights).reindex(graph.index).fillna(0.0).to_numpy()
    return float(aligned @ graph.to_numpy() @ aligned)


@dataclass(frozen=True)
class GovernanceParams:
    alpha: float = 0.95
    max_weight: float = 0.30
    gamma_return: float = 0.80
    graph_lambda: float = 0.10
    lambda_max: float = 0.80
    turnover_lambda: float = 0.02
    rho_cvar: float = 0.98
    rho_graph: float = 0.90
    rho_return: float = 0.90
    slack_penalty: float = 10.0
    instability_quantile: float = 0.80
    sigmoid_steepness: float = 8.0
    tail_quantile: float = 0.20
    graph_threshold: float = 0.25


def _historical_cvar_loss(values: pd.Series, alpha: float) -> float:
    series = pd.Series(values).dropna()
    if series.empty:
        return np.nan
    cutoff = series.quantile(1.0 - alpha)
    return float(-series[series <= cutoff].mean())


def optimize_governance_gcvar(
    returns: pd.DataFrame,
    graph_matrix: pd.DataFrame,
    lambda_t: float,
    previous_weights: pd.Series | None,
    params: GovernanceParams,
) -> tuple[pd.Series, dict[str, Any]]:
    clean = pd.DataFrame(returns).dropna(how="any")
    columns = list(clean.columns)
    observations, assets = clean.shape
    if observations == 0 or assets == 0:
        raise ValueError("Optimizer requires nonempty aligned returns")

    matrix = clean.to_numpy()
    mean = clean.mean().to_numpy()
    graph = (
        pd.DataFrame(graph_matrix)
        .reindex(index=columns, columns=columns)
        .fillna(0.0)
        .to_numpy()
    )
    graph = (graph + graph.T) / 2.0
    equal = np.ones(assets) / assets
    equal_returns = clean @ equal
    equal_cvar = max(_historical_cvar_loss(equal_returns, params.alpha), 1e-8)
    equal_mean = max(abs(float(mean @ equal)), 1e-8)
    equal_graph = max(float(equal @ graph @ equal), 1e-8)

    weight = cp.Variable(assets)
    threshold = cp.Variable()
    excess = cp.Variable(observations)
    cvar_slack = cp.Variable(nonneg=True)
    graph_slack = cp.Variable(nonneg=True)
    return_slack = cp.Variable(nonneg=True)

    losses = -matrix @ weight
    cvar = threshold + cp.sum(excess) / ((1.0 - params.alpha) * observations)
    cvar_normalized = cvar / equal_cvar
    return_normalized = (mean @ weight) / equal_mean
    graph_expression = cp.quad_form(weight, cp.psd_wrap(graph))
    graph_normalized = graph_expression / equal_graph
    graph_constraint_active = float(lambda_t) > 0

    objective = cvar_normalized - params.gamma_return * return_normalized
    if graph_constraint_active:
        objective += float(lambda_t) * graph_normalized
    if previous_weights is not None:
        previous = (
            pd.Series(previous_weights)
            .reindex(columns)
            .fillna(0.0)
            .to_numpy()
        )
        previous = previous / previous.sum() if previous.sum() > 0 else equal
        objective += params.turnover_lambda * cp.norm1(weight - previous)
    objective += params.slack_penalty * (cvar_slack + graph_slack + return_slack)

    max_weight = max(params.max_weight, 1.0 / assets + 1e-6)
    constraints = [
        excess >= losses - threshold,
        excess >= 0,
        cp.sum(weight) == 1,
        weight >= 0,
        weight <= max_weight,
        cvar_normalized <= params.rho_cvar + cvar_slack,
        mean @ weight >= params.rho_return * float(mean @ equal) - return_slack,
    ]
    if graph_constraint_active:
        constraints.append(
            graph_normalized <= params.rho_graph + graph_slack
        )
    problem = cp.Problem(cp.Minimize(objective), constraints)

    used_solver = None
    for solver in ("CLARABEL", "ECOS", "SCS"):
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver, verbose=False)
            if weight.value is not None:
                used_solver = solver
                break
        except Exception:
            continue

    if weight.value is None:
        result = pd.Series(equal, index=columns)
        return result, {
            "solver": "equal_weight_fallback",
            "status": "fallback",
            "cvar_slack": np.nan,
            "graph_slack": np.nan,
            "return_slack": np.nan,
            "fallback": True,
            "graph_constraint_active": graph_constraint_active,
            "weight_sum": 1.0,
            "maximum_weight": float(result.max()),
        }

    raw = np.maximum(np.asarray(weight.value).ravel(), 0.0)
    result = pd.Series(raw / raw.sum(), index=columns)
    return result, {
        "solver": used_solver,
        "status": problem.status,
        "cvar_slack": max(float(cvar_slack.value), 0.0),
        "graph_slack": max(float(graph_slack.value), 0.0),
        "return_slack": max(float(return_slack.value), 0.0),
        "fallback": False,
        "graph_constraint_active": graph_constraint_active,
        "weight_sum": float(result.sum()),
        "maximum_weight": float(result.max()),
    }


@dataclass
class WalkForwardResult:
    returns: pd.Series
    weight_history: pd.DataFrame
    decision_log: pd.DataFrame
    solver_audit: pd.DataFrame


def adaptive_lambda_quantile(
    history: pd.Series,
    current_value: float,
    lambda_max: float,
    quantile: float,
    steepness: float,
) -> float:
    past = pd.Series(history).dropna()
    if len(past) < 30 or not np.isfinite(current_value):
        return 0.0
    boundary = float(past.quantile(quantile))
    if current_value < boundary:
        return 0.0
    return float(
        lambda_max
        / (1.0 + np.exp(-steepness * (float(current_value) - boundary)))
    )


def run_walk_forward_gcvar(
    returns: pd.DataFrame,
    instability: pd.Series,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    params: GovernanceParams,
    adaptive: bool,
    rebalance_frequency: str = "QE",
    lookback_days: int = 756,
    minimum_training_observations: int = 240,
) -> WalkForwardResult:
    aligned = pd.DataFrame(returns).dropna(how="any").sort_index()
    if aligned.empty:
        return WalkForwardResult(
            returns=pd.Series(dtype=float),
            weight_history=pd.DataFrame(),
            decision_log=pd.DataFrame(),
            solver_audit=pd.DataFrame(),
        )
    evaluation = aligned.loc[evaluation_start:evaluation_end]
    if evaluation.empty:
        return WalkForwardResult(
            returns=pd.Series(dtype=float),
            weight_history=pd.DataFrame(),
            decision_log=pd.DataFrame(),
            solver_audit=pd.DataFrame(),
        )

    decision_dates = (
        pd.Series(evaluation.index, index=evaluation.index)
        .resample(rebalance_frequency)
        .first()
        .dropna()
        .tolist()
    )
    realized: list[pd.Series] = []
    weights: list[pd.Series] = []
    logs: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    previous: pd.Series | None = None
    instability = pd.Series(instability).sort_index()

    for position, decision_date in enumerate(decision_dates):
        next_date = (
            decision_dates[position + 1]
            if position + 1 < len(decision_dates)
            else evaluation_end + pd.Timedelta(days=1)
        )
        history = aligned.loc[aligned.index < decision_date].tail(lookback_days)
        if len(history) < minimum_training_observations:
            continue
        graph, graph_diagnostics = make_tail_graph_psd_matrix(
            history,
            params.tail_quantile,
            params.graph_threshold,
        )

        instability_available = instability.loc[instability.index < decision_date].dropna()
        current = (
            float(instability_available.iloc[-1])
            if len(instability_available)
            else np.nan
        )
        lambda_t = (
            adaptive_lambda_quantile(
                instability_available.iloc[:-1],
                current,
                params.lambda_max,
                params.instability_quantile,
                params.sigmoid_steepness,
            )
            if adaptive
            else params.graph_lambda
        )
        allocation, audit = optimize_governance_gcvar(
            history,
            graph,
            lambda_t,
            previous,
            params,
        )
        period = evaluation.loc[
            (evaluation.index >= decision_date) & (evaluation.index < next_date)
        ]
        if not period.empty:
            realized.append((period @ allocation).rename("return"))

        turnover = (
            float(
                (
                    allocation
                    - previous.reindex(allocation.index).fillna(0.0)
                )
                .abs()
                .sum()
            )
            if previous is not None
            else 0.0
        )
        regime = (
            "crisis"
            if lambda_t > 0.5 * params.lambda_max
            else ("elevated" if lambda_t > 0 else "calm")
        )
        weights.append(allocation.rename(decision_date))
        logs.append(
            {
                "decision_date": decision_date,
                "training_start": history.index.min(),
                "training_end": history.index.max(),
                "lambda_t": lambda_t,
                "instability_index": current,
                "regime": regime,
                "graph_exposure": graph_exposure(allocation, graph),
                "turnover": turnover,
                "effective_n": float(1.0 / np.square(allocation).sum()),
                **graph_diagnostics,
            }
        )
        audits.append({"decision_date": decision_date, **audit})
        previous = allocation

    return WalkForwardResult(
        returns=pd.concat(realized).sort_index()
        if realized
        else pd.Series(dtype=float),
        weight_history=pd.DataFrame(weights),
        decision_log=pd.DataFrame(logs),
        solver_audit=pd.DataFrame(audits),
    )


GOVERNANCE_SCORE_WEIGHTS = {
    "sharpe_ratio": 0.20,
    "annual_return": 0.15,
    "historical_cvar_loss_95": 0.25,
    "max_drawdown_magnitude": 0.15,
    "graph_exposure": 0.10,
    "effective_n": 0.10,
    "turnover": 0.05,
}

LOWER_IS_BETTER = {
    "historical_cvar_loss_95",
    "max_drawdown_magnitude",
    "graph_exposure",
    "turnover",
}


def assert_disjoint_windows(
    validation_index: pd.Index,
    test_index: pd.Index,
) -> None:
    overlap = pd.DatetimeIndex(validation_index).intersection(
        pd.DatetimeIndex(test_index)
    )
    if len(overlap):
        raise ValueError(
            f"Validation and test windows overlap on {overlap.min().date()}"
        )


def compute_governance_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(metrics).copy()
    required = list(GOVERNANCE_SCORE_WEIGHTS)
    for metric in required:
        if metric not in frame:
            frame[metric] = np.nan
    frame["score_status"] = np.where(
        frame[required].notna().all(axis=1),
        "complete",
        "incomplete",
    )
    for metric in required:
        score_column = f"score_{metric}"
        frame[score_column] = np.nan
        for _, positions in frame.groupby("universe").groups.items():
            values = frame.loc[positions, metric].astype(float)
            finite = values.dropna()
            if finite.empty:
                continue
            minimum, maximum = finite.min(), finite.max()
            normalized = (
                pd.Series(0.5, index=values.index)
                if maximum == minimum
                else (values - minimum) / (maximum - minimum)
            )
            if metric in LOWER_IS_BETTER:
                normalized = 1.0 - normalized
            frame.loc[positions, score_column] = normalized

    complete = frame["score_status"].eq("complete")
    frame["composite_governance_score"] = np.nan
    weighted = sum(
        frame.loc[complete, f"score_{metric}"] * weight
        for metric, weight in GOVERNANCE_SCORE_WEIGHTS.items()
    )
    frame.loc[complete, "composite_governance_score"] = weighted
    frame["governance_rank"] = frame.groupby("universe")[
        "composite_governance_score"
    ].rank(ascending=False, method="min")
    frame["is_governance_winner"] = frame["governance_rank"].eq(1.0)
    return frame


def _annual_return(values: pd.Series, periods: int = 252) -> float:
    series = pd.Series(values).dropna()
    if series.empty:
        return np.nan
    growth = float((1.0 + series).prod())
    if growth <= 0:
        return np.nan
    return growth ** (periods / len(series)) - 1.0


def _maximum_drawdown_magnitude(values: pd.Series) -> float:
    wealth = (1.0 + pd.Series(values).dropna()).cumprod()
    if wealth.empty:
        return np.nan
    drawdown = wealth / wealth.cummax() - 1.0
    return float(-drawdown.min())


def summarize_walk_forward(
    universe: str,
    strategy: str,
    result: WalkForwardResult,
) -> dict[str, Any]:
    returns = result.returns.dropna()
    annual_return = _annual_return(returns)
    annual_volatility = (
        float(returns.std() * np.sqrt(252)) if len(returns) else np.nan
    )
    sharpe = (
        annual_return / annual_volatility
        if annual_volatility > 0 and np.isfinite(annual_return)
        else np.nan
    )
    final_weights = (
        result.weight_history.iloc[-1]
        if not result.weight_history.empty
        else pd.Series(dtype=float)
    )
    squared_sum = float(np.square(final_weights).sum())
    effective_n = 1.0 / squared_sum if squared_sum > 0 else np.nan
    return {
        "universe": universe,
        "strategy": strategy,
        "sharpe_ratio": sharpe,
        "annual_return": annual_return,
        "historical_cvar_loss_95": _historical_cvar_loss(returns, 0.95),
        "max_drawdown_magnitude": _maximum_drawdown_magnitude(returns),
        "graph_exposure": float(result.decision_log["graph_exposure"].mean())
        if not result.decision_log.empty
        else np.nan,
        "effective_n": effective_n,
        "turnover": float(result.decision_log["turnover"].mean())
        if not result.decision_log.empty
        else np.nan,
    }


def calibrate_governance_gcvar(
    universe: str,
    train_returns: pd.DataFrame,
    validation_returns: pd.DataFrame,
    instability: pd.Series,
    candidate_grid: Iterable[Mapping[str, Any]],
    rebalance_frequency: str = "QE",
    lookback_days: int = 756,
) -> tuple[GovernanceParams, pd.DataFrame]:
    if validation_returns.empty:
        raise ValueError("Validation returns must not be empty")
    combined = pd.concat([train_returns, validation_returns]).sort_index()
    rows: list[dict[str, Any]] = []
    for candidate in candidate_grid:
        params = GovernanceParams(**dict(candidate))
        result = run_walk_forward_gcvar(
            returns=combined,
            instability=instability,
            evaluation_start=pd.Timestamp(validation_returns.index.min()),
            evaluation_end=pd.Timestamp(validation_returns.index.max()),
            params=params,
            adaptive=True,
            rebalance_frequency=rebalance_frequency,
            lookback_days=lookback_days,
        )
        row = summarize_walk_forward(universe, "adaptive_graph_cvar", result)
        row["parameter_json"] = json.dumps(asdict(params), sort_keys=True)
        rows.append(row)

    scored = compute_governance_scores(pd.DataFrame(rows))
    scored = scored.sort_values(
        [
            "composite_governance_score",
            "historical_cvar_loss_95",
            "graph_exposure",
            "turnover",
            "parameter_json",
        ],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if scored.empty or scored.loc[0, "score_status"] != "complete":
        raise ValueError("No complete validation candidate was available")
    scored["selected"] = False
    scored.loc[0, "selected"] = True
    selected = GovernanceParams(
        **json.loads(scored.loc[0, "parameter_json"])
    )
    return selected, scored


def build_behavioral_validation_table(
    strategy_results: Mapping[tuple[str, str], WalkForwardResult],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (universe, strategy), result in strategy_results.items():
        if result.returns.empty or result.decision_log.empty:
            continue
        decisions = result.decision_log.sort_values("decision_date").copy()
        decisions["next_date"] = decisions["decision_date"].shift(-1)
        decisions.loc[decisions.index[-1], "next_date"] = (
            result.returns.index.max() + pd.Timedelta(days=1)
        )
        for _, decision in decisions.iterrows():
            period = result.returns.loc[
                (result.returns.index >= decision["decision_date"])
                & (result.returns.index < decision["next_date"])
            ]
            for date, value in period.items():
                rows.append(
                    {
                        "universe": universe,
                        "strategy": strategy,
                        "date": date,
                        "return": value,
                        "regime": decision["regime"],
                        "lambda_t": decision["lambda_t"],
                        "graph_exposure": decision["graph_exposure"],
                        "turnover": decision["turnover"],
                        "effective_n": decision["effective_n"],
                    }
                )

    daily = pd.DataFrame(rows)
    if daily.empty:
        return pd.DataFrame()
    output: list[dict[str, Any]] = []
    for (universe, regime, strategy), group in daily.groupby(
        ["universe", "regime", "strategy"]
    ):
        values = group["return"].dropna()
        annual_return = _annual_return(values)
        annual_volatility = (
            float(values.std() * np.sqrt(252)) if len(values) else np.nan
        )
        output.append(
            {
                "universe": universe,
                "regime": regime,
                "strategy": strategy,
                "observations": len(values),
                "annual_return": annual_return,
                "sharpe_ratio": annual_return / annual_volatility
                if annual_volatility > 0 and np.isfinite(annual_return)
                else np.nan,
                "historical_cvar_loss_95": _historical_cvar_loss(
                    values, 0.95
                ),
                "max_drawdown_magnitude": _maximum_drawdown_magnitude(values),
                "graph_exposure": float(group["graph_exposure"].mean()),
                "turnover": float(group["turnover"].mean()),
                "effective_n": float(group["effective_n"].mean()),
                "adaptive_activation_frequency": float(
                    (group["lambda_t"] > 0).mean()
                ),
            }
        )
    return pd.DataFrame(output)


def validate_adaptive_graph_behavior(logs: pd.DataFrame) -> pd.DataFrame:
    required = {"universe", "strategy", "regime", "graph_exposure"}
    missing = required.difference(logs.columns)
    if missing:
        raise ValueError(f"Behavior logs missing columns: {sorted(missing)}")
    adaptive = logs.loc[logs["strategy"].eq("adaptive_graph_cvar")]
    rows: list[dict[str, Any]] = []
    for universe, group in adaptive.groupby("universe"):
        calm = group.loc[
            group["regime"].eq("calm"), "graph_exposure"
        ].mean()
        crisis = group.loc[
            group["regime"].eq("crisis"), "graph_exposure"
        ].mean()
        rows.append(
            {
                "universe": universe,
                "calm_graph_exposure": calm,
                "crisis_graph_exposure": crisis,
                "crisis_graph_exposure_lower_than_calm": bool(
                    pd.notna(calm) and pd.notna(crisis) and crisis < calm
                ),
            }
        )
    return pd.DataFrame(rows)
