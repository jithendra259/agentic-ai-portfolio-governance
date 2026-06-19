"""Leakage-safe helpers for the Walk-Forward Governance G-CVaR protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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

    objective = cvar_normalized - params.gamma_return * return_normalized
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
        graph_normalized <= params.rho_graph + graph_slack,
        mean @ weight >= params.rho_return * float(mean @ equal) - return_slack,
    ]
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
        "weight_sum": float(result.sum()),
        "maximum_weight": float(result.max()),
    }
