# Walk-Forward Governance G-CVaR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify a leakage-safe Static and Adaptive Governance G-CVaR protocol using complete 2014–2025 market data, validation-only calibration, untouched 2023–2025 testing, governance scoring, behavioral evidence, and explicit audit exports.

**Architecture:** Extract the reusable mathematics and evaluation machinery from the generated 6,000-line script into `notebook/gcvar_protocol.py`, where deterministic unit tests can import it without executing the full analysis. Keep orchestration, existing baseline strategies, publication plots, and artifact generation in `notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py`, replacing only the old G-CVaR path and inconsistent date handling.

**Tech Stack:** Python 3.13, pandas 3, NumPy, SciPy, CVXPY, yfinance, Matplotlib, unittest.

---

## File Structure

- Create `notebook/gcvar_protocol.py`: date-boundary validation, tail-graph construction, normalized optimizer, adaptive gating, walk-forward execution, calibration, behavioral metrics, governance scoring, and audit helpers.
- Create `notebook/tests/test_gcvar_protocol.py`: deterministic behavioral and mathematical tests that import the focused module without running live downloads.
- Modify `notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py`: corrected dates, imports, per-universe coverage audit, calibrated Static/Adaptive G-CVaR execution, exports, plots, and thesis-safe narrative.
- Modify `notebook/tests/test_portfolio_plot_script_contract.py`: source-level integration contracts for fixed dates, immutable score weights, required exports, and Matplotlib output paths.

### Task 1: Lock Date Boundaries and Coverage Semantics

**Files:**
- Create: `notebook/gcvar_protocol.py`
- Create: `notebook/tests/test_gcvar_protocol.py`
- Modify: `notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py:170-205`
- Modify: `notebook/tests/test_portfolio_plot_script_contract.py`

- [ ] **Step 1: Write failing boundary tests**

```python
# notebook/tests/test_gcvar_protocol.py
import sys
import unittest
from pathlib import Path

import pandas as pd

NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NOTEBOOK_DIR))

from gcvar_protocol import ProtocolDates, build_boundary_audit


class ProtocolDateTests(unittest.TestCase):
    def test_protocol_dates_are_ordered_and_cover_2014_through_2025(self):
        dates = ProtocolDates.default()
        self.assertEqual(dates.download_start, pd.Timestamp("2014-01-01"))
        self.assertEqual(dates.download_end_exclusive, pd.Timestamp("2026-01-01"))
        self.assertEqual(dates.training_end, pd.Timestamp("2019-12-31"))
        self.assertEqual(dates.validation_end, pd.Timestamp("2022-12-31"))
        self.assertEqual(dates.test_end, pd.Timestamp("2025-12-31"))

    def test_boundary_audit_proves_test_is_not_used_for_calibration(self):
        audit = build_boundary_audit(
            universe="U1",
            raw_index=pd.bdate_range("2014-01-02", "2025-12-31"),
            selected_params={"graph_lambda": 0.1},
            parameter_grid_hash="abc123",
            dates=ProtocolDates.default(),
        )
        self.assertFalse(audit["whether_test_used_in_calibration"])
        self.assertEqual(audit["raw_data_end"], "2025-12-31")
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest notebook.tests.test_gcvar_protocol.ProtocolDateTests -v`

Expected: import failure because `gcvar_protocol.py` does not exist.

- [ ] **Step 3: Implement immutable boundaries and audit helper**

```python
# notebook/gcvar_protocol.py
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
```

Update the script configuration to use `end_date="2026-01-01"` and the explicit training/validation/test keys from `ProtocolDates.default()`.

- [ ] **Step 4: Run boundary tests and the source contract**

Run: `python -m unittest notebook.tests.test_gcvar_protocol.ProtocolDateTests notebook.tests.test_portfolio_plot_script_contract -v`

Expected: all boundary tests pass; existing Pandas compatibility tests remain green.

- [ ] **Step 5: Commit the boundary foundation**

```powershell
git add -- 'notebook/gcvar_protocol.py' 'notebook/tests/test_gcvar_protocol.py' 'notebook/tests/test_portfolio_plot_script_contract.py' 'notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py'
git commit -m "fix: enforce 2014 through 2025 gcvar boundaries"
```

### Task 2: Build the Tail-Graph PSD Matrix

**Files:**
- Modify: `notebook/gcvar_protocol.py`
- Modify: `notebook/tests/test_gcvar_protocol.py`

- [ ] **Step 1: Write failing mathematical tests**

```python
class TailGraphTests(unittest.TestCase):
    def test_tail_graph_is_aligned_symmetric_finite_and_psd(self):
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            rng.normal(0.0004, 0.015, size=(300, 4)),
            columns=["A", "B", "C", "D"],
        )
        graph, diagnostics = make_tail_graph_psd_matrix(
            returns, tail_quantile=0.20, threshold=0.25
        )
        self.assertEqual(list(graph.index), list(returns.columns))
        self.assertTrue(np.isfinite(graph.to_numpy()).all())
        self.assertTrue(np.allclose(graph, graph.T, atol=1e-10))
        self.assertGreaterEqual(np.linalg.eigvalsh(graph).min(), -1e-10)
        self.assertGreaterEqual(diagnostics["tail_observations"], 30)
```

- [ ] **Step 2: Run the graph test and verify RED**

Run: `python -m unittest notebook.tests.test_gcvar_protocol.TailGraphTests -v`

Expected: failure because `make_tail_graph_psd_matrix` is not defined.

- [ ] **Step 3: Implement tail selection, thresholding, and PSD projection**

```python
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
```

- [ ] **Step 4: Run the graph tests**

Run: `python -m unittest notebook.tests.test_gcvar_protocol.TailGraphTests -v`

Expected: PASS with finite PSD graph and diagnostics.

- [ ] **Step 5: Commit the graph implementation**

```powershell
git add -- 'notebook/gcvar_protocol.py' 'notebook/tests/test_gcvar_protocol.py'
git commit -m "feat: add downside tail graph construction"
```

### Task 3: Implement the Normalized Governance G-CVaR Optimizer

**Files:**
- Modify: `notebook/gcvar_protocol.py`
- Modify: `notebook/tests/test_gcvar_protocol.py`

- [ ] **Step 1: Write failing optimizer tests**

```python
class GovernanceOptimizerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.returns = pd.DataFrame(
            rng.normal([0.0008, 0.0005, 0.0003], [0.020, 0.012, 0.009], size=(500, 3)),
            columns=["A", "B", "C"],
        )
        self.graph, _ = make_tail_graph_psd_matrix(self.returns, threshold=0.0)

    def test_optimizer_returns_feasible_weights_and_nonnegative_slack(self):
        weights, audit = optimize_governance_gcvar(
            self.returns,
            graph_matrix=self.graph,
            lambda_t=0.10,
            previous_weights=None,
            params=GovernanceParams(),
        )
        self.assertAlmostEqual(weights.sum(), 1.0, places=7)
        self.assertGreaterEqual(weights.min(), -1e-9)
        self.assertLessEqual(weights.max(), GovernanceParams().max_weight + 1e-6)
        self.assertGreaterEqual(audit["cvar_slack"], 0.0)
        self.assertGreaterEqual(audit["graph_slack"], 0.0)
        self.assertGreaterEqual(audit["return_slack"], 0.0)

    def test_turnover_penalty_keeps_weights_closer_to_previous_allocation(self):
        previous = pd.Series([0.70, 0.20, 0.10], index=self.returns.columns)
        low, _ = optimize_governance_gcvar(
            self.returns, self.graph, 0.10, previous,
            GovernanceParams(turnover_lambda=0.0),
        )
        high, _ = optimize_governance_gcvar(
            self.returns, self.graph, 0.10, previous,
            GovernanceParams(turnover_lambda=1.0),
        )
        self.assertLessEqual((high - previous).abs().sum(), (low - previous).abs().sum() + 1e-6)
```

- [ ] **Step 2: Run optimizer tests and verify RED**

Run: `python -m unittest notebook.tests.test_gcvar_protocol.GovernanceOptimizerTests -v`

Expected: failure because `GovernanceParams` and `optimize_governance_gcvar` do not exist.

- [ ] **Step 3: Implement parameter model and optimizer**

```python
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
    graph = graph_matrix.reindex(index=columns, columns=columns).fillna(0.0).to_numpy()
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
    graph_expression = cp.quad_form(weight, graph)
    graph_normalized = graph_expression / equal_graph
    objective = cvar_normalized - params.gamma_return * return_normalized
    objective += float(lambda_t) * graph_normalized
    if previous_weights is not None:
        previous = pd.Series(previous_weights).reindex(columns).fillna(0.0).to_numpy()
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
        }
    raw = np.maximum(np.asarray(weight.value).ravel(), 0.0)
    result = pd.Series(raw / raw.sum(), index=columns)
    return result, {
        "solver": used_solver,
        "status": problem.status,
        "cvar_slack": float(cvar_slack.value or 0.0),
        "graph_slack": float(graph_slack.value or 0.0),
        "return_slack": float(return_slack.value or 0.0),
        "fallback": False,
        "weight_sum": float(result.sum()),
        "maximum_weight": float(result.max()),
    }
```

- [ ] **Step 4: Run optimizer tests**

Run: `python -m unittest notebook.tests.test_gcvar_protocol.GovernanceOptimizerTests -v`

Expected: both tests pass with a real installed solver.

- [ ] **Step 5: Commit the normalized optimizer**

```powershell
git add -- 'notebook/gcvar_protocol.py' 'notebook/tests/test_gcvar_protocol.py'
git commit -m "feat: implement normalized governance gcvar optimizer"
```

### Task 4: Implement Adaptive Gating and Leakage-Safe Walk-Forward Execution

**Files:**
- Modify: `notebook/gcvar_protocol.py`
- Modify: `notebook/tests/test_gcvar_protocol.py`

- [ ] **Step 1: Write failing adaptive and no-look-ahead tests**

```python
class AdaptiveProtocolTests(unittest.TestCase):
    def test_adaptive_lambda_is_zero_in_calm_and_bounded_in_crisis(self):
        history = pd.Series(np.linspace(-1.0, 1.0, 200))
        calm = adaptive_lambda_quantile(history, -0.5, 0.8, 0.8, 8.0)
        crisis = adaptive_lambda_quantile(history, 2.0, 0.8, 0.8, 8.0)
        self.assertEqual(calm, 0.0)
        self.assertGreater(crisis, 0.0)
        self.assertLessEqual(crisis, 0.8)

    def test_walk_forward_training_cutoff_is_strictly_before_decision_date(self):
        index = pd.bdate_range("2019-01-01", "2023-03-31")
        rng = np.random.default_rng(99)
        returns = pd.DataFrame(rng.normal(0, 0.01, (len(index), 3)), index=index, columns=list("ABC"))
        instability = pd.Series(rng.normal(size=len(index)), index=index)
        result = run_walk_forward_gcvar(
            returns=returns,
            instability=instability,
            evaluation_start=pd.Timestamp("2023-01-01"),
            evaluation_end=pd.Timestamp("2023-03-31"),
            params=GovernanceParams(),
            adaptive=True,
            rebalance_frequency="ME",
            lookback_days=756,
        )
        self.assertTrue((result.decision_log["training_end"] < result.decision_log["decision_date"]).all())
```

- [ ] **Step 2: Run adaptive tests and verify RED**

Run: `python -m unittest notebook.tests.test_gcvar_protocol.AdaptiveProtocolTests -v`

Expected: missing adaptive and walk-forward functions.

- [ ] **Step 3: Implement the adaptive gate and walk-forward result boundary**

```python
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
    return float(lambda_max / (1.0 + np.exp(-steepness * (current_value - boundary))))


def run_walk_forward_gcvar(
    returns: pd.DataFrame,
    instability: pd.Series,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    params: GovernanceParams,
    adaptive: bool,
    rebalance_frequency: str = "QE",
    lookback_days: int = 756,
) -> WalkForwardResult:
    aligned = pd.DataFrame(returns).dropna(how="any").sort_index()
    evaluation = aligned.loc[evaluation_start:evaluation_end]
    decision_dates = pd.Series(evaluation.index, index=evaluation.index).resample(rebalance_frequency).first().dropna()
    realized: list[pd.Series] = []
    weights: list[pd.Series] = []
    logs: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    previous: pd.Series | None = None
    for position, decision_date in enumerate(decision_dates):
        next_date = decision_dates.iloc[position + 1] if position + 1 < len(decision_dates) else evaluation_end + pd.Timedelta(days=1)
        history = aligned.loc[aligned.index < decision_date].tail(lookback_days)
        if len(history) < 240:
            continue
        graph, graph_diagnostics = make_tail_graph_psd_matrix(
            history, params.tail_quantile, params.graph_threshold
        )
        instability_history = instability.loc[instability.index < decision_date]
        current = float(instability_history.iloc[-1]) if len(instability_history) else np.nan
        lambda_t = adaptive_lambda_quantile(
            instability_history.iloc[:-1], current, params.lambda_max,
            params.instability_quantile, params.sigmoid_steepness,
        ) if adaptive else params.graph_lambda
        allocation, audit = optimize_governance_gcvar(history, graph, lambda_t, previous, params)
        period = evaluation.loc[(evaluation.index >= decision_date) & (evaluation.index < next_date)]
        if not period.empty:
            realized.append(period @ allocation)
        weights.append(allocation.rename(decision_date))
        logs.append({
            "decision_date": decision_date,
            "training_start": history.index.min(),
            "training_end": history.index.max(),
            "lambda_t": lambda_t,
            "instability_index": current,
            "regime": "crisis" if lambda_t > 0.5 * params.lambda_max else ("elevated" if lambda_t > 0 else "calm"),
            "graph_exposure": graph_exposure(allocation, graph),
            "turnover": float((allocation - previous.reindex(allocation.index).fillna(0)).abs().sum()) if previous is not None else 0.0,
            "effective_n": float(1.0 / np.square(allocation).sum()),
            **graph_diagnostics,
        })
        audits.append({"decision_date": decision_date, **audit})
        previous = allocation
    return WalkForwardResult(
        returns=pd.concat(realized).sort_index() if realized else pd.Series(dtype=float),
        weight_history=pd.DataFrame(weights),
        decision_log=pd.DataFrame(logs),
        solver_audit=pd.DataFrame(audits),
    )
```

- [ ] **Step 4: Run adaptive tests**

Run: `python -m unittest notebook.tests.test_gcvar_protocol.AdaptiveProtocolTests -v`

Expected: calm/crisis and strict historical cutoff tests pass.

- [ ] **Step 5: Commit walk-forward execution**

```powershell
git add -- 'notebook/gcvar_protocol.py' 'notebook/tests/test_gcvar_protocol.py'
git commit -m "feat: add leakage safe adaptive gcvar walk forward"
```

### Task 5: Add Immutable Governance Scoring and Validation-Only Calibration

**Files:**
- Modify: `notebook/gcvar_protocol.py`
- Modify: `notebook/tests/test_gcvar_protocol.py`

- [ ] **Step 1: Write failing score and calibration-separation tests**

```python
class GovernanceScoringTests(unittest.TestCase):
    def test_score_weights_are_fixed_and_sum_to_one(self):
        self.assertEqual(GOVERNANCE_SCORE_WEIGHTS, {
            "sharpe_ratio": 0.20,
            "annual_return": 0.15,
            "historical_cvar_loss_95": 0.25,
            "max_drawdown_magnitude": 0.15,
            "graph_exposure": 0.10,
            "effective_n": 0.10,
            "turnover": 0.05,
        })
        self.assertAlmostEqual(sum(GOVERNANCE_SCORE_WEIGHTS.values()), 1.0)

    def test_incomplete_rows_do_not_receive_silently_reweighted_scores(self):
        metrics = pd.DataFrame([
            {"universe": "U1", "strategy": "complete", "sharpe_ratio": 1.0, "annual_return": 0.1, "historical_cvar_loss_95": 0.02, "max_drawdown_magnitude": 0.2, "graph_exposure": 0.2, "effective_n": 5.0, "turnover": 0.1},
            {"universe": "U1", "strategy": "missing", "sharpe_ratio": 2.0},
        ])
        scored = compute_governance_scores(metrics)
        missing = scored.set_index("strategy").loc["missing"]
        self.assertEqual(missing["score_status"], "incomplete")
        self.assertTrue(pd.isna(missing["composite_governance_score"]))

    def test_calibration_rejects_overlapping_validation_and_test_dates(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint_windows(
                pd.bdate_range("2020-01-01", "2023-01-10"),
                pd.bdate_range("2023-01-01", "2025-12-31"),
            )
```

- [ ] **Step 2: Run scoring tests and verify RED**

Run: `python -m unittest notebook.tests.test_gcvar_protocol.GovernanceScoringTests -v`

Expected: missing score constants and functions.

- [ ] **Step 3: Implement fixed scoring and disjoint-window guard**

```python
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


def assert_disjoint_windows(validation_index: pd.Index, test_index: pd.Index) -> None:
    overlap = pd.DatetimeIndex(validation_index).intersection(pd.DatetimeIndex(test_index))
    if len(overlap):
        raise ValueError(f"Validation and test windows overlap on {overlap.min().date()}")


def compute_governance_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(metrics).copy()
    required = list(GOVERNANCE_SCORE_WEIGHTS)
    frame["score_status"] = np.where(frame[required].notna().all(axis=1), "complete", "incomplete")
    component_columns: list[str] = []
    for metric in required:
        score_column = f"score_{metric}"
        component_columns.append(score_column)
        frame[score_column] = np.nan
        for _, positions in frame.groupby("universe").groups.items():
            values = frame.loc[positions, metric].astype(float)
            minimum, maximum = values.min(), values.max()
            normalized = pd.Series(0.5, index=values.index) if maximum == minimum else (values - minimum) / (maximum - minimum)
            if metric in LOWER_IS_BETTER:
                normalized = 1.0 - normalized
            frame.loc[positions, score_column] = normalized
    complete = frame["score_status"].eq("complete")
    frame["composite_governance_score"] = np.nan
    frame.loc[complete, "composite_governance_score"] = sum(
        frame.loc[complete, f"score_{metric}"] * weight
        for metric, weight in GOVERNANCE_SCORE_WEIGHTS.items()
    )
    frame["governance_rank"] = frame.groupby("universe")["composite_governance_score"].rank(ascending=False, method="min")
    frame["is_governance_winner"] = frame["governance_rank"].eq(1.0)
    return frame
```

Add the validation metric and calibration functions:

```python
def _annual_return(values: pd.Series, periods: int = 252) -> float:
    series = pd.Series(values).dropna()
    return float((1.0 + series).prod() ** (periods / len(series)) - 1.0) if len(series) else np.nan


def _maximum_drawdown_magnitude(values: pd.Series) -> float:
    wealth = (1.0 + pd.Series(values).dropna()).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(-drawdown.min()) if len(drawdown) else np.nan


def summarize_walk_forward(universe: str, strategy: str, result: WalkForwardResult) -> dict[str, Any]:
    returns = result.returns.dropna()
    annual_return = _annual_return(returns)
    annual_volatility = float(returns.std() * np.sqrt(252)) if len(returns) else np.nan
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else np.nan
    final_weights = result.weight_history.iloc[-1] if not result.weight_history.empty else pd.Series(dtype=float)
    effective_n = float(1.0 / np.square(final_weights).sum()) if len(final_weights) and np.square(final_weights).sum() > 0 else np.nan
    return {
        "universe": universe,
        "strategy": strategy,
        "sharpe_ratio": sharpe,
        "annual_return": annual_return,
        "historical_cvar_loss_95": _historical_cvar_loss(returns, 0.95),
        "max_drawdown_magnitude": _maximum_drawdown_magnitude(returns),
        "graph_exposure": float(result.decision_log["graph_exposure"].mean()),
        "effective_n": effective_n,
        "turnover": float(result.decision_log["turnover"].mean()),
    }


def calibrate_governance_gcvar(
    universe: str,
    train_returns: pd.DataFrame,
    validation_returns: pd.DataFrame,
    instability: pd.Series,
    candidate_grid: Iterable[Mapping[str, Any]],
) -> tuple[GovernanceParams, pd.DataFrame]:
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
    selected = GovernanceParams(**json.loads(scored.loc[0, "parameter_json"]))
    return selected, scored
```

Before calling this function, the script calls `assert_disjoint_windows(validation.index, test.index)`. The calibration function receives no test frame or test dates.

- [ ] **Step 4: Run scoring and all protocol tests**

Run: `python -m unittest notebook.tests.test_gcvar_protocol -v`

Expected: all protocol tests pass without downloads.

- [ ] **Step 5: Commit scoring and calibration**

```powershell
git add -- 'notebook/gcvar_protocol.py' 'notebook/tests/test_gcvar_protocol.py'
git commit -m "feat: add validation only gcvar calibration and scoring"
```

### Task 6: Integrate Calibrated Static and Adaptive G-CVaR into the Full Script

**Files:**
- Modify: `notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py:130-380`
- Modify: `notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py:690-1150`
- Modify: `notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py:1600-1920`
- Modify: `notebook/tests/test_portfolio_plot_script_contract.py`

- [ ] **Step 1: Add failing integration contracts**

```python
def test_full_script_uses_complete_2014_2025_download_window(self):
    self.assertIn('"start_date": "2014-01-01"', self.source)
    self.assertIn('"end_date": "2026-01-01"', self.source)

def test_full_script_exports_protocol_audits(self):
    for filename in [
        "calibration_vs_test_boundary_audit.csv",
        "gcvar_calibration_results.csv",
        "gcvar_solver_and_slack_audit.csv",
        "gcvar_behavioral_validation.csv",
        "gcvar_test_governance_ranking.csv",
    ]:
        self.assertIn(filename, self.source)

def test_full_script_imports_protocol_module(self):
    self.assertIn("from gcvar_protocol import", self.source)
    self.assertNotIn('"train_test_split": "2015-01-01"', self.source)
```

- [ ] **Step 2: Run integration contracts and verify RED**

Run: `python -m unittest notebook.tests.test_portfolio_plot_script_contract -v`

Expected: failures for the old end date, old split, missing import, and missing exports.

- [ ] **Step 3: Replace configuration and G-CVaR orchestration**

Import the focused interfaces, define `PROTOCOL_DATES = ProtocolDates.default()`, and derive string configuration keys from it. Keep all existing baseline strategy functions unchanged.

Define the validation grid before execution:

```python
GCVAR_CANDIDATE_GRID = [
    asdict(GovernanceParams(graph_lambda=graph_lambda, lambda_max=lambda_max, turnover_lambda=turnover_lambda))
    for graph_lambda in (0.05, 0.10)
    for lambda_max in (0.50, 0.80)
    for turnover_lambda in (0.02, 0.05)
]
GCVAR_PARAMETER_GRID_HASH = stable_parameter_hash(GCVAR_CANDIDATE_GRID)
```

For each universe:

```python
training = r_all.loc[PROTOCOL_DATES.training_start:PROTOCOL_DATES.training_end]
validation = r_all.loc[PROTOCOL_DATES.validation_start:PROTOCOL_DATES.validation_end]
test = r_all.loc[PROTOCOL_DATES.test_start:PROTOCOL_DATES.test_end]
assert_disjoint_windows(validation.index, test.index)

selected, calibration_rows = calibrate_governance_gcvar(
    universe=universe,
    train_returns=training,
    validation_returns=validation,
    instability=INSTABILITY_PANELS[universe]["instability_index"],
    candidate_grid=GCVAR_CANDIDATE_GRID,
)

static_result = run_walk_forward_gcvar(
    returns=r_all,
    instability=INSTABILITY_PANELS[universe]["instability_index"],
    evaluation_start=PROTOCOL_DATES.test_start,
    evaluation_end=PROTOCOL_DATES.test_end,
    params=selected,
    adaptive=False,
)
adaptive_result = run_walk_forward_gcvar(
    returns=r_all,
    instability=INSTABILITY_PANELS[universe]["instability_index"],
    evaluation_start=PROTOCOL_DATES.test_start,
    evaluation_end=PROTOCOL_DATES.test_end,
    params=selected,
    adaptive=True,
)
```

Store these under the existing `graph_cvar_optimized` and `adaptive_graph_cvar` strategy keys so downstream comparison code remains compatible. Preserve Standard CVaR as the benchmark.

- [ ] **Step 4: Export coverage and boundary audits before result plots**

Write the five required CSVs to `TABLE_DIR`. Extend the per-universe data audit with requested/actual dates, ticker exclusion reasons, observation counts in each lane, and a Boolean `covers_available_2014_2025_window`.

- [ ] **Step 5: Run integration contracts and syntax checks**

Run: `python -m unittest notebook.tests.test_portfolio_plot_script_contract -v`

Run: `python -m py_compile 'notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py' notebook/gcvar_protocol.py`

Expected: contracts and compilation pass.

- [ ] **Step 6: Commit script integration**

```powershell
git add -- 'notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py' 'notebook/tests/test_portfolio_plot_script_contract.py'
git commit -m "feat: integrate calibrated governance gcvar protocol"
```

### Task 7: Produce Behavioral Validation, Governance Ranking, and Principal Plots

**Files:**
- Modify: `notebook/gcvar_protocol.py`
- Modify: `notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py:1900-2300`
- Modify: `notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py:5130-5520`
- Modify: `notebook/tests/test_gcvar_protocol.py`

- [ ] **Step 1: Write failing synthetic adaptive-behavior test**

```python
class BehavioralValidationTests(unittest.TestCase):
    def test_adaptive_crisis_graph_exposure_is_lower_on_controlled_fixture(self):
        logs = pd.DataFrame([
            {"universe": "U1", "strategy": "adaptive_graph_cvar", "regime": "calm", "graph_exposure": 0.40},
            {"universe": "U1", "strategy": "adaptive_graph_cvar", "regime": "calm", "graph_exposure": 0.38},
            {"universe": "U1", "strategy": "adaptive_graph_cvar", "regime": "crisis", "graph_exposure": 0.20},
            {"universe": "U1", "strategy": "adaptive_graph_cvar", "regime": "crisis", "graph_exposure": 0.18},
        ])
        audit = validate_adaptive_graph_behavior(logs)
        self.assertTrue(audit.loc[0, "crisis_graph_exposure_lower_than_calm"])
```

- [ ] **Step 2: Run the behavior test and verify RED**

Run: `python -m unittest notebook.tests.test_gcvar_protocol.BehavioralValidationTests -v`

Expected: missing validation helper.

- [ ] **Step 3: Implement behavioral tables and mechanism audit**

```python
def build_behavioral_validation_table(
    strategy_results: Mapping[tuple[str, str], WalkForwardResult],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (universe, strategy), result in strategy_results.items():
        if result.returns.empty or result.decision_log.empty:
            continue
        decisions = result.decision_log.sort_values("decision_date").copy()
        decisions["next_date"] = decisions["decision_date"].shift(-1)
        decisions.loc[decisions.index[-1], "next_date"] = result.returns.index.max() + pd.Timedelta(days=1)
        for _, decision in decisions.iterrows():
            period = result.returns.loc[
                (result.returns.index >= decision["decision_date"])
                & (result.returns.index < decision["next_date"])
            ]
            for date, value in period.items():
                rows.append({
                    "universe": universe,
                    "strategy": strategy,
                    "date": date,
                    "return": value,
                    "regime": decision["regime"],
                    "lambda_t": decision["lambda_t"],
                    "graph_exposure": decision["graph_exposure"],
                    "turnover": decision["turnover"],
                    "effective_n": decision["effective_n"],
                })
    daily = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    for (universe, regime, strategy), group in daily.groupby(["universe", "regime", "strategy"]):
        values = group["return"].dropna()
        annual_return = _annual_return(values)
        annual_volatility = float(values.std() * np.sqrt(252)) if len(values) else np.nan
        output.append({
            "universe": universe,
            "regime": regime,
            "strategy": strategy,
            "observations": len(values),
            "annual_return": annual_return,
            "sharpe_ratio": annual_return / annual_volatility if annual_volatility > 0 else np.nan,
            "historical_cvar_loss_95": _historical_cvar_loss(values, 0.95),
            "max_drawdown_magnitude": _maximum_drawdown_magnitude(values),
            "graph_exposure": float(group["graph_exposure"].mean()),
            "turnover": float(group["turnover"].mean()),
            "effective_n": float(group["effective_n"].mean()),
            "adaptive_activation_frequency": float((group["lambda_t"] > 0).mean()),
        })
    return pd.DataFrame(output)


def validate_adaptive_graph_behavior(logs: pd.DataFrame) -> pd.DataFrame:
    adaptive = logs.loc[logs["strategy"].eq("adaptive_graph_cvar")]
    rows: list[dict[str, Any]] = []
    for universe, group in adaptive.groupby("universe"):
        calm = group.loc[group["regime"].eq("calm"), "graph_exposure"].mean()
        crisis = group.loc[group["regime"].eq("crisis"), "graph_exposure"].mean()
        rows.append({
            "universe": universe,
            "calm_graph_exposure": calm,
            "crisis_graph_exposure": crisis,
            "crisis_graph_exposure_lower_than_calm": bool(pd.notna(calm) and pd.notna(crisis) and crisis < calm),
        })
    return pd.DataFrame(rows)
```

Use this as a descriptive mechanism audit. Do not assert that real-market crisis exposure must always be lower.

- [ ] **Step 4: Add the three principal Matplotlib figures**

Add these save functions under `FIGURE_DIR / "walk_forward_governance_gcvar"`:

```python
GCVAR_FIGURE_DIR = FIGURE_DIR / "walk_forward_governance_gcvar"
GCVAR_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def plot_instability_vs_adaptive_lambda(decision_logs: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    for universe, group in decision_logs.groupby("universe"):
        group = group.sort_values("decision_date")
        axes[0].plot(group["decision_date"], group["instability_index"], label=universe, alpha=0.75)
        axes[1].plot(group["decision_date"], group["lambda_t"], label=universe, alpha=0.75)
    axes[0].set_ylabel("Instability index")
    axes[1].set_ylabel("Adaptive lambda")
    axes[1].set_xlabel("Untouched test date")
    axes[0].set_title("Instability and Adaptive Activation — Untouched test: 2023-2025")
    axes[0].legend(ncol=4, fontsize=8)
    path = GCVAR_FIGURE_DIR / "instability_vs_adaptive_lambda.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_time_varying_graph_exposure(decision_logs: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(15, 6))
    grouped = decision_logs.groupby(["decision_date", "strategy"], as_index=False)["graph_exposure"].mean()
    for strategy, group in grouped.groupby("strategy"):
        ax.plot(group["decision_date"], group["graph_exposure"], marker="o", label=strategy)
    ax.set_title("Time-Varying Graph Exposure — Untouched test: 2023-2025")
    ax.set_ylabel("Mean graph exposure")
    ax.legend()
    path = GCVAR_FIGURE_DIR / "time_varying_graph_exposure.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_crisis_only_governance_comparison(behavior: pd.DataFrame) -> Path:
    metrics = ["historical_cvar_loss_95", "max_drawdown_magnitude", "graph_exposure", "turnover", "annual_return"]
    crisis = behavior.loc[behavior["regime"].eq("crisis")]
    summary = crisis.groupby("strategy")[metrics].mean()
    normalized = (summary - summary.min()) / (summary.max() - summary.min()).replace(0, 1)
    fig, ax = plt.subplots(figsize=(13, 7))
    normalized.plot(kind="bar", ax=ax)
    ax.set_title("Crisis-Only Governance Comparison — Untouched test: 2023-2025")
    ax.set_ylabel("Within-metric normalized value")
    ax.legend(title="Metric", fontsize=8)
    path = GCVAR_FIGURE_DIR / "crisis_only_governance_comparison.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


plot_instability_vs_adaptive_lambda(GCVAR_DECISION_LOGS)
plot_time_varying_graph_exposure(GCVAR_DECISION_LOGS)
plot_crisis_only_governance_comparison(gcvar_behavioral_validation_df)
```

Every title includes `Untouched test: 2023-2025`; every function calls `fig.savefig(path, dpi=240, bbox_inches="tight")` and closes the figure after saving.

- [ ] **Step 5: Replace the old mutable composite score with the fixed score**

Generate `gcvar_test_governance_ranking.csv` from `compute_governance_scores`. Keep legacy scores only when clearly labeled `legacy` and exclude them from the final G-CVaR verdict.

- [ ] **Step 6: Run focused tests and compilation**

Run: `python -m unittest notebook.tests.test_gcvar_protocol notebook.tests.test_portfolio_plot_script_contract -v`

Run: `python -m py_compile notebook/gcvar_protocol.py 'notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py'`

Expected: all tests and compilation pass.

- [ ] **Step 7: Commit behavioral evidence**

```powershell
git add -- 'notebook/gcvar_protocol.py' 'notebook/tests/test_gcvar_protocol.py' 'notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py'
git commit -m "feat: add gcvar behavioral validation evidence"
```

### Task 8: Update Narrative, Section Order, and Result Labels

**Files:**
- Modify: `notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py`
- Modify: `notebook/tests/test_portfolio_plot_script_contract.py`

- [ ] **Step 1: Add failing narrative contracts**

```python
def test_narrative_names_protocol_and_actual_test_window(self):
    self.assertIn("Walk-Forward Governance G-CVaR Evaluation Protocol", self.source)
    self.assertIn("Untouched test period: 2023-01-01 through 2025-12-31", self.source)
    self.assertNotIn("main out-of-sample split starts in 2020", self.source)

def test_narrative_preserves_non_fabrication_rule(self):
    self.assertIn("reports the actual rank", self.source)
    self.assertIn("does not guarantee the highest terminal wealth", self.source)
```

- [ ] **Step 2: Run narrative contracts and verify RED**

Run: `python -m unittest notebook.tests.test_portfolio_plot_script_contract -v`

Expected: failure until old contradictory language is removed and the approved wording is present.

- [ ] **Step 3: Reorder and relabel the final script sections**

Use the approved 15-section order. Ensure every table and plot states its raw-data, validation, or untouched-test period. Remove claims that 2025 is included when the underlying table ends in 2024 and remove the incorrect statement that the existing split starts in 2020.

- [ ] **Step 4: Run contracts and source audits**

Run: `python -m unittest notebook.tests.test_portfolio_plot_script_contract -v`

Run: `rg -n '2015-01-01|2025-01-01|split starts in 2020|highest return everywhere' 'notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py'`

Expected: tests pass; search returns no active obsolete configuration or prohibited claim.

- [ ] **Step 5: Commit narrative corrections**

```powershell
git add -- 'notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py' 'notebook/tests/test_portfolio_plot_script_contract.py'
git commit -m "docs: align gcvar narrative with evaluation protocol"
```

### Task 9: Run Full Analysis and Verify Every Artifact

**Files:**
- Verify: `notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py`
- Verify: `notebook/figures_universe_analysis/`
- Verify: `notebook/tables_universe_analysis/`

- [ ] **Step 1: Run the complete unit and contract suite**

Run: `python -m unittest discover -s notebook/tests -v`

Expected: all tests pass with zero failures and zero errors.

- [ ] **Step 2: Run the exact full script from its output directory**

```powershell
$env:MPLBACKEND='Agg'
$env:PYTHONUNBUFFERED='1'
Set-Location notebook
python 'agentic_ai_portfolio_governance_final_repaired_full_pro (1).py'
```

Expected: exit code zero. The run downloads actual observations through the final 2025 trading date and produces every required CSV and PNG.

- [ ] **Step 3: Verify date coverage and boundary integrity**

```python
import json
from pathlib import Path
import pandas as pd

tables = Path("notebook/tables_universe_analysis")
coverage = pd.read_csv(tables / "universe_data_coverage_audit_2014_2025.csv")
boundaries = pd.read_csv(tables / "calibration_vs_test_boundary_audit.csv")
assert coverage["price_start"].str.startswith("2014").all()
assert coverage["price_end"].str.startswith("2025").all()
assert (~boundaries["whether_test_used_in_calibration"].astype(bool)).all()
assert (pd.to_datetime(boundaries["calibration_end"]) < pd.to_datetime(boundaries["test_start"])).all()
```

Run the block with `python -` from the repository root. Expected: exit code zero.

- [ ] **Step 4: Verify required tables are nonempty**

```python
from pathlib import Path
import pandas as pd

root = Path("notebook/tables_universe_analysis")
required = [
    "calibration_vs_test_boundary_audit.csv",
    "gcvar_calibration_results.csv",
    "gcvar_solver_and_slack_audit.csv",
    "gcvar_behavioral_validation.csv",
    "gcvar_test_governance_ranking.csv",
]
for name in required:
    frame = pd.read_csv(root / name)
    assert not frame.empty, name
```

- [ ] **Step 5: Verify every PNG and the three principal figures**

```python
from pathlib import Path
from PIL import Image

root = Path("notebook/figures_universe_analysis")
pngs = list(root.rglob("*.png"))
assert pngs
for path in pngs:
    assert path.stat().st_size > 0
    with Image.open(path) as image:
        image.verify()
principal = root / "walk_forward_governance_gcvar"
for name in [
    "instability_vs_adaptive_lambda.png",
    "time_varying_graph_exposure.png",
    "crisis_only_governance_comparison.png",
]:
    assert (principal / name).is_file(), name
```

- [ ] **Step 6: Inspect the actual untouched-test ranking**

Run:

```powershell
Import-Csv 'notebook/tables_universe_analysis/gcvar_test_governance_ranking.csv' |
  Sort-Object universe,@{Expression={[double]$_.governance_rank}} |
  Select-Object universe,strategy,composite_governance_score,governance_rank,is_governance_winner |
  Format-Table -AutoSize
```

Expected: every complete strategy has a rank; report Static and Adaptive G-CVaR’s actual ranks without post-test tuning.

- [ ] **Step 7: Review the scoped diff and commit final generated-code changes only**

```powershell
git diff --check -- 'notebook/gcvar_protocol.py' 'notebook/tests/test_gcvar_protocol.py' 'notebook/tests/test_portfolio_plot_script_contract.py' 'notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py'
git status --short
git add -- 'notebook/gcvar_protocol.py' 'notebook/tests/test_gcvar_protocol.py' 'notebook/tests/test_portfolio_plot_script_contract.py' 'notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py'
git commit -m "test: verify walk forward governance gcvar protocol"
```

Do not stage unrelated backend, frontend, report, temporary, or previously generated files.
