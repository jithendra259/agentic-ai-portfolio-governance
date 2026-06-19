# Supplemental Linear-Centrality Adaptive G-CVaR V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a supplemental linear-centrality Adaptive G-CVaR V2 across all 11 universes without modifying the completed quadratic protocol, and export family-separated NaN-safe rankings.

**Architecture:** `notebook/gcvar_v2.py` owns V2 data, signal, optimization, walk-forward, and ranking functions. The full Python analysis imports this module after the primary protocol, runs it over the untouched 2023-2025 lane, appends V2 only to supplemental comparisons, and exports explicit audits.

**Tech Stack:** Python, pandas, NumPy, CVXPY, NetworkX, Matplotlib, unittest.

---

### Task 1: Add V2 graph and signal primitives

**Files:**
- Create: `notebook/gcvar_v2.py`
- Create: `notebook/tests/test_gcvar_v2.py`

- [ ] **Step 1: Write failing graph-source and gate tests**

Add tests that call the wished-for API:

```python
def test_missing_holdings_uses_correlation_proxy(self):
    penalty, source = get_linear_graph_penalty(
        self.returns, self.returns.columns, pd.Timestamp("2023-03-31"),
        holdings=pd.DataFrame(), threshold=0.0,
    )
    self.assertEqual(source, "correlation_proxy")
    self.assertTrue(penalty.between(0, 1).all())

def test_holdings_filter_is_publication_date_safe(self):
    H = build_holdings_matrix(
        self.holdings, ["A", "B"], pd.Timestamp("2023-05-01")
    )
    self.assertNotIn("future_manager", H.index)

def test_activation_uses_multiplier_not_effective_lambda(self):
    signal = adaptive_gate_signal(2.0, theta=0.0, steepness=2.0, graph_lambda=0.0025)
    self.assertEqual(signal.active, signal.multiplier > 0.5)
    self.assertLess(signal.effective_lambda, 0.5)
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```powershell
python -m unittest notebook.tests.test_gcvar_v2 -v
```

Expected: import failure because `gcvar_v2.py` does not exist.

- [ ] **Step 3: Implement graph and signal functions**

Implement these public interfaces:

```python
@dataclass(frozen=True)
class AdaptiveGateSignal:
    instability: float
    threshold: float
    steepness: float
    multiplier: float
    effective_lambda: float
    active: bool

def load_clean_13f_holdings(path: Path, publication_lag_days: int = 45) -> pd.DataFrame: ...
def build_holdings_matrix(holdings, tickers, asof_date) -> pd.DataFrame: ...
def institutional_centrality_penalty(H, tickers) -> pd.Series: ...
def correlation_centrality_penalty(returns, threshold=0.30) -> pd.Series: ...
def get_linear_graph_penalty(returns, tickers, asof_date, holdings, threshold=0.30) -> tuple[pd.Series, str]: ...
def expanding_zscore(values, min_periods=60) -> pd.Series: ...
def compute_instability_series(returns, window=126) -> pd.DataFrame: ...
def calibrate_gate(instability, validation_start, validation_end, target_frequency=0.10) -> tuple[float, float]: ...
def adaptive_gate_signal(instability, theta, steepness, graph_lambda) -> AdaptiveGateSignal: ...
```

The correlation penalty uses absolute correlations above the threshold, zero diagonal, eigenvector centrality with weighted-degree fallback, and min-max normalization. Calibration slices validation dates only.

- [ ] **Step 4: Run tests and confirm GREEN**

Run the Task 1 command and expect all tests to pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- notebook/gcvar_v2.py notebook/tests/test_gcvar_v2.py
git commit -m "feat: add supplemental gcvar v2 signals"
```

### Task 2: Add the linear-centrality CVaR optimizer and walk-forward engine

**Files:**
- Modify: `notebook/gcvar_v2.py`
- Modify: `notebook/tests/test_gcvar_v2.py`

- [ ] **Step 1: Write failing optimizer and look-ahead tests**

```python
def test_linear_optimizer_returns_feasible_weights(self):
    weights, audit = optimize_linear_centrality_cvar(
        self.returns, pd.Series([0.1, 0.4, 0.7, 1.0], index=self.returns.columns),
        effective_lambda=0.0025, params=LinearGCVarParams(max_weight=0.4),
    )
    self.assertAlmostEqual(weights.sum(), 1.0, places=7)
    self.assertTrue((weights >= 0).all())
    self.assertLessEqual(weights.max(), 0.4 + 1e-6)
    self.assertEqual(audit["graph_objective_type"], "linear_centrality")

def test_walk_forward_never_uses_current_or_future_returns(self):
    result = run_linear_gcvar_walk_forward(...)
    self.assertTrue((result.audit["training_end"] < result.audit["decision_date"]).all())
    self.assertTrue(result.audit["turnover"].notna().all())
```

- [ ] **Step 2: Run the two new tests and confirm RED**

Expected: missing `LinearGCVarParams`, optimizer, and walk-forward interfaces.

- [ ] **Step 3: Implement the exact V2 objective**

Add:

```python
@dataclass(frozen=True)
class LinearGCVarParams:
    alpha: float = 0.95
    max_weight: float = 0.30
    return_tradeoff: float = 0.25
    graph_lambda: float = 0.0025
    lookback_days: int = 756
    rebalance_frequency: str = "QE"
    graph_threshold: float = 0.30

def optimize_linear_centrality_cvar(returns, centrality, effective_lambda, params):
    losses = -returns.to_numpy() @ w
    cvar = nu + cp.sum(u) / ((1 - params.alpha) * len(returns))
    objective = cvar - params.return_tradeoff * (returns.mean().to_numpy() @ w)
    objective += effective_lambda * (centrality.to_numpy() @ w)
```

Use only the CVaR auxiliary constraints, long-only budget, and maximum weight cap. Return solver status and `graph_objective_type="linear_centrality"`.

- [ ] **Step 4: Implement quarterly walk-forward evaluation**

Add `LinearGCVarWalkForwardResult` containing returns, weights, audit, and instability. Calibrate `theta` and `k` only from 2020-2022. Evaluate only 2023-2025. At each decision, use returns and holdings strictly available before that date, record the graph source, multiplier, effective lambda, active flag, linear graph exposure, turnover, HHI, and effective N.

- [ ] **Step 5: Run all V2 tests and confirm GREEN**

Run:

```powershell
python -m unittest notebook.tests.test_gcvar_v2 -v
```

- [ ] **Step 6: Commit Task 2**

```powershell
git add -- notebook/gcvar_v2.py notebook/tests/test_gcvar_v2.py
git commit -m "feat: add linear centrality gcvar optimizer"
```

### Task 3: Add family-separated NaN-safe rankings

**Files:**
- Modify: `notebook/gcvar_v2.py`
- Modify: `notebook/tests/test_gcvar_v2.py`

- [ ] **Step 1: Write failing ranking tests**

```python
def test_incomplete_governance_row_is_rejected(self):
    rankings, rejected = compute_family_rankings(self.metrics_with_nan)
    self.assertNotIn("missing_turnover", set(rankings["strategy"]))
    self.assertEqual(
        rejected.set_index("strategy").loc["missing_turnover", "rejection_reason"],
        "missing_required_governance_metrics:turnover",
    )

def test_ranking_families_are_separate(self):
    rankings, _ = compute_family_rankings(self.complete_metrics)
    self.assertEqual(
        set(rankings["ranking_family"]), {"core", "supplemental", "hitl_simulation"}
    )
```

- [ ] **Step 2: Run tests and confirm RED**

Expected: `compute_family_rankings` is absent.

- [ ] **Step 3: Implement fixed-metric eligibility and ranking**

Use these required fields:

```python
REQUIRED_GOVERNANCE_METRICS = (
    "annual_return", "annual_volatility", "sharpe_ratio", "sortino_ratio",
    "historical_cvar_loss_95", "max_drawdown_magnitude", "turnover", "hhi",
    "effective_n", "graph_exposure",
)
```

Reject any row with missing or non-finite required values. Score eligible rows within each `(universe, ranking_family)` only, using frozen weights and directions declared as module constants. Return both ranked and rejected DataFrames.

- [ ] **Step 4: Run V2 tests and confirm GREEN**

- [ ] **Step 5: Commit Task 3**

```powershell
git add -- notebook/gcvar_v2.py notebook/tests/test_gcvar_v2.py
git commit -m "feat: add nan safe governance rankings"
```

### Task 4: Integrate V2 into the full Python analysis

**Files:**
- Modify: `notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py`
- Modify: `notebook/tests/test_portfolio_plot_script_contract.py`

- [ ] **Step 1: Write failing integration contract tests**

Require the full script to contain the V2 label and all output filenames, and verify the V2 section occurs after `FINAL WALK-FORWARD GOVERNANCE G-CVaR EVALUATION PROTOCOL`:

```python
def test_v2_is_supplemental_and_runs_after_primary_protocol(self):
    marker = self.source.index("FINAL WALK-FORWARD GOVERNANCE G-CVaR EVALUATION PROTOCOL")
    v2 = self.source.index("Supplemental Linear-Centrality Adaptive G-CVaR")
    self.assertGreater(v2, marker)
    self.assertIn('"adaptive_graph_cvar_v2"', self.source[v2:])
```

- [ ] **Step 2: Run contract tests and confirm RED**

- [ ] **Step 3: Add the supplemental execution block**

After the primary results are finalized:

1. load the optional local holdings CSV without downloading;
2. run V2 over every `WORKING_UNIVERSES` return panel;
3. calculate portfolio metrics plus finite turnover, HHI, effective N, and graph exposure;
4. append V2 rows to a copy of `ten_algo_results_df` for supplemental ranking only;
5. classify primary strategies as `core`, V2/fixed-quarterly as `supplemental`, and sample HITL as `hitl_simulation`;
6. export the audit, weights, results, activation summary, three family ranking files, rejection file, and technical checks.

- [ ] **Step 4: Add explicit technical checks**

The checks CSV must prove: 11 V2 universes generated, fallback source recorded, nonzero multiplier activation, no test data in gate calibration, finite V2 governance metrics, incomplete rows rejected, supplemental rows excluded from core rankings, and the frozen protocol module unchanged.

- [ ] **Step 5: Run contracts, unit tests, and compilation**

```powershell
python -m unittest discover -s notebook/tests -v
python -m py_compile notebook/gcvar_v2.py "notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py"
git diff --exit-code 68c844a -- notebook/gcvar_protocol.py
```

Expected: tests pass, compilation succeeds, and no diff exists for the frozen primary protocol module.

- [ ] **Step 6: Commit Task 4**

```powershell
git add -- notebook/gcvar_v2.py notebook/tests/test_gcvar_v2.py notebook/tests/test_portfolio_plot_script_contract.py "notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py"
git commit -m "feat: integrate supplemental adaptive gcvar v2"
```

### Task 5: Execute, audit, and report the supplemental experiment

**Files:**
- Generate: `notebook/tables_universe_analysis/*.csv`
- Generate: `notebook/figures_universe_analysis/*.png`

- [ ] **Step 1: Run the complete Python analysis**

```powershell
$env:MPLBACKEND="Agg"
python "agentic_ai_portfolio_governance_final_repaired_full_pro (1).py"
```

Run from `notebook/`. Expected: exit code 0 and all 11 V2 universes reported.

- [ ] **Step 2: Verify V2 outputs and strict boundaries**

Assert required files are nonempty, every V2 audit has `training_end < decision_date`, every graph source is an allowed value, all V2 governance fields are finite, and all technical checks pass.

- [ ] **Step 3: Evaluate empirical results honestly**

Report V2 rank and activation frequency by universe, comparison with the original Adaptive G-CVaR, graph-source usage, solver fallbacks, and rejected ranking rows. Do not alter parameters, family assignments, metrics, or score weights after reading the test results.

- [ ] **Step 4: Run final verification**

```powershell
python -m unittest discover -s notebook/tests -v
python -m py_compile notebook/gcvar_v2.py "notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py"
```

Re-open every new V2 CSV and figure, then confirm the frozen quadratic protocol output files remain present.
