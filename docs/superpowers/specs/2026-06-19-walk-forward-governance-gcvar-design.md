# Walk-Forward Governance G-CVaR Evaluation Protocol Design

## Goal

Improve Static and Adaptive G-CVaR as mathematically structured governance optimizers, evaluate them without look-ahead bias, and report whether they outperform baselines on a fixed governance-adjusted score. The implementation must never alter parameters, score weights, or test boundaries after observing test results.

## Research Positioning

- **Static G-CVaR:** fixed graph-aware tail-risk optimizer.
- **Adaptive G-CVaR:** instability-gated graph-aware tail-risk optimizer.
- G-CVaR is not claimed to maximize terminal wealth everywhere.
- Final superiority is determined by an immutable composite governance score on an untouched test period.
- If Adaptive G-CVaR does not rank first, the actual rank and failure dimensions are reported.

## Data Coverage and Experimental Boundaries

Yahoo Finance uses an exclusive end date. The configured acquisition window will therefore be:

```python
start_date = "2014-01-01"
end_date = "2026-01-01"
```

This requests observations from 2014-01-01 through 2025-12-31. Actual first and last trading dates are recorded rather than assumed.

The experiment uses fixed boundaries:

| Lane | Requested dates | Purpose |
| --- | --- | --- |
| Training | 2014-01-01 through 2019-12-31 | Estimate return, CVaR, tail graph, and instability components |
| Validation | 2020-01-01 through 2022-12-31 | Select G-CVaR parameters using governance score |
| Test | 2023-01-01 through 2025-12-31 | Final comparison and reporting only |

Raw-data coverage and strategy-result coverage are distinct. Descriptive data audits may span 2014–2025, while validation and test results must be labeled with their actual windows. No result beginning in 2023 may be described as a full-period 2014–2025 backtest.

Every universe receives its own aligned price and return panel. Coverage exports must include requested tickers, available tickers, first/last price date, first/last return date, observation count, missingness, and exclusion reasons. Silent global intersection truncation is prohibited.

## Mathematical Design

### Tail-Graph Matrix

For each optimization date, use only returns available before that date. Select downside-market observations using a training-only tail quantile, estimate absolute tail correlations, threshold weak edges, symmetrize the adjacency matrix, and project it to positive semidefinite form.

The graph exposure is:

```text
G(w) = w^T A w
```

This penalizes allocation to connected downside-risk clusters without forbidding ownership of individually central assets.

### Normalized Governance G-CVaR Optimizer

The objective combines normalized CVaR, normalized expected return, normalized graph exposure, turnover, and soft-constraint slack:

```text
CVaR(w) / CVaR(EW)
- gamma * expected_return(w) / abs(expected_return(EW))
+ lambda_t * G(w) / G(EW)
+ tau * ||w - previous_weights||_1
+ M * (cvar_slack + graph_slack + return_slack)
```

Constraints:

- weights sum to one;
- long-only weights;
- configurable maximum weight;
- CVaR target relative to Equal Weight with nonnegative slack;
- graph-exposure target relative to Standard CVaR with nonnegative slack;
- expected-return floor relative to Equal Weight with nonnegative slack.

All normalization denominators use guarded positive floors. Solver status, fallback path, and all slack values are exported.

### Static and Adaptive Behavior

Static G-CVaR uses a fixed graph penalty selected on validation data.

Adaptive G-CVaR uses a rolling, training-only instability threshold:

```text
lambda_t = 0                              below historical q80
lambda_max * sigmoid(k * (I_t - q80))     at or above historical q80
```

The quantile at date `t` is calculated only from instability observations before `t`. In calm regimes Adaptive G-CVaR should remain close to Standard CVaR. In crisis regimes it should reduce graph exposure while controlling CVaR, drawdown, and turnover.

## Calibration Protocol

Candidate parameters include graph penalty, adaptive maximum penalty, instability quantile, sigmoid steepness, turnover penalty, and soft-dominance ratios. The candidate grid is defined before test execution.

For each universe:

1. Estimate the initial model using training data.
2. Run walk-forward validation from 2020 through 2022.
3. Score each candidate using only validation observations.
4. Apply deterministic tie-breaking: governance score, then CVaR, then graph exposure, then lower turnover, then lexical parameter order.
5. Freeze selected parameters.
6. Run the untouched 2023–2025 test once for final reporting.

The test period cannot influence candidate generation, selection, score normalization, score weights, thresholds, or fallback decisions.

## Governance Score

The immutable test score is:

| Component | Weight | Direction |
| --- | ---: | --- |
| Sharpe ratio | 0.20 | Higher is better |
| Annual return | 0.15 | Higher is better |
| CVaR loss | 0.25 | Lower is better |
| Maximum drawdown magnitude | 0.15 | Lower is better |
| Graph exposure | 0.10 | Lower is better |
| Diversification/effective N | 0.10 | Higher is better |
| Turnover | 0.05 | Lower is better |

Components are normalized within each universe across all eligible strategies using one documented normalization rule. Missing components do not silently reweight a strategy; the row is marked incomplete unless a predeclared fallback applies.

## Function Interfaces

The implementation introduces focused helpers with these responsibilities:

- `make_tail_graph_psd_matrix(returns, tail_quantile, threshold)` returns an aligned PSD adjacency matrix plus diagnostics.
- `optimize_governance_gcvar(returns, graph_matrix, lambda_t, previous_weights, params)` returns weights plus solver and slack diagnostics.
- `adaptive_lambda_quantile(history, current_value, lambda_max, quantile, steepness)` returns the current graph penalty without future observations.
- `calibrate_governance_gcvar(train_returns, validation_returns, candidate_grid)` returns frozen parameters and candidate-level evidence.
- `run_walk_forward_gcvar(returns, evaluation_start, evaluation_end, frozen_params, adaptive)` returns realized returns, weights, graph exposures, regimes, and decision logs.
- `build_behavioral_validation_table(...)` returns regime/strategy metrics.
- `compute_governance_scores(...)` returns component scores, composite score, completeness, and rank.

## Output Schemas

### `calibration_vs_test_boundary_audit.csv`

- universe
- raw_data_start
- raw_data_end
- training_start
- training_end
- calibration_start
- calibration_end
- test_start
- test_end
- selected_params
- parameter_grid_hash
- whether_test_used_in_calibration, always `False`

### `gcvar_calibration_results.csv`

- universe
- candidate parameters
- validation component metrics
- validation governance score
- selected flag
- tie-break fields

### `gcvar_solver_and_slack_audit.csv`

- universe, strategy, optimization date
- solver and status
- CVaR, graph, and return slack
- fallback indicator and reason
- maximum/sum weight checks

### `gcvar_behavioral_validation.csv`

- universe and regime
- strategy
- observation count
- return, Sharpe, CVaR loss, maximum drawdown
- graph exposure, turnover, effective N
- adaptive activation frequency

### `gcvar_test_governance_ranking.csv`

- universe and strategy
- raw metrics
- normalized component scores
- composite governance score
- completeness status
- rank and winner indicator

All exports include their actual measurement window where relevant.

## Visual Evidence

The principal behavioral figures are:

1. Instability index versus adaptive lambda, with regime thresholds.
2. Time-varying graph exposure for Standard CVaR, Static G-CVaR, and Adaptive G-CVaR.
3. Crisis-only governance comparison for CVaR loss, drawdown, graph exposure, turnover, and return.

Existing terminal-value plots remain secondary evidence. Titles and captions must identify the measurement period and must not imply guaranteed G-CVaR dominance.

## Validation and Testing

Automated tests cover:

- tail-graph symmetry and PSD eigenvalues;
- finite, aligned graph matrices;
- calm lambda equal to zero and crisis lambda positive/bounded;
- feasible long-only normalized weights;
- turnover penalty reducing weight changes in a controlled fixture;
- soft constraints yielding nonnegative, reported slack;
- calibration receiving no test observations;
- every optimization date using only strictly earlier data;
- fixed governance-score weights and normalization behavior;
- correct Yahoo exclusive-end handling and per-universe 2014–2025 coverage audit;
- exported boundary audit setting `whether_test_used_in_calibration=False`;
- Adaptive G-CVaR crisis graph exposure being lower than its calm exposure on a deterministic synthetic behavior fixture.

The last test validates mechanism behavior on controlled data. Real-market test results remain empirical and are never forced to satisfy a dominance assertion.

## Notebook Section Order

1. Research claim and non-fabrication rule
2. Configuration and immutable date boundaries
3. Data download, per-universe alignment, and coverage audit
4. Training/validation/test boundary audit
5. Tail-graph construction and diagnostics
6. Normalized governance G-CVaR optimizer
7. Static and adaptive mechanisms
8. Training and validation calibration
9. Frozen-parameter untouched test execution
10. Behavioral validation table
11. Governance ranking
12. Three principal behavioral plots
13. Terminal wealth and secondary comparisons
14. Solver, slack, coverage, and look-ahead audits
15. Thesis-safe interpretation

## Completion Criteria

- The script completes end to end with exit code zero.
- Raw market data covers the available trading observations from 2014 through 2025 for every retained universe.
- Boundary, calibration, solver/slack, behavioral, and ranking exports exist and are nonempty.
- All principal plots are valid nonempty images.
- Tests prove date separation and no look-ahead behavior.
- Final reporting states the actual untouched-test winner and rank of both G-CVaR variants.
