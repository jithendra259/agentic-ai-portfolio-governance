# Supplemental Linear-Centrality Adaptive G-CVaR V2 Design

## Goal

Add `adaptive_graph_cvar_v2` as a supplemental experiment using a linear centrality penalty, while freezing the completed quadratic G-CVaR protocol and producing NaN-safe, family-separated rankings.

## Frozen primary protocol

The existing primary thesis strategies and their implementation remain unchanged:

- Standard CVaR
- Static G-CVaR
- Adaptive G-CVaR
- Fixed Quarterly G-CVaR
- Sample HITL-Governed Adaptive G-CVaR

Their authoritative graph-risk term remains

\[
\lambda_t w^\top A_t w.
\]

No V2 code may alter `optimize_governance_gcvar`, `make_tail_graph_psd_matrix`, the current calibration grid, existing score weights, or existing primary-protocol outputs.

## V2 modules

### Data and graph module

V2 consumes the existing per-universe Yahoo Finance return panels. At each rebalance date it constructs a centrality vector `c_t` using:

1. a locally supplied, publication-date-safe `data/sec13f_holdings_clean.csv`, when available; or
2. a correlation-network proxy when the file is absent or contains no usable observations for the universe.

No SEC data is downloaded. Every decision log records either `correlation_proxy` or `sec_13f_institutional_coownership` as `graph_source`. The optional local loader accepts `filing_date`, or derives it as `report_date + 45 days`, and never uses a filing published after the decision date.

### Signal module

V2 uses the look-ahead-safe instability components already specified by the project: covariance drift, rolling volatility, and mean off-diagonal correlation, standardized using expanding statistics shifted by one observation.

Only 2020-2022 validation observations calibrate:

\[
\theta = Q_{0.90}(I_{validation}), \qquad
k = \log(9)/IQR(I_{validation}).
\]

For test decision date `t`:

\[
m_t = \frac{1}{1 + e^{-k(I_t-\theta)}}, \qquad
\lambda_t = \lambda_{base}m_t.
\]

Activation is always measured as `m_t > 0.5`; it is never inferred from an absolute effective-lambda threshold.

### Optimization module

For each test rebalance date, V2 solves:

\[
\min_{w,\nu,u}
\nu + \frac{1}{(1-\alpha)T}\sum_s u_s
- \eta\mu^\top w
+ \lambda_t c_t^\top w
\]

subject to:

\[
u_s \ge -r_s^\top w-\nu,\quad u_s\ge0,\quad
\mathbf{1}^\top w=1,\quad0\le w_i\le w_{max}.
\]

V2 runs quarterly over the untouched 2023-2025 test lane using only observations available before each decision. Its audit stores the gate, effective lambda, graph source, solver status, weights, turnover, HHI, effective N, and linear graph exposure.

## Integration and outputs

V2 is implemented in a separate module so the frozen primary optimizer cannot be modified accidentally:

- `notebook/gcvar_v2.py`
- `notebook/tests/test_gcvar_v2.py`

The full Python analysis imports and runs V2 after the primary protocol, then exports:

- `adaptive_graph_cvar_v2_audit.csv`
- `adaptive_graph_cvar_v2_results.csv`
- `adaptive_graph_cvar_v2_weights.csv`
- `adaptive_graph_cvar_v2_activation_summary.csv`
- `nan_safe_core_governance_ranking.csv`
- `nan_safe_supplemental_governance_ranking.csv`
- `nan_safe_hitl_simulation_ranking.csv`
- `nan_safe_governance_rejections.csv`
- `final_technical_validation_checks.csv`

V2 is appended to comparison data under the label `Supplemental Linear-Centrality Adaptive G-CVaR`. It is never relabeled as the primary Adaptive G-CVaR.

## NaN-safe ranking

Ranking is performed separately for:

1. core thesis strategies;
2. supplemental strategies, including V2; and
3. HITL simulations.

A row is eligible only when all fixed required governance metrics are finite. Ineligible rows receive no composite score or rank and are exported with an explicit rejection reason. Scores are computed only within universe and ranking family. Score directions and weights are fixed before reading test outcomes.

Required fields are annual return, annual volatility, Sharpe, Sortino, CVaR loss, drawdown magnitude, turnover, HHI, effective N, and graph exposure. V2 computes turnover from successive allocations rather than leaving it missing.

## Verification

Tests must prove:

- primary-protocol source hashes are unchanged by the V2 patch;
- the V2 objective uses `c_t @ w`, not `w.T @ A_t @ w`;
- the correlation fallback is selected when the 13F file is absent;
- 13F filtering uses only public filing dates at or before the decision date;
- gate calibration uses validation only and `active == (lambda_multiplier > 0.5)`;
- every optimization date trains strictly on earlier observations;
- weights are feasible and turnover is finite;
- incomplete ranking rows are rejected rather than silently reweighted;
- core, supplemental, and HITL rankings are separate;
- all 11 universes produce V2 results or an explicit failure audit.

The final report states observed results without changing parameters or ranking weights after the untouched test is evaluated.

## Thesis wording

The original Static and Adaptive G-CVaR models remain the primary thesis protocol because they use the quadratic graph exposure term \(w^\top A_t w\), preserving pairwise graph-risk interaction. A separate Adaptive G-CVaR V2 is added as a supplemental extension using a linear centrality penalty \(c_t^\top w\). V2 is not used to replace the original protocol, because doing so would alter the mathematical definition of the completed experiment. In the absence of SEC 13F holdings data, V2 transparently uses the correlation-network fallback.
