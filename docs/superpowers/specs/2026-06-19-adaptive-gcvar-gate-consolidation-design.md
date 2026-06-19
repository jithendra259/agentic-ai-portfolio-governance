# Adaptive G-CVaR Gate Consolidation Design

## Goal

Make the rolling quantile gate the single authoritative Adaptive G-CVaR mechanism, expose its optimizer penalty separately, and produce honest validation- and test-period behavioral evidence.

## Root cause

The full script contains a legacy fixed-threshold adaptive implementation and the newer walk-forward q80 implementation. Legacy diagnostics therefore report a dormant gate even when the final protocol activates. The protocol also stores its gate directly as `lambda_t`, obscuring the distinction between a unitless activation gate and the graph penalty supplied to the optimizer.

## Design

- `lambda_gate` is a unitless value in `[0, 1]`, computed only from observations strictly before each decision date.
- Its boundary is the expanding historical q80 instability quantile.
- It is exactly zero below the boundary and sigmoid-scaled above it.
- `active_graph_lambda = graph_lambda * lambda_gate` is the only adaptive graph coefficient passed to the optimizer.
- Static G-CVaR continues to use `graph_lambda` directly.
- Decision logs and audits store the boundary, gate, and active coefficient separately.
- Behavioral evidence is generated independently for validation (2020-2022) and untouched test (2023-2025).
- Legacy diagnostics consume the authoritative protocol fields or are relabeled as legacy; they cannot present the fixed 1.75 rule as final protocol evidence.

## Outputs

- `gcvar_validation_behavioral_validation.csv`
- `gcvar_test_behavioral_validation.csv`
- `gcvar_adaptive_gate_audit_validation.csv`
- `gcvar_adaptive_gate_audit_test.csv`
- validation and test versions of instability/gate and crisis-comparison figures
- unchanged untouched-test governance ranking with actual results

## Verification

- Unit tests prove no look-ahead, bounded gate values, separate active coefficients, and optimizer use of the active coefficient.
- Contract tests prove separate validation/test exports and explicit period wording.
- The full script runs successfully using 2014-2025 market data.
- Every generated PNG is opened and checked for nonzero dimensions, corruption, blank/near-blank rendering, clipped layout indicators, and suspicious visual distributions; findings are exported as a plot QA manifest and manually reviewed where flagged.

No score weights or test-selected parameters may be altered after observing the untouched test outcome.
