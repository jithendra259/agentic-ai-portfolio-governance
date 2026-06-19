# Adaptive G-CVaR Gate Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate Adaptive G-CVaR around a look-ahead-safe q80 gate, separate gate strength from optimizer penalty, and validate every generated plot.

**Architecture:** Put all authoritative gate behavior in `gcvar_protocol.py`. The full analysis script runs the same calibrated parameters over validation and untouched test lanes, exports separate evidence, and performs deterministic image QA after plot generation.

**Tech Stack:** Python, pandas, NumPy, CVXPY, Matplotlib, Pillow, unittest.

---

### Task 1: Specify gate semantics with failing tests

**Files:**
- Modify: `notebook/tests/test_gcvar_protocol.py`

- [ ] Add tests asserting a q80 boundary, `lambda_gate` in `[0,1]`, zero calm activation, and `active_graph_lambda == graph_lambda * lambda_gate`.
- [ ] Run `python -m unittest notebook.tests.test_gcvar_protocol -v` and confirm the new field assertions fail.

### Task 2: Consolidate the protocol implementation

**Files:**
- Modify: `notebook/gcvar_protocol.py`

- [ ] Return gate and historical boundary from the adaptive gate calculation.
- [ ] Log `instability_threshold`, `lambda_gate`, and `active_graph_lambda` separately.
- [ ] Pass only `active_graph_lambda` into `optimize_governance_gcvar` for adaptive runs.
- [ ] Preserve `lambda_t` as an explicit compatibility alias for `lambda_gate`, with no ambiguous optimizer semantics.
- [ ] Run the protocol unit tests and confirm they pass.

### Task 3: Export validation and untouched-test evidence separately

**Files:**
- Modify: `notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py`
- Modify: `notebook/tests/test_portfolio_plot_script_contract.py`

- [ ] Add contract tests for separate validation/test behavioral tables, gate audits, and figures.
- [ ] Retain validation walk-forward results during calibration without using test observations.
- [ ] Generate validation evidence for 2020-2022 and final evidence for 2023-2025.
- [ ] Ensure crisis comparisons state when a lane has no qualifying crisis observations.
- [ ] Run all notebook tests and confirm they pass.

### Task 4: Remove contradictory legacy reporting

**Files:**
- Modify: `notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py`

- [ ] Route final adaptive plots and audits exclusively through protocol decision logs.
- [ ] Rename any retained fixed-threshold analysis as legacy exploratory evidence.
- [ ] Ensure final claims describe only untouched-test realized results for 2023-2025.
- [ ] Compile the full script with `python -m py_compile`.

### Task 5: Execute and evaluate all plots

**Files:**
- Modify: `notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py`
- Generate: `notebook/tables_universe_analysis/plot_quality_audit.csv`

- [ ] Run the complete script with `MPLBACKEND=Agg`.
- [ ] Verify all required CSV files are nonempty and calibration/test boundaries remain disjoint.
- [ ] Open every PNG with Pillow and calculate dimensions, file size, grayscale variance, entropy, edge clipping ratios, and near-white fraction.
- [ ] Export one QA row per figure and manually inspect every flagged image plus every principal G-CVaR plot.
- [ ] Correct genuine plotting defects, rerun, and repeat QA until no unexplained critical flags remain.
- [ ] Re-run all tests and report actual Adaptive/Static G-CVaR ranks without changing scores.
