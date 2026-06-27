# Portfolio Script Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the exported portfolio-governance script to completion and improve only confirmed correctness or runtime failures.

**Architecture:** Preserve the single generated Python script and its existing output contracts. Add configurable validation at failure boundaries, then use full execution plus CSV QA as the integration test.

**Tech Stack:** Python 3.13, pandas, NumPy, yfinance, CVXPY, NetworkX, Matplotlib, unittest.

---

### Task 1: Repair the central train/test gate

**Files:**
- Modify: `notebook/agentic_ai_portfolio_governance_final_repaired_full_pro.py`
- Create: `notebook/tests/test_portfolio_script_contract.py`

- [ ] Write a source-contract test requiring a configurable minimum training count and descriptive empty-backtest guard.
- [ ] Run the test and verify it fails on the current script.
- [ ] Add `min_training_observations=240`, replace hard-coded central gates, and add the guard.
- [ ] Run the focused test and Python compilation.

### Task 2: Execute and diagnose the complete pipeline

**Files:**
- Modify only if demonstrated by a new runtime failure: `notebook/agentic_ai_portfolio_governance_final_repaired_full_pro.py`

- [ ] Run the script with `MPLBACKEND=Agg` and unbuffered output.
- [ ] Record the exact failing line or successful completion boundary.
- [ ] For each new code defect, add a focused failing contract test before the minimal repair.
- [ ] Repeat until the script exits successfully or an external blocker is proven.

### Task 3: Analyze generated artifacts

**Files:**
- Verify: `notebook/tables_universe_analysis/*.csv`
- Verify: `notebook/figures_universe_analysis/*.png`

- [ ] Check artifact counts, CSV readability, required columns, empty tables, duplicate rows, and non-finite metrics.
- [ ] Summarize strategy coverage and comparative return, Sharpe, CVaR, drawdown, and governance results.
- [ ] Run a fresh final compilation, focused tests, and output-QA command.
