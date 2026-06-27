# Portfolio Script Execution Design

## Goal

Make `notebook/agentic_ai_portfolio_governance_final_repaired_full_pro.py` complete a reproducible run, retain the thesis-defined 2015 out-of-sample start, and produce auditable tables and figures.

## Design

Keep the monolithic exported-notebook structure intact. Fix only failures demonstrated by execution, beginning with the mismatch between the approximately 251 observations available before the 2015 split and the hard-coded 252-observation gate. Express the minimum through `CONFIG`, validate that the central backtest produced rows before downstream analysis, and preserve every existing strategy and output contract.

Run with the project virtual environment and a non-interactive Matplotlib backend. After each fix, rerun from the beginning and inspect generated CSV schemas, row counts, missing values, strategy coverage, and key comparative metrics. Later runtime changes are permitted only when a measured stage is prohibitively slow.

## Error handling and verification

Missing training data must report the observed and required counts. An empty central backtest must raise a descriptive runtime error instead of failing later with a pandas `KeyError`. Verification consists of focused source tests, Python compilation, a full script run, and programmatic output QA.
