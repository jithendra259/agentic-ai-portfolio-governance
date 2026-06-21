# Chatbot Portfolio Optimization Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the production chatbot's historical Graph-CVaR allocation auditable, cap-preserving, geometrically consistent, and more informative without changing notebook research code.

**Architecture:** Keep the active optimizer in `backend/src/agents/live_data_tools.py`, but extract small pure helpers for return conversion and portfolio diagnostics. Preserve the existing linear graph-centrality objective, remove the uncapped fallback, enrich the lightweight governance contract with scalar audit values, and render those values deterministically in the governance markdown response.

**Tech Stack:** Python 3, pandas, NumPy, CVXPY, LangChain tools, `unittest`.

---

## File Structure

- Modify `backend/src/agents/live_data_tools.py`: pure audit helpers, capped fallback behavior, optional previous weights, scalar optimization audit.
- Modify `backend/src/orchestrator/chatbot_orchestrator.py`: optional prior-weight forwarding and deterministic optimization-audit markdown.
- Create `backend/test/test_chatbot_optimizer_audit.py`: optimizer math, fallback, payload, and response regression tests.
- Preserve all notebook, report, frontend, and unrelated dirty-worktree files.
- Do not commit implementation files during this execution because both production targets already contain unrelated user changes. Keep the patch unstaged for review unless the user later requests a focused publish workflow.

### Task 1: Add pure portfolio-audit mathematics

**Files:**
- Modify: `backend/src/agents/live_data_tools.py:465`
- Create: `backend/test/test_chatbot_optimizer_audit.py`

- [ ] **Step 1: Write failing tests for geometric conversion and audit metrics**

```python
import math
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.live_data_tools import _annual_to_daily_return, _portfolio_audit_metrics


class PortfolioAuditMathTests(unittest.TestCase):
    def test_annual_return_is_converted_geometrically(self):
        daily = _annual_to_daily_return(0.15)
        self.assertAlmostEqual((1.0 + daily) ** 252 - 1.0, 0.15, places=10)

    def test_invalid_annual_return_below_minus_one_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "greater than -1"):
            _annual_to_daily_return(-1.0)

    def test_audit_metrics_include_concentration_graph_exposure_and_turnover(self):
        metrics = _portfolio_audit_metrics(
            weights={"AAPL": 0.6, "MSFT": 0.4},
            graph_scores={"AAPL": 0.8, "MSFT": 0.2},
            previous_weights={"AAPL": 0.5, "MSFT": 0.5},
            max_weight_constraint=0.65,
        )
        self.assertAlmostEqual(metrics["hhi"], 0.52)
        self.assertAlmostEqual(metrics["effective_number_of_holdings"], 1.0 / 0.52)
        self.assertAlmostEqual(metrics["graph_exposure"], 0.56)
        self.assertAlmostEqual(metrics["turnover"], 0.1)
        self.assertAlmostEqual(metrics["max_observed_weight"], 0.6)
        self.assertAlmostEqual(metrics["weight_cap_utilization"], 0.6 / 0.65)

    def test_turnover_is_none_without_previous_weights(self):
        metrics = _portfolio_audit_metrics(
            weights={"AAPL": 0.6, "MSFT": 0.4},
            graph_scores={},
            previous_weights=None,
            max_weight_constraint=0.65,
        )
        self.assertIsNone(metrics["turnover"])
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m unittest backend.test.test_chatbot_optimizer_audit.PortfolioAuditMathTests -v
```

Expected: import failure because `_annual_to_daily_return` and `_portfolio_audit_metrics` do not exist.

- [ ] **Step 3: Implement the pure helpers**

Add above `_build_optimization_payload`:

```python
def _annual_to_daily_return(annual_return: float, periods: int = 252) -> float:
    value = float(annual_return)
    if value <= -1.0:
        raise ValueError("annual return target must be greater than -1")
    return float((1.0 + value) ** (1.0 / periods) - 1.0)


def _portfolio_audit_metrics(
    weights: dict[str, float],
    graph_scores: Optional[dict[str, float]],
    previous_weights: Optional[dict[str, float]],
    max_weight_constraint: float,
) -> dict:
    current = pd.Series(weights, dtype=float).clip(lower=0.0)
    current = current / current.sum()
    hhi = float(np.square(current).sum())
    graph = pd.Series(graph_scores or {}, dtype=float).reindex(current.index).fillna(0.0)
    turnover = None
    if previous_weights:
        previous = pd.Series(previous_weights, dtype=float).clip(lower=0.0)
        previous = previous.reindex(current.index).fillna(0.0)
        if float(previous.sum()) > 0:
            previous = previous / previous.sum()
            turnover = float(0.5 * (current - previous).abs().sum())
    maximum = float(current.max())
    return {
        "hhi": hhi,
        "effective_number_of_holdings": float(1.0 / hhi) if hhi > 0 else None,
        "graph_exposure": float(current @ graph),
        "turnover": turnover,
        "max_observed_weight": maximum,
        "weight_cap_utilization": float(maximum / max_weight_constraint),
    }
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run the command from Step 2. Expected: four tests pass.

- [ ] **Step 5: Inspect the scoped diff without staging**

```powershell
git diff -- backend/src/agents/live_data_tools.py backend/test/test_chatbot_optimizer_audit.py
```

### Task 2: Preserve the weight cap and expose solver fallback state

**Files:**
- Modify: `backend/src/agents/live_data_tools.py:465-669`
- Test: `backend/test/test_chatbot_optimizer_audit.py`

- [ ] **Step 1: Add failing optimizer contract tests**

Append:

```python
from unittest.mock import patch
import numpy as np
import pandas as pd

from src.agents.live_data_tools import _build_optimization_payload


class OptimizerAuditContractTests(unittest.TestCase):
    @staticmethod
    def prices():
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-01", periods=180)
        returns = rng.normal(
            loc=[0.0002, 0.0003, 0.0004, 0.0001],
            scale=[0.008, 0.009, 0.010, 0.007],
            size=(len(dates), 4),
        )
        values = 100.0 * np.exp(np.cumsum(returns, axis=0))
        return pd.DataFrame(values, index=dates, columns=["A", "B", "C", "D"])

    def test_successful_solution_reports_and_respects_cap(self):
        result = _build_optimization_payload(
            self.prices(),
            {ticker: "2024-09-06" for ticker in ["A", "B", "C", "D"]},
            "2024-09-06",
            previous_weights={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
        )
        self.assertEqual(result["status"], "success")
        self.assertLessEqual(result["max_observed_weight"], result["max_weight_constraint"] + 1e-6)
        self.assertIn(result["solver_status"], {"optimal", "optimal_inaccurate"})
        self.assertIsNotNone(result["turnover"])

    def test_return_floor_fallback_keeps_cap(self):
        with patch("src.agents.live_data_tools.np.percentile", return_value=10.0):
            result = _build_optimization_payload(
                self.prices(),
                {ticker: "2024-09-06" for ticker in ["A", "B", "C", "D"]},
                "2024-09-06",
            )
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["fallback_applied"])
        self.assertFalse(result["target_return_constraint_used"])
        self.assertLessEqual(result["max_observed_weight"], result["max_weight_constraint"] + 1e-6)

    def test_solver_failure_returns_error_instead_of_uncapped_weights(self):
        with patch("src.agents.live_data_tools.cp.Problem.solve", side_effect=RuntimeError("solver unavailable")):
            result = _build_optimization_payload(
                self.prices(),
                {ticker: "2024-09-06" for ticker in ["A", "B", "C", "D"]},
                "2024-09-06",
            )
        self.assertEqual(result["status"], "error")
        self.assertIn("capped solution", result["message"])
```

- [ ] **Step 2: Run the optimizer tests and verify RED**

```powershell
python -m unittest backend.test.test_chatbot_optimizer_audit.OptimizerAuditContractTests -v
```

Expected: failures for the missing `previous_weights` argument and audit keys.

- [ ] **Step 3: Refactor the capped solve sequence**

Change the signature to accept:

```python
previous_weights: Optional[dict[str, float]] = None,
```

Use `_annual_to_daily_return(target_annual_return)`. Track `solver_name`, `solver_status`, `fallback_applied`, `fallback_reason`, and `target_return_constraint_used`. First solve with the profile return floor. If infeasible, solve again with the same cap and no return-floor constraint. Delete the fallback that removes `weights <= max_weight_limit`.

On total solver failure return:

```python
return {
    "status": "error",
    "message": (
        f"Historical CVaR optimization could not find a capped solution for {target_date}. "
        f"Solver status: {solver_status or problem.status}."
    ),
    "max_weight_constraint": round(max_weight_limit, 6),
    "fallback_applied": True,
    "fallback_reason": "profile return floor was infeasible and no capped fallback solution was available",
}
```

After cleaning weights, verify the normalized result still satisfies the cap within `1e-6`; otherwise return an error. Merge `_portfolio_audit_metrics(...)` into the success payload and add:

```python
"target_daily_return_floor": round(target_daily_return, 10),
"target_return_constraint_used": target_return_constraint_used,
"fallback_applied": fallback_applied,
"fallback_reason": fallback_reason,
"solver_name": solver_name,
"solver_status": solver_status,
"max_weight_constraint": round(max_weight_limit, 6),
"effective_window_start": overlapping_prices.index.min().strftime("%Y-%m-%d"),
```

- [ ] **Step 4: Run the optimizer tests and verify GREEN**

Run the command from Step 2. Expected: three tests pass.

- [ ] **Step 5: Run existing optimizer-agent tests**

```powershell
python -m unittest backend.test.test_optimizer_agent -v
```

Expected: existing optimizer-agent tests pass unchanged.

- [ ] **Step 6: Add auditable optimization logging**

After the solve sequence, log only scalar diagnostics:

```python
logger.info(
    "Governance optimizer target_date=%s solver=%s status=%s cap=%.6f fallback=%s reason=%s",
    target_date,
    solver_name,
    solver_status,
    max_weight_limit,
    fallback_applied,
    fallback_reason or "none",
)
```

Do not log matrices, price histories, or previous-weight payloads.

- [ ] **Step 7: Inspect the scoped diff without staging**

```powershell
git diff -- backend/src/agents/live_data_tools.py backend/test/test_chatbot_optimizer_audit.py
```

### Task 3: Preserve scalar audit values in the governance tool contract

**Files:**
- Modify: `backend/src/agents/live_data_tools.py:2228-2423`
- Modify: `backend/src/orchestrator/chatbot_orchestrator.py:355-388`
- Test: `backend/test/test_chatbot_optimizer_audit.py`

- [ ] **Step 1: Add a failing lightweight-contract test**

```python
class LightweightGovernanceContractTests(unittest.TestCase):
    def test_scalar_audit_fields_are_preserved_without_large_matrices(self):
        from src.agents.live_data_tools import _lightweight_optimization_payload

        full = {
            "weights": {"A": 0.5, "B": 0.5},
            "solver_name": "CLARABEL",
            "solver_status": "optimal",
            "max_weight_constraint": 0.6,
            "hhi": 0.5,
            "effective_number_of_holdings": 2.0,
            "turnover": None,
            "fallback_applied": False,
            "correlation_matrix": {"A": {"A": 1.0}},
            "covariance_matrix": {"A": {"A": 0.1}},
        }
        light = _lightweight_optimization_payload(full)
        self.assertEqual(light["solver_name"], "CLARABEL")
        self.assertIn("turnover", light)
        self.assertNotIn("correlation_matrix", light)
        self.assertNotIn("covariance_matrix", light)
```

- [ ] **Step 2: Run the contract test and verify RED**

```powershell
python -m unittest backend.test.test_chatbot_optimizer_audit.LightweightGovernanceContractTests -v
```

Expected: import failure because `_lightweight_optimization_payload` does not exist.

- [ ] **Step 3: Implement the scalar allowlist and wire optional previous weights**

Add `_lightweight_optimization_payload(payload)` using an explicit scalar-field allowlist from the approved spec plus `weights`, `graph_scores_used`, and `historical_pricing_dates`. Replace the hand-built `lightweight_optimization` dictionary in `run_full_governance_pipeline` with this helper.

Add optional `previous_weights` to both the LangChain wrapper and `run_full_governance_pipeline`, then forward it into `_build_optimization_payload`. Existing callers remain compatible because the default is `None`.

- [ ] **Step 4: Run the contract test and verify GREEN**

Run the command from Step 2. Expected: one test passes.

- [ ] **Step 5: Inspect the scoped contract diff without staging**

```powershell
git diff -- backend/src/agents/live_data_tools.py backend/src/orchestrator/chatbot_orchestrator.py backend/test/test_chatbot_optimizer_audit.py
```

### Task 4: Render the optimization audit in chatbot responses

**Files:**
- Modify: `backend/src/orchestrator/chatbot_orchestrator.py:1485-1542`
- Test: `backend/test/test_chatbot_optimizer_audit.py`

- [ ] **Step 1: Add failing response tests**

```python
class GovernanceMarkdownAuditTests(unittest.TestCase):
    def test_markdown_reports_constraints_solver_and_concentration(self):
        from src.orchestrator.chatbot_orchestrator import _build_governance_markdown

        payload = {
            "status": "success",
            "target_date": "2026-06-20",
            "valid_tickers": ["A", "B"],
            "optimization": {
                "weights": {"A": 0.6, "B": 0.4},
                "risk_tolerance": "moderate",
                "solver_name": "CLARABEL",
                "solver_status": "optimal",
                "max_weight_constraint": 0.65,
                "max_observed_weight": 0.6,
                "hhi": 0.52,
                "effective_number_of_holdings": 1.923077,
                "graph_exposure": 0.56,
                "turnover": None,
                "fallback_applied": False,
                "effective_window_start": "2024-01-01",
                "effective_window_end": "2026-06-20",
            },
        }
        text = _build_governance_markdown(payload, "")
        self.assertIn("Solver: CLARABEL (optimal)", text)
        self.assertIn("Maximum-weight constraint: 65.00%", text)
        self.assertIn("HHI concentration: 0.5200", text)
        self.assertIn("Effective holdings: 1.92", text)
        self.assertIn("Turnover: unavailable", text)

    def test_markdown_warns_when_return_floor_is_relaxed(self):
        from src.orchestrator.chatbot_orchestrator import _build_governance_markdown

        payload = {
            "status": "success",
            "target_date": "2026-06-20",
            "valid_tickers": ["A", "B"],
            "optimization": {
                "weights": {"A": 0.5, "B": 0.5},
                "fallback_applied": True,
                "fallback_reason": "profile return floor was infeasible",
                "target_return_constraint_used": False,
            },
        }
        text = _build_governance_markdown(payload, "")
        self.assertIn("Optimization warning", text)
        self.assertIn("profile return floor was infeasible", text)
```

- [ ] **Step 2: Run the markdown tests and verify RED**

```powershell
python -m unittest backend.test.test_chatbot_optimizer_audit.GovernanceMarkdownAuditTests -v
```

Expected: assertions fail because the audit section is not rendered.

- [ ] **Step 3: Add compact deterministic audit rendering**

Extend `_build_governance_markdown` with guarded scalar formatting. Never format missing turnover as zero. Add a warning whenever `fallback_applied` is true, and include `fallback_reason` verbatim after normal string conversion.

- [ ] **Step 4: Run the markdown tests and verify GREEN**

Run the command from Step 2. Expected: two tests pass.

- [ ] **Step 5: Inspect the scoped response diff without staging**

```powershell
git diff -- backend/src/orchestrator/chatbot_orchestrator.py backend/test/test_chatbot_optimizer_audit.py
```

### Task 5: Complete regression verification

**Files:**
- Verify only; no planned production edits.

- [ ] **Step 1: Run the new complete test module**

```powershell
python -m unittest backend.test.test_chatbot_optimizer_audit -v
```

Expected: all new tests pass.

- [ ] **Step 2: Run affected chatbot and governance suites**

```powershell
python -m unittest backend.test.test_optimizer_agent backend.test.test_governance_plot_continuity backend.test.test_chat_sessions_api backend.test.test_session_memory_context -v
```

Expected: all affected tests pass.

- [ ] **Step 3: Compile modified Python files**

```powershell
python -m py_compile backend/src/agents/live_data_tools.py backend/src/orchestrator/chatbot_orchestrator.py backend/test/test_chatbot_optimizer_audit.py
```

Expected: exit code 0 with no output.

- [ ] **Step 4: Check patch hygiene**

```powershell
git diff --check -- backend/src/agents/live_data_tools.py backend/src/orchestrator/chatbot_orchestrator.py backend/test/test_chatbot_optimizer_audit.py
git status --short
```

Expected: no whitespace errors; unrelated dirty files remain present but unstaged and unchanged by this implementation.

- [ ] **Step 5: Review the final diff against the approved spec**

Confirm explicitly that the maximum-weight constraint is present in both solve attempts, no uncapped fallback remains, turnover is unavailable without prior weights, large matrices are excluded from the lightweight payload, and notebook/report/frontend files were not modified by this work.
