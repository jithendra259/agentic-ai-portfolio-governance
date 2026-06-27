import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.live_data_tools import (
    _annual_to_daily_return,
    _build_optimization_payload,
    _lightweight_optimization_payload,
    _portfolio_audit_metrics,
)
from src.orchestrator.chatbot_orchestrator import _build_governance_markdown


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

    def effective_dates(self):
        return {ticker: "2024-09-06" for ticker in self.prices().columns}

    def test_successful_solution_reports_and_respects_cap(self):
        result = _build_optimization_payload(
            self.prices(),
            self.effective_dates(),
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
                self.effective_dates(),
                "2024-09-06",
            )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["fallback_applied"])
        self.assertFalse(result["target_return_constraint_used"])
        self.assertLessEqual(result["max_observed_weight"], result["max_weight_constraint"] + 1e-6)

    def test_solver_failure_returns_error_instead_of_uncapped_weights(self):
        with patch("cvxpy.Problem.solve", side_effect=RuntimeError("solver unavailable")):
            result = _build_optimization_payload(
                self.prices(),
                self.effective_dates(),
                "2024-09-06",
            )

        self.assertEqual(result["status"], "error")
        self.assertIn("capped solution", result["message"])


class LightweightGovernanceContractTests(unittest.TestCase):
    def test_scalar_audit_fields_are_preserved_without_large_matrices(self):
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


class GovernanceMarkdownAuditTests(unittest.TestCase):
    def test_markdown_reports_constraints_solver_and_concentration(self):
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


if __name__ == "__main__":
    unittest.main()
