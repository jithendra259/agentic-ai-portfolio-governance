import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np


NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NOTEBOOK_DIR))


class ProtocolDateTests(unittest.TestCase):
    def test_protocol_dates_are_ordered_and_cover_2014_through_2025(self):
        from gcvar_protocol import ProtocolDates

        dates = ProtocolDates.default()

        self.assertEqual(dates.download_start, pd.Timestamp("2014-01-01"))
        self.assertEqual(dates.download_end_exclusive, pd.Timestamp("2026-01-01"))
        self.assertEqual(dates.training_end, pd.Timestamp("2019-12-31"))
        self.assertEqual(dates.validation_end, pd.Timestamp("2022-12-31"))
        self.assertEqual(dates.test_end, pd.Timestamp("2025-12-31"))

    def test_boundary_audit_proves_test_is_not_used_for_calibration(self):
        from gcvar_protocol import ProtocolDates, build_boundary_audit

        audit = build_boundary_audit(
            universe="U1",
            raw_index=pd.bdate_range("2014-01-02", "2025-12-31"),
            selected_params={"graph_lambda": 0.1},
            parameter_grid_hash="abc123",
            dates=ProtocolDates.default(),
        )

        self.assertFalse(audit["whether_test_used_in_calibration"])
        self.assertEqual(audit["raw_data_end"], "2025-12-31")


class TailGraphTests(unittest.TestCase):
    def test_tail_graph_is_aligned_symmetric_finite_and_psd(self):
        from gcvar_protocol import make_tail_graph_psd_matrix

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


class GovernanceOptimizerTests(unittest.TestCase):
    def setUp(self):
        from gcvar_protocol import make_tail_graph_psd_matrix

        rng = np.random.default_rng(7)
        self.returns = pd.DataFrame(
            rng.normal(
                loc=np.array([0.0008, 0.0005, 0.0003, 0.0004]),
                scale=np.array([0.020, 0.012, 0.009, 0.011]),
                size=(500, 4),
            ),
            columns=["A", "B", "C", "D"],
        )
        self.graph, _ = make_tail_graph_psd_matrix(self.returns, threshold=0.0)

    def test_optimizer_returns_feasible_weights_and_nonnegative_slack(self):
        from gcvar_protocol import GovernanceParams, optimize_governance_gcvar

        params = GovernanceParams()
        weights, audit = optimize_governance_gcvar(
            self.returns,
            graph_matrix=self.graph,
            lambda_t=0.10,
            previous_weights=None,
            params=params,
        )

        self.assertAlmostEqual(weights.sum(), 1.0, places=7)
        self.assertGreaterEqual(weights.min(), -1e-9)
        self.assertLessEqual(weights.max(), params.max_weight + 1e-6)
        self.assertGreaterEqual(audit["cvar_slack"], 0.0)
        self.assertGreaterEqual(audit["graph_slack"], 0.0)
        self.assertGreaterEqual(audit["return_slack"], 0.0)

    def test_turnover_penalty_keeps_weights_closer_to_previous_allocation(self):
        from gcvar_protocol import GovernanceParams, optimize_governance_gcvar

        previous = pd.Series([0.30, 0.30, 0.25, 0.15], index=self.returns.columns)
        low, _ = optimize_governance_gcvar(
            self.returns,
            self.graph,
            0.10,
            previous,
            GovernanceParams(turnover_lambda=0.0),
        )
        high, _ = optimize_governance_gcvar(
            self.returns,
            self.graph,
            0.10,
            previous,
            GovernanceParams(turnover_lambda=1.0),
        )

        self.assertLessEqual(
            (high - previous).abs().sum(),
            (low - previous).abs().sum() + 1e-6,
        )


class AdaptiveProtocolTests(unittest.TestCase):
    def test_adaptive_lambda_is_zero_in_calm_and_bounded_in_crisis(self):
        from gcvar_protocol import adaptive_lambda_quantile

        history = pd.Series(np.linspace(-1.0, 1.0, 200))

        calm = adaptive_lambda_quantile(history, -0.5, 0.8, 0.8, 8.0)
        crisis = adaptive_lambda_quantile(history, 2.0, 0.8, 0.8, 8.0)

        self.assertEqual(calm, 0.0)
        self.assertGreater(crisis, 0.0)
        self.assertLessEqual(crisis, 0.8)

    def test_walk_forward_training_cutoff_is_strictly_before_decision_date(self):
        from gcvar_protocol import GovernanceParams, run_walk_forward_gcvar

        index = pd.bdate_range("2019-01-01", "2023-03-31")
        rng = np.random.default_rng(99)
        returns = pd.DataFrame(
            rng.normal(0, 0.01, (len(index), 4)),
            index=index,
            columns=list("ABCD"),
        )
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

        self.assertFalse(result.decision_log.empty)
        self.assertTrue(
            (
                result.decision_log["training_end"]
                < result.decision_log["decision_date"]
            ).all()
        )


class GovernanceScoringTests(unittest.TestCase):
    def test_score_weights_are_fixed_and_sum_to_one(self):
        from gcvar_protocol import GOVERNANCE_SCORE_WEIGHTS

        self.assertEqual(
            GOVERNANCE_SCORE_WEIGHTS,
            {
                "sharpe_ratio": 0.20,
                "annual_return": 0.15,
                "historical_cvar_loss_95": 0.25,
                "max_drawdown_magnitude": 0.15,
                "graph_exposure": 0.10,
                "effective_n": 0.10,
                "turnover": 0.05,
            },
        )
        self.assertAlmostEqual(sum(GOVERNANCE_SCORE_WEIGHTS.values()), 1.0)

    def test_incomplete_rows_do_not_receive_silently_reweighted_scores(self):
        from gcvar_protocol import compute_governance_scores

        metrics = pd.DataFrame(
            [
                {
                    "universe": "U1",
                    "strategy": "complete",
                    "sharpe_ratio": 1.0,
                    "annual_return": 0.1,
                    "historical_cvar_loss_95": 0.02,
                    "max_drawdown_magnitude": 0.2,
                    "graph_exposure": 0.2,
                    "effective_n": 5.0,
                    "turnover": 0.1,
                },
                {
                    "universe": "U1",
                    "strategy": "missing",
                    "sharpe_ratio": 2.0,
                },
            ]
        )

        scored = compute_governance_scores(metrics)
        missing = scored.set_index("strategy").loc["missing"]

        self.assertEqual(missing["score_status"], "incomplete")
        self.assertTrue(pd.isna(missing["composite_governance_score"]))

    def test_calibration_rejects_overlapping_validation_and_test_dates(self):
        from gcvar_protocol import assert_disjoint_windows

        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint_windows(
                pd.bdate_range("2020-01-01", "2023-01-10"),
                pd.bdate_range("2023-01-01", "2025-12-31"),
            )


if __name__ == "__main__":
    unittest.main()
