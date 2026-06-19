import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NOTEBOOK_DIR))


class LinearGraphAndSignalTests(unittest.TestCase):
    def setUp(self):
        index = pd.bdate_range("2019-01-01", periods=420)
        rng = np.random.default_rng(42)
        common = rng.normal(0, 0.008, len(index))
        self.returns = pd.DataFrame(
            {
                name: common + rng.normal(0, 0.006, len(index))
                for name in list("ABCD")
            },
            index=index,
        )
        self.holdings = pd.DataFrame(
            [
                {
                    "filing_date": "2023-04-15",
                    "manager_id": "public_manager",
                    "ticker": "A",
                    "market_value": 80.0,
                },
                {
                    "filing_date": "2023-04-15",
                    "manager_id": "public_manager",
                    "ticker": "B",
                    "market_value": 20.0,
                },
                {
                    "filing_date": "2023-06-15",
                    "manager_id": "future_manager",
                    "ticker": "A",
                    "market_value": 100.0,
                },
            ]
        )
        self.holdings["filing_date"] = pd.to_datetime(
            self.holdings["filing_date"]
        )

    def test_missing_holdings_uses_correlation_proxy(self):
        from gcvar_v2 import get_linear_graph_penalty

        penalty, source = get_linear_graph_penalty(
            self.returns,
            self.returns.columns,
            pd.Timestamp("2023-03-31"),
            holdings=pd.DataFrame(),
            threshold=0.0,
        )

        self.assertEqual(source, "correlation_proxy")
        self.assertEqual(list(penalty.index), list(self.returns.columns))
        self.assertTrue(penalty.between(0, 1).all())

    def test_holdings_filter_is_publication_date_safe(self):
        from gcvar_v2 import build_holdings_matrix

        matrix = build_holdings_matrix(
            self.holdings,
            ["A", "B"],
            pd.Timestamp("2023-05-01"),
        )

        self.assertIn("public_manager", matrix.index)
        self.assertNotIn("future_manager", matrix.index)
        self.assertAlmostEqual(matrix.loc["public_manager"].sum(), 1.0)

    def test_loader_derives_publication_date_from_report_date(self):
        from gcvar_v2 import load_clean_13f_holdings

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "holdings.csv"
            pd.DataFrame(
                [
                    {
                        "report_date": "2023-03-31",
                        "manager_id": "m1",
                        "ticker": "a",
                        "market_value": 10,
                    }
                ]
            ).to_csv(path, index=False)
            loaded = load_clean_13f_holdings(path, publication_lag_days=45)

        self.assertEqual(loaded.loc[0, "ticker"], "A")
        self.assertEqual(
            loaded.loc[0, "filing_date"], pd.Timestamp("2023-05-15")
        )

    def test_activation_uses_multiplier_not_effective_lambda(self):
        from gcvar_v2 import adaptive_gate_signal

        signal = adaptive_gate_signal(
            instability=2.0,
            theta=0.0,
            steepness=2.0,
            graph_lambda=0.0025,
        )

        self.assertEqual(signal.active, signal.multiplier > 0.5)
        self.assertTrue(signal.active)
        self.assertLess(signal.effective_lambda, 0.5)

    def test_gate_calibration_uses_only_validation_dates(self):
        from gcvar_v2 import calibrate_gate

        index = pd.bdate_range("2019-01-01", "2023-12-31")
        instability = pd.Series(np.linspace(-1, 1, len(index)), index=index)
        changed_test = instability.copy()
        changed_test.loc["2023-01-01":] = 1_000_000

        first = calibrate_gate(
            instability,
            "2020-01-01",
            "2022-12-31",
            target_frequency=0.10,
        )
        second = calibrate_gate(
            changed_test,
            "2020-01-01",
            "2022-12-31",
            target_frequency=0.10,
        )

        self.assertEqual(first, second)


class LinearOptimizerAndWalkForwardTests(unittest.TestCase):
    def setUp(self):
        index = pd.bdate_range("2018-01-01", "2023-06-30")
        rng = np.random.default_rng(77)
        common = rng.normal(0.0003, 0.006, len(index))
        self.returns = pd.DataFrame(
            {
                "A": common + rng.normal(0, 0.005, len(index)),
                "B": common * 0.7 + rng.normal(0, 0.007, len(index)),
                "C": -common * 0.2 + rng.normal(0, 0.006, len(index)),
                "D": rng.normal(0.0002, 0.008, len(index)),
            },
            index=index,
        )

    def test_linear_optimizer_returns_feasible_weights(self):
        from gcvar_v2 import LinearGCVarParams, optimize_linear_centrality_cvar

        weights, audit = optimize_linear_centrality_cvar(
            self.returns.iloc[:260],
            pd.Series([0.1, 0.4, 0.7, 1.0], index=self.returns.columns),
            effective_lambda=0.0025,
            params=LinearGCVarParams(max_weight=0.4),
        )

        self.assertAlmostEqual(float(weights.sum()), 1.0, places=7)
        self.assertTrue((weights >= -1e-8).all())
        self.assertLessEqual(float(weights.max()), 0.4 + 1e-6)
        self.assertEqual(audit["graph_objective_type"], "linear_centrality")
        self.assertFalse(audit["fallback"])

    def test_walk_forward_never_uses_current_or_future_returns(self):
        from gcvar_v2 import LinearGCVarParams, run_linear_gcvar_walk_forward

        result = run_linear_gcvar_walk_forward(
            self.returns,
            universe="U_TEST",
            holdings=pd.DataFrame(),
            validation_start=pd.Timestamp("2020-01-01"),
            validation_end=pd.Timestamp("2022-12-31"),
            test_start=pd.Timestamp("2023-01-01"),
            test_end=pd.Timestamp("2023-06-30"),
            params=LinearGCVarParams(
                lookback_days=260,
                rebalance_frequency="ME",
                max_weight=0.5,
            ),
        )

        self.assertFalse(result.returns.empty)
        self.assertFalse(result.audit.empty)
        self.assertTrue(
            (result.audit["training_end"] < result.audit["decision_date"]).all()
        )
        self.assertTrue(result.audit["turnover"].notna().all())
        self.assertEqual(set(result.audit["graph_source"]), {"correlation_proxy"})
        self.assertTrue(
            (
                result.audit["active"]
                == (result.audit["lambda_multiplier"] > 0.5)
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
