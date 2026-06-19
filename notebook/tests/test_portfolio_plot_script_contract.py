import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "agentic_ai_portfolio_governance_final_repaired_full_pro (1).py"
)


class PortfolioPlotScriptContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT_PATH.read_text(encoding="utf-8")

    def test_quarterly_resampling_uses_pandas_3_alias(self):
        self.assertNotIn('rebalance_frequency="Q"', self.source)
        self.assertIn('rebalance_frequency="QE"', self.source)

    def test_monthly_resampling_uses_pandas_3_alias(self):
        self.assertNotIn('.resample("M")', self.source)
        self.assertIn('.resample("ME")', self.source)

    def test_hrp_distance_matrix_uses_writable_numpy_copy(self):
        self.assertNotIn("np.fill_diagonal(dist.values, 0)", self.source)
        self.assertIn("dist_values = dist.to_numpy(copy=True)", self.source)

    def test_full_script_uses_complete_2014_2025_download_window(self):
        self.assertIn('"start_date": "2014-01-01"', self.source)
        self.assertIn('"end_date": "2026-01-01"', self.source)

    def test_full_script_exports_protocol_audits(self):
        for filename in [
            "calibration_vs_test_boundary_audit.csv",
            "gcvar_calibration_results.csv",
            "gcvar_solver_and_slack_audit.csv",
            "gcvar_behavioral_validation.csv",
            "gcvar_test_governance_ranking.csv",
        ]:
            self.assertIn(filename, self.source)

    def test_full_script_imports_protocol_module(self):
        self.assertIn("from gcvar_protocol import", self.source)
        self.assertNotIn('"train_test_split": "2015-01-01"', self.source)

    def test_narrative_names_protocol_and_actual_test_window(self):
        self.assertIn(
            "Walk-Forward Governance G-CVaR Evaluation Protocol", self.source
        )
        self.assertIn(
            "Untouched test period: 2023-01-01 through 2025-12-31",
            self.source,
        )
        self.assertNotIn("main out-of-sample split starts in 2020", self.source)

    def test_narrative_preserves_non_fabrication_rule(self):
        self.assertIn("reports the actual rank", self.source)
        self.assertIn(
            "does not guarantee the highest terminal wealth", self.source
        )


if __name__ == "__main__":
    unittest.main()
