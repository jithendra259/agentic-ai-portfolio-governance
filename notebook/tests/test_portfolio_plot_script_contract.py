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

    def test_yfinance_download_avoids_threaded_rate_limit_failures(self):
        download_start = self.source.index("def download_adjusted_close")
        download_end = self.source.index("raw_prices =", download_start)
        download_block = self.source[download_start:download_end]
        self.assertIn("threads=False", download_block)
        self.assertNotIn("threads=True", download_block)

    def test_yfinance_download_uses_incremental_persistent_cache(self):
        download_start = self.source.index("def download_adjusted_close")
        download_end = self.source.index("raw_prices =", download_start)
        download_block = self.source[download_start:download_end]
        self.assertIn("update_adjusted_close_cache", self.source)
        self.assertIn("yfinance_adjusted_close_2014_2025.csv", download_block)
        self.assertIn("MARKET_DATA_DOWNLOAD_AUDIT", download_block)

    def test_full_script_exports_protocol_audits(self):
        for filename in [
            "calibration_vs_test_boundary_audit.csv",
            "gcvar_calibration_results.csv",
            "gcvar_solver_and_slack_audit.csv",
            "gcvar_behavioral_validation.csv",
            "gcvar_test_governance_ranking.csv",
        ]:
            self.assertIn(filename, self.source)

    def test_full_script_exports_separate_validation_and_test_behavior(self):
        for filename in [
            "gcvar_validation_behavioral_validation.csv",
            "gcvar_test_behavioral_validation.csv",
            "gcvar_adaptive_gate_audit_validation.csv",
            "gcvar_adaptive_gate_audit_test.csv",
            "instability_vs_adaptive_gate_validation.png",
            "instability_vs_adaptive_gate_test.png",
            "crisis_governance_comparison_validation.png",
            "crisis_governance_comparison_test.png",
        ]:
            self.assertIn(filename, self.source)

    def test_full_script_exports_plot_quality_audit(self):
        self.assertIn("plot_quality_audit.csv", self.source)
        self.assertIn("Image.open", self.source)

    def test_final_protocol_overwrites_legacy_adaptive_evidence(self):
        marker = self.source.index(
            "FINAL WALK-FORWARD GOVERNANCE G-CVaR EVALUATION PROTOCOL"
        )
        for filename in [
            "adaptive_gcvar_evidence_triangle.png",
            "adaptive_lambda_diagnostics_grid.png",
            "gcvar_implementation_audit.png",
            "instability_vs_adaptive_lambda.png",
            "crisis_only_governance_comparison.png",
        ]:
            self.assertGreater(self.source.rfind(filename), marker)

    def test_boxplot_uses_current_matplotlib_keyword(self):
        self.assertNotIn("boxplot(data, labels=", self.source)
        self.assertIn("boxplot(data, tick_labels=", self.source)

    def test_crisis_plot_and_zero_fallback_panel_are_explicit(self):
        crisis_start = self.source.index(
            "def plot_crisis_governance_comparison"
        )
        crisis_end = self.source.index(
            "plot_instability_vs_adaptive_gate(", crisis_start
        )
        crisis_block = self.source[crisis_start:crisis_end]
        self.assertIn("constrained_layout=True", crisis_block)
        self.assertIn("strategy_labels", crisis_block)
        self.assertIn("No solver fallbacks", self.source)

    def test_v2_is_supplemental_and_runs_after_primary_protocol(self):
        marker = self.source.index(
            "FINAL WALK-FORWARD GOVERNANCE G-CVaR EVALUATION PROTOCOL"
        )
        label = "Supplemental Linear-Centrality Adaptive G-CVaR"
        v2 = self.source.index(label)

        self.assertGreater(v2, marker)
        self.assertIn('"adaptive_graph_cvar_v2"', self.source[v2:])
        self.assertIn("from gcvar_v2 import", self.source[v2:])

    def test_v2_exports_thesis_safe_audits_and_rankings(self):
        for filename in [
            "adaptive_graph_cvar_v2_audit.csv",
            "adaptive_graph_cvar_v2_results.csv",
            "adaptive_graph_cvar_v2_weights.csv",
            "adaptive_graph_cvar_v2_activation_summary.csv",
            "nan_safe_core_governance_ranking.csv",
            "nan_safe_supplemental_governance_ranking.csv",
            "nan_safe_hitl_simulation_ranking.csv",
            "nan_safe_governance_rejections.csv",
            "final_technical_validation_checks.csv",
        ]:
            self.assertIn(filename, self.source)

    def test_v2_does_not_replace_primary_strategy_name(self):
        label = "Supplemental Linear-Centrality Adaptive G-CVaR"
        v2_block = self.source[self.source.index(label):]
        self.assertIn('"strategy": "adaptive_graph_cvar_v2"', v2_block)
        self.assertIn('"strategy_label": "Supplemental Linear-Centrality Adaptive G-CVaR"', v2_block)

    def test_v2_finite_metric_check_reduces_to_one_boolean(self):
        label = "Supplemental Linear-Centrality Adaptive G-CVaR"
        v2_block = self.source[self.source.index(label):]
        self.assertIn(").all().all()", v2_block)

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
