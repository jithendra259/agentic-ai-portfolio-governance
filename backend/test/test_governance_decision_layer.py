import unittest
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.decision.decision_orchestrator import DecisionOrchestrator
from src.decision.governance_policy import GovernancePolicy
from src.decision.governance_decision_agent import GovernanceDecisionAgent


class GovernanceDecisionLayerTests(unittest.TestCase):
    def setUp(self):
        self.policy = GovernancePolicy()
        self.agent = GovernanceDecisionAgent(policy=self.policy)
        self.orchestrator = DecisionOrchestrator(agent=self.agent)

    def test_regime_thresholds_match_thesis_policy(self):
        self.assertEqual(self.policy.classify_regime(0.49).label, "Calm")
        self.assertEqual(self.policy.classify_regime(0.50).label, "Elevated")
        self.assertEqual(self.policy.classify_regime(0.84).label, "Elevated")
        self.assertEqual(self.policy.classify_regime(0.85).label, "Crisis")

    def test_allocation_method_selection_uses_regime_and_graph_context(self):
        calm = self.policy.select_allocation_method("Calm", graph_available=False)
        elevated = self.policy.select_allocation_method("Elevated", graph_available=False)
        crisis_graph = self.policy.select_allocation_method("Crisis", graph_available=True)
        failed = self.policy.select_allocation_method("Elevated", graph_available=True, solver_failed=True)

        self.assertEqual(calm.method, "mean_variance_or_equal_weight_baseline")
        self.assertEqual(elevated.method, "cvar_risk_constrained_allocation")
        self.assertEqual(crisis_graph.method, "graph_regularized_cvar_allocation")
        self.assertEqual(failed.method, "fallback_regularized_allocation")

    def test_hitl_required_for_crisis_turnover_concentration_and_missing_graph(self):
        action = self.policy.evaluate_hitl(
            regime_label="Crisis",
            metrics={"turnover": 0.42, "max_weight": 0.38},
            graph_required=True,
            graph_available=False,
        )

        self.assertTrue(action.required)
        self.assertIn("crisis_regime", action.reason_codes)
        self.assertIn("turnover_above_policy", action.reason_codes)
        self.assertIn("concentration_above_policy", action.reason_codes)
        self.assertIn("graph_data_missing", action.reason_codes)

    def test_forbidden_language_is_rejected_by_validator(self):
        result = self.policy.validate_response_text("Buy AAPL today and sell MSFT tomorrow.")

        self.assertFalse(result.valid)
        self.assertIn("forbidden_trading_language", result.issues)

    def test_weight_validation_requires_non_negative_weights_and_sum_near_one(self):
        valid = self.policy.validate_weights({"AAPL": 0.5, "MSFT": 0.5})
        invalid = self.policy.validate_weights({"AAPL": 0.8, "MSFT": -0.1})

        self.assertTrue(valid.valid)
        self.assertFalse(invalid.valid)
        self.assertIn("negative_weight", invalid.issues)
        self.assertIn("weights_do_not_sum_to_one", invalid.issues)

    def test_smart_plots_selects_bounded_relevant_subset(self):
        response = self.orchestrator.select_plots(
            {
                "message": "Show the important portfolio health plots only",
                "decision_context": {"decision_type": "portfolio_health_decision"},
                "metrics": {"instability_index": 0.61},
            }
        )

        self.assertGreaterEqual(len(response["plots"]), 5)
        self.assertLessEqual(len(response["plots"]), 8)
        self.assertTrue(all("plot_id" in item and "reason" in item for item in response["plots"]))
        self.assertTrue(all("priority" in item for item in response["plots"]))
        self.assertTrue(all(item["trigger_chips"] for item in response["plots"]))
        self.assertTrue(all(item["required_fields"] for item in response["plots"]))
        self.assertTrue(all(item["endpoint"] for item in response["plots"]))
        self.assertNotEqual(response["plot_mode"], "full_analytics")

    def test_smart_view_prompt_with_negated_all_88_stays_bounded(self):
        response = self.orchestrator.select_plots(
            {
                "message": "Pick the most relevant plots. Do not show all 88 plots unless I explicitly ask for Full Analytics View. Use Smart View only.",
                "decision_context": {"decision_type": "plot_decision"},
            }
        )

        self.assertEqual(response["plot_mode"], "smart_view")
        self.assertLessEqual(len(response["plots"]), 8)
        self.assertNotIn("registry", response)

    def test_full_plot_request_keeps_explicit_full_mode(self):
        response = self.orchestrator.select_plots(
            {
                "message": "Audit the full analytics plot registry across all 88 plots",
                "decision_context": {"decision_type": "plot_decision"},
            }
        )

        self.assertEqual(response["plot_mode"], "full_analytics")
        self.assertEqual(response["max_plots"], 88)
        self.assertEqual(len(response["registry"]), 88)
        plot_ids = [item["plot_id"] for item in response["registry"]]
        self.assertEqual(len(plot_ids), len(set(plot_ids)))
        self.assertTrue(all(item["status"] in {"ready", "missing_data", "fallback", "failed"} for item in response["registry"]))

    def test_plot_registry_covers_expected_tabs_and_advisory_metadata(self):
        from src.decision.plot_registry import FULL_PLOT_REGISTRY, TAB_NAMES

        self.assertEqual(len(FULL_PLOT_REGISTRY), 88)
        self.assertEqual({item["tab"] for item in FULL_PLOT_REGISTRY}, set(TAB_NAMES))
        self.assertTrue(all(item["explanation_present"] for item in FULL_PLOT_REGISTRY))
        self.assertTrue(all(item["advisory_interpretation_present"] for item in FULL_PLOT_REGISTRY))

    def test_plot_prompt_formatter_returns_required_smart_view_fields(self):
        from src.decision.plot_prompt_response import build_plot_prompt_response

        response = build_plot_prompt_response(
            "Pick the most relevant plots. Use Smart View only. Do not use buy/sell/trading language."
        )

        self.assertIsNotNone(response)
        self.assertIn("Smart View", response)
        self.assertIn("plot_id", response)
        self.assertIn("trigger chips", response.lower())
        self.assertIn("required fields", response.lower())
        self.assertIn("/api/", response)
        self.assertNotIn("buy", response.lower())
        self.assertNotIn("sell", response.lower())

    def test_apg_bench_data_quality_uses_default_context_and_expected_plots(self):
        from src.decision.apg_bench_response import build_apg_bench_response

        response = build_apg_bench_response("Run APG-Bench Test 1: Data Quality.")

        self.assertIsNotNone(response)
        self.assertIn("APG-Bench Test 1", response)
        self.assertIn("/api/analytics/eda", response)
        self.assertIn("analysis_id", response)
        self.assertIn("ticker availability", response.lower())
        self.assertIn("plot_09_missing_data_heatmap", response)
        self.assertIn("plot_01_adjusted_close_price_trend", response)
        self.assertIn("plot_10_outlier_return_detection_plot", response)
        self.assertIn("plot_03_daily_log_returns_plot", response)
        self.assertIn("What this means", response)
        self.assertIn("Why it matters", response)
        self.assertIn("How to read this", response)
        self.assertNotIn("buy", response.lower())
        self.assertNotIn("sell", response.lower())

    def test_ticker_and_sector_concentration_metrics_are_separate(self):
        from src.decision.concentration_metrics import compute_concentration_metrics

        weights = {f"T{i:02d}": 0.05 for i in range(1, 21)}
        sectors = {ticker: "Technology" for ticker in weights}
        metrics = compute_concentration_metrics(weights, sectors)

        self.assertAlmostEqual(metrics["ticker_hhi"], 0.05, places=6)
        self.assertAlmostEqual(metrics["ticker_effective_holdings"], 20.0, places=6)
        self.assertAlmostEqual(metrics["sector_hhi"], 1.0, places=6)
        self.assertAlmostEqual(metrics["sector_effective_sectors"], 1.0, places=6)
        self.assertNotEqual(metrics["ticker_hhi"], metrics["sector_hhi"])

    def test_advisory_safe_label_normalization(self):
        from src.decision.advisory_labels import normalize_advisory_language

        text = "Optimal Allocation Weights. Recommended allocation weights. Expected annualized return."
        normalized = normalize_advisory_language(text)

        self.assertIn("Advisory Allocation Weights", normalized)
        self.assertIn("Suggested exposure weights", normalized)
        self.assertIn("Estimated/backtested annualized return", normalized)
        self.assertNotIn("Optimal Allocation Weights", normalized)
        self.assertNotIn("Recommended allocation weights", normalized)

    def test_multiple_requested_plots_are_all_acknowledged(self):
        from src.decision.plot_acknowledgement import acknowledge_requested_plots

        text = acknowledge_requested_plots(
            ["ticker_concentration", "sector_concentration", "return_correlation_heatmap"]
        )

        self.assertIn("ticker_concentration", text)
        self.assertIn("sector_concentration", text)
        self.assertIn("return_correlation_heatmap", text)

    def test_heatmap_payload_normalization_adds_matrix_and_type(self):
        from src.agents.generate_dynamic_plot import _normalize_heatmap_payload

        payload = {
            "correlation_heatmap": [
                {"tickerX": "AAPL", "tickerY": "AAPL", "correlation": 1.0},
                {"tickerX": "AAPL", "tickerY": "MSFT", "correlation": 0.4},
                {"tickerX": "MSFT", "tickerY": "AAPL", "correlation": 0.4},
                {"tickerX": "MSFT", "tickerY": "MSFT", "correlation": 1.0},
            ]
        }

        normalized = _normalize_heatmap_payload(payload)

        self.assertIn("matrix", normalized)
        self.assertEqual(normalized["metadata"]["heatmap_type"], "correlation")
        self.assertEqual(normalized["matrix"]["AAPL"]["MSFT"], 0.4)

    def test_diversification_endpoint_exposes_separate_hhi_metrics(self):
        from api.analytics_router import get_diversification_diagnostics

        data = get_diversification_diagnostics(
            tickers="AAPL,MSFT,NVDA,AMZN,JPM",
            start_date="2024-01-01",
            end_date="2024-06-30",
        )

        self.assertIn("ticker_hhi", data)
        self.assertIn("ticker_effective_holdings", data)
        self.assertIn("sector_hhi", data)
        self.assertIn("sector_effective_sectors", data)
        self.assertAlmostEqual(data["ticker_effective_holdings"], 1 / data["ticker_hhi"], places=6)
        self.assertAlmostEqual(data["sector_effective_sectors"], 1 / data["sector_hhi"], places=6)

    def test_regime_only_response_does_not_return_allocation_weights(self):
        from src.decision.regime_response import build_regime_only_response

        response = build_regime_only_response(
            "Only answer regime question for U1. Do not generate allocation. Do not show optimal weights."
        )

        self.assertIsNotNone(response)
        self.assertIn("modules skipped: optimizer, allocation_engine", response)
        self.assertIn("allocation weights: not generated", response)
        self.assertNotIn("Advisory Allocation Weights", response)
        self.assertNotIn("Suggested exposure weights", response)

    def test_decision_response_contains_traceability_and_missing_graph_disclosure(self):
        response = self.orchestrator.decide(
            {
                "message": "Why is my portfolio critical? AAPL 70% MSFT 30%",
                "metrics": {"instability_index": 0.9},
                "graph": {"available": False},
            }
        )

        self.assertEqual(response["decision_type"], "instability_decision")
        self.assertTrue(response["hitl"]["required"])
        self.assertIn("graph_data_missing", response["hitl"]["reason_codes"])
        self.assertGreaterEqual(len(response["traceability"]["claims"]), 3)
        self.assertIn("limitations", response)

    def test_advisory_allocation_response_is_advisory_only_and_validated(self):
        response = self.orchestrator.advisory_allocation(
            {
                "message": "Give advisory allocation for AAPL MSFT NVDA",
                "tickers": ["AAPL", "MSFT", "NVDA"],
                "current_weights": {"AAPL": 0.6, "MSFT": 0.25, "NVDA": 0.15},
                "metrics": {"instability_index": 0.62},
            }
        )

        text = response["chatbot_plan"]["draft_response"]
        self.assertEqual(response["decision_type"], "advisory_allocation_decision")
        self.assertTrue(response["validation"]["valid"])
        self.assertTrue(response["advisory_allocation"]["weights"])
        self.assertNotIn("buy", text.lower())
        self.assertNotIn("sell", text.lower())

    def test_validate_endpoint_contract_checks_traceability_and_response_text(self):
        response = self.orchestrator.validate(
            {
                "response_text": "This is an advisory diversification review.",
                "decision": {
                    "traceability": {"claims": [{"claim_id": "c1", "source": "policy", "status": "validated"}]},
                    "validation": {"valid": True, "issues": []},
                },
            }
        )

        self.assertTrue(response["valid"])
        self.assertEqual(response["issues"], [])

    def test_governance_router_exposes_required_endpoint_paths(self):
        from api.governance_router import router

        paths = {route.path for route in router.routes}

        self.assertIn("/api/governance/decision", paths)
        self.assertIn("/api/governance/decision/portfolio-health", paths)
        self.assertIn("/api/governance/decision/advisory-allocation", paths)
        self.assertIn("/api/governance/decision/plots", paths)
        self.assertIn("/api/governance/decision/validate", paths)


if __name__ == "__main__":
    unittest.main()
