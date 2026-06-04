import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.memory.context_resolver import (
    ContextResolver,
    build_direct_context_response,
    build_missing_input_response,
    build_pending_execution_response,
)
from src.memory.audit_log import AuditLogger
from src.memory.fallback_computation import compute_hhi_bundle, compute_risk_contribution
from src.memory.intent_lock import build_intent_lock
from src.memory.missing_data_resolver import MissingDataResolver
from src.memory.memory_store import InProcessSessionMemoryStore
from src.memory.plot_validation import validate_plot_payload
from src.memory.session_state import U1_TECH_TICKERS, default_session_state


class SessionMemoryContextTests(unittest.TestCase):
    def setUp(self):
        self.resolver = ContextResolver()
        self.missing = MissingDataResolver()

    def resolve_full(self, message, state):
        return self.missing.resolve(self.resolver.resolve(message, state))

    def test_u1_follow_up_keeps_all_u1_tickers(self):
        state = default_session_state("s1")
        first = self.resolver.resolve("For U1 tickers, show ticker concentration", state)
        self.assertEqual(first["session_state"]["active_universe"], "U1")
        self.assertEqual(first["session_state"]["active_tickers"], U1_TECH_TICKERS)
        self.assertEqual(first["session_state"]["active_ticker_count"], 20)

        next_state = first["session_state"]
        second = self.resolver.resolve("now plot risk contribution", next_state)
        self.assertEqual(second["session_state"]["active_universe"], "U1")
        self.assertEqual(second["session_state"]["active_tickers"], U1_TECH_TICKERS)
        self.assertEqual(second["session_state"]["last_plot_id"], "plot_55_risk_contribution_by_ticker")

    def test_u1_registry_contains_expected_twenty_tickers(self):
        expected = [
            "AAPL", "ADBE", "ADI", "AMAT", "AMD", "AVGO", "CRM", "CSCO", "IBM", "INTU",
            "KLAC", "LRCX", "MSFT", "MU", "NOW", "NVDA", "ORCL", "PANW", "QCOM", "TXN",
        ]
        self.assertEqual(U1_TECH_TICKERS, expected)
        self.assertEqual(len(U1_TECH_TICKERS), 20)

    def test_same_prompt_equal_weight_proxy_approval_renders_u1_ticker_concentration(self):
        state = default_session_state("s1")
        prompt = (
            "For U1 tickers, generate only the Ticker Concentration Plot as a horizontal bar chart. "
            "Use all U1 tickers. If current weights are missing, use equal-weight current allocation "
            "proxy for this test and clearly label it as equal-weight proxy. Do not run optimizer."
        )

        resolved = self.resolve_full(prompt, state)
        weights = resolved["session_state"]["active_weights"]
        payload = resolved["fallback_result"]["plot_payload"]

        self.assertIsNone(resolved["pending_action"])
        self.assertEqual(resolved["fallback_result"]["status"], "success")
        self.assertTrue(resolved["validation_result"]["can_render"])
        self.assertEqual(weights["type"], "equal_weight_proxy")
        self.assertTrue(weights["approved_by_user"])
        self.assertEqual(weights["source"], "same_prompt_user_approved")
        self.assertEqual(len(weights["weights"]), 20)
        self.assertAlmostEqual(sum(weights["weights"].values()), 100.0)
        self.assertEqual(payload["plot_id"], "plot_42_ticker_concentration_plot")
        self.assertEqual(payload["chart_type"], "bar")
        self.assertEqual(payload["bar_mode"], "horizontal")
        self.assertEqual(payload["universe"], "U1")
        self.assertEqual(payload["ticker_count"], 20)
        self.assertEqual(payload["tickers_used"], U1_TECH_TICKERS)
        self.assertEqual(payload["x_axis"], "allocation_percent")
        self.assertEqual(payload["y_axis"], "ticker")
        self.assertEqual(payload["unit"], "%")
        self.assertEqual(payload["data_source"], "equal_weight_proxy")
        self.assertTrue(payload["fallback_used"])
        self.assertFalse(payload["optimizer_called"])
        self.assertFalse(payload["advisory_allocation_generated"])
        self.assertTrue(all(row["allocation_percent"] == 5.0 for row in payload["data"]))
        self.assertFalse(resolved["router_plan"]["execution"]["needs_optimizer"])

    def test_pending_action_does_not_override_list_u1_tickers_command(self):
        state = default_session_state("s1")
        pending = self.resolve_full("For U1 tickers, show ticker concentration", state)

        resolved = self.resolver.resolve("list the tickers of U1", pending["session_state"])
        response = build_direct_context_response(resolved)

        self.assertIsNone(resolved["pending_action"])
        self.assertIsNotNone(response)
        self.assertIn("source: universe_registry", response)
        self.assertIn("ticker_count: 20", response)
        self.assertIn(", ".join(U1_TECH_TICKERS), response)
        self.assertNotIn("current_weights", resolved["session_state"]["missing_inputs"])

    def test_fetch_u1_tickers_does_not_require_weights(self):
        state = default_session_state("s1")
        pending = self.resolve_full("For U1 tickers, show ticker concentration", state)

        resolved = self.resolver.resolve("fetch the u1 tickers", pending["session_state"])
        response = build_direct_context_response(resolved)

        self.assertIsNone(resolved["pending_action"])
        self.assertIsNotNone(response)
        self.assertIn("universe: U1", response)
        self.assertIn("ticker_count: 20", response)
        self.assertNotIn("Do you want me to use an equal-weight proxy", response)

    def test_calculate_real_values_explains_holdings_required(self):
        state = default_session_state("s1")
        pending = self.resolve_full("For U1 tickers, show ticker concentration", state)

        resolved = self.resolver.resolve("calculate the real values", pending["session_state"])
        response = build_direct_context_response(resolved)

        self.assertIsNone(resolved["pending_action"])
        self.assertIsNotNone(response)
        self.assertIn("cannot calculate real current allocation weights from price data alone", response)
        self.assertIn("shares", response)
        self.assertIn("invested amount", response)

    def test_pending_action_only_captures_short_confirmation_replies(self):
        state = default_session_state("s1")
        pending = self.resolve_full("For U1 tickers, show ticker concentration", state)

        yes_resolved = self.resolve_full("use proxy", pending["session_state"])
        self.assertEqual(yes_resolved["pending_status"], "executed")

        pending_again = self.resolve_full("For U1 tickers, show ticker concentration", state)
        new_command = self.resolver.resolve("fetch the U1 tickers", pending_again["session_state"])
        self.assertIsNone(new_command["pending_status"])
        self.assertIsNone(new_command["pending_action"])

    def test_missing_current_weights_create_pending_equal_weight_proxy_action(self):
        state = default_session_state("s1")
        resolved = self.resolve_full("For U1 tickers, show ticker concentration", state)
        pending = resolved["pending_action"]
        self.assertEqual(pending["type"], "use_equal_weight_proxy")
        self.assertEqual(pending["target_plot_id"], "plot_42_ticker_concentration_plot")
        self.assertIn("current_weights", resolved["session_state"]["missing_inputs"])
        self.assertFalse(resolved["validation_result"]["can_execute"])
        self.assertIsNone(build_direct_context_response(resolved))
        self.assertIn("equal-weight proxy", build_missing_input_response(resolved))

    def test_yes_after_proxy_prompt_executes_pending_action(self):
        state = default_session_state("s1")
        pending_resolved = self.resolve_full("For U1 tickers, show ticker concentration", state)
        yes_resolved = self.resolve_full("yes", pending_resolved["session_state"])

        weights = yes_resolved["session_state"]["active_weights"]
        self.assertEqual(weights["type"], "equal_weight_proxy")
        self.assertTrue(weights["approved_by_user"])
        self.assertEqual(len(weights["weights"]), 20)
        self.assertAlmostEqual(sum(weights["weights"].values()), 100.0)
        self.assertIn("equal-weight proxy", build_pending_execution_response(yes_resolved))
        self.assertEqual(yes_resolved["fallback_result"]["status"], "success")
        self.assertTrue(yes_resolved["fallback_result"]["plot_payload"]["proxy_declared"])

    def test_use_them_and_plot_executes_pending_action(self):
        state = default_session_state("s1")
        pending_resolved = self.resolve_full("For U1 tickers, show ticker concentration", state)
        use_resolved = self.resolve_full("use them and plot", pending_resolved["session_state"])

        self.assertEqual(use_resolved["pending_status"], "executed")
        self.assertEqual(use_resolved["session_state"]["active_weights"]["type"], "equal_weight_proxy")
        self.assertIsNone(use_resolved["session_state"]["pending_action"])

    def test_advisory_weights_are_not_used_as_current_weights(self):
        state = default_session_state("s1")
        state["active_universe"] = "U1"
        state["active_tickers"] = U1_TECH_TICKERS
        state["active_ticker_count"] = 20
        state["active_advisory_weights"] = {
            "available": True,
            "weights": {ticker: 5 for ticker in U1_TECH_TICKERS},
            "source": "latest_advisory",
            "target_date": "2025-12-31",
        }
        resolved = self.resolve_full("plot ticker concentration", state)
        self.assertIn("current_weights", resolved["session_state"]["missing_inputs"])
        self.assertEqual(resolved["session_state"]["active_weights"]["type"], "missing")

    def test_current_vs_advisory_grouped_bar_requires_both_weight_sets(self):
        state = default_session_state("s1")
        state["active_universe"] = "U1"
        state["active_tickers"] = U1_TECH_TICKERS
        resolved = self.resolve_full("show current vs advisory allocation by ticker", state)
        self.assertEqual(resolved["session_state"]["last_bar_mode"], "grouped")
        self.assertIn("advisory_weights", resolved["session_state"]["missing_inputs"])
        self.assertFalse(resolved["validation_result"]["can_execute"])

    def test_eigenvector_centrality_preserves_u1_and_no_placeholder_x(self):
        state = default_session_state("s1")
        state["active_universe"] = "U1"
        state["active_tickers"] = U1_TECH_TICKERS
        resolved = self.resolve_full("plot eigenvector centrality by ticker", state)
        self.assertEqual(resolved["session_state"]["last_plot_id"], "plot_61_eigenvector_centrality_by_ticker")
        self.assertNotIn("X", resolved["session_state"]["active_tickers"])
        self.assertEqual(resolved["fallback_result"]["status"], "unavailable")
        self.assertIn("placeholder", resolved["fallback_result"]["reason"])

    def test_last_25_messages_are_stored_compactly(self):
        store = InProcessSessionMemoryStore()
        for index in range(30):
            store.append_message("s1", "user", f"message {index}")
        rows = store.get_last_messages("s1")
        self.assertEqual(len(rows), 25)
        self.assertEqual(rows[0]["content"], "message 5")

    def test_u1_risk_contribution_missing_weights_asks_for_proxy_approval(self):
        state = default_session_state("s1")
        resolved = self.resolve_full("For U1, plot risk contribution", state)
        self.assertEqual(resolved["pending_action"]["type"], "use_equal_weight_proxy")
        self.assertIn("current_weights", resolved["session_state"]["missing_inputs"])
        self.assertFalse(resolved["validation_result"]["can_execute"])

    def test_after_yes_risk_contribution_is_computed_from_covariance(self):
        state = default_session_state("s1")
        pending = self.resolve_full("For U1, plot risk contribution", state)
        approved = self.resolve_full("yes", pending["session_state"])
        result = approved["fallback_result"]
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["plot_payload"]["plot_id"], "plot_55_risk_contribution_by_ticker")
        self.assertEqual(len(result["plot_payload"]["data"]), 20)
        self.assertIn("covariance_matrix", result["plot_payload"])
        total_rc = sum(row["risk_contribution_percent"] for row in result["plot_payload"]["data"])
        self.assertAlmostEqual(total_rc, 100.0, places=4)

    def test_hhi_separates_ticker_and_sector_hhi(self):
        weights = {ticker: 5.0 for ticker in U1_TECH_TICKERS}
        metrics = compute_hhi_bundle(weights, {ticker: "Technology" for ticker in U1_TECH_TICKERS})
        self.assertAlmostEqual(metrics["ticker_hhi"], 0.05)
        self.assertAlmostEqual(metrics["ticker_effective_holdings"], 20.0)
        self.assertAlmostEqual(metrics["sector_hhi"], 1.0)
        self.assertAlmostEqual(metrics["sector_effective_sectors"], 1.0)

    def test_allocation_change_does_not_render_without_both_weight_sets(self):
        state = default_session_state("s1")
        state["active_universe"] = "U1"
        state["active_tickers"] = U1_TECH_TICKERS
        state["active_weights"] = {
            "type": "equal_weight_proxy",
            "weights": {ticker: 5.0 for ticker in U1_TECH_TICKERS},
            "source": "test",
            "approved_by_user": True,
        }
        resolved = self.resolve_full("plot allocation change by ticker", state)
        self.assertEqual(resolved["fallback_result"]["status"], "missing_data")
        self.assertIn("advisory_weights", resolved["fallback_result"]["missing_inputs"])

    def test_strategy_comparison_asks_for_optimizer_permission(self):
        state = default_session_state("s1")
        state["active_universe"] = "U1"
        state["active_tickers"] = U1_TECH_TICKERS
        resolved = self.resolve_full("show strategy performance comparison", state)
        self.assertEqual(resolved["fallback_result"]["status"], "requires_confirmation")
        self.assertIn("optimizer_permission", resolved["fallback_result"]["missing_inputs"])

    def test_covariance_computed_from_returns_when_missing(self):
        weights = {ticker: 5.0 for ticker in U1_TECH_TICKERS}
        result = compute_risk_contribution(U1_TECH_TICKERS, weights, "2024-01-01", "2024-12-31")
        self.assertEqual(result["status"], "success")
        self.assertEqual(set(result["covariance_matrix"].keys()), set(U1_TECH_TICKERS))

    def test_data_quality_only_intent_lock_blocks_optimizer(self):
        lock = build_intent_lock("Only data quality for U1")
        self.assertEqual(lock["intent"], "data_quality_only")
        self.assertIn("optimizer", lock["blocked_modules"])

    def test_regime_only_intent_lock_blocks_optimal_weights(self):
        lock = build_intent_lock(
            "Only regime for U1. Do not generate advisory allocation.",
            {"sub_intent": "instability_regime"},
        )
        self.assertEqual(lock["intent"], "regime_diagnosis_only")
        self.assertIn("optimal_weights", lock["blocked_modules"])

    def test_diversification_only_intent_lock_blocks_new_allocation(self):
        lock = build_intent_lock("Only diversification diagnosis for U1")
        self.assertEqual(lock["intent"], "diversification_diagnosis_only")
        self.assertIn("advisory_allocation", lock["blocked_modules"])

    def test_plot_validation_blocks_u1_with_eight_tickers(self):
        request = {
            "requested_plot_id": "plot_42_ticker_concentration_plot",
            "requested_universe": "U1",
            "requested_chart_type": "bar",
            "requested_bar_mode": "horizontal",
            "requested_subset_explicit": False,
        }
        payload = {
            "plot_id": "plot_42_ticker_concentration_plot",
            "universe": "U1",
            "chart_type": "bar",
            "bar_mode": "horizontal",
            "tickers": U1_TECH_TICKERS[:8],
            "data": [],
            "data_source": "equal_weight_proxy",
            "proxy_used": True,
            "proxy_declared": True,
        }
        result = validate_plot_payload(request, payload)
        self.assertFalse(result["can_render"])
        self.assertIn("all 20 U1 tickers", result["reason"])

    def test_plot_validation_blocks_mismatches_and_placeholder_x(self):
        base_request = {
            "requested_plot_id": "plot_55_risk_contribution_by_ticker",
            "requested_universe": "U1",
            "requested_chart_type": "bar",
            "requested_bar_mode": "horizontal",
            "requested_subset_explicit": True,
            "required_fields": ["ticker", "risk_contribution_percent"],
        }
        payload = {
            "plot_id": "wrong_plot",
            "universe": "U1",
            "chart_type": "line",
            "bar_mode": "vertical",
            "tickers": ["X"],
            "data": [{"ticker": "X"}],
            "data_source": "",
        }
        result = validate_plot_payload(base_request, payload)
        self.assertFalse(result["can_render"])
        self.assertIn("requested_plot_id", result["reason"])
        self.assertIn("chart_type", result["reason"])
        self.assertIn("bar_mode", result["reason"])
        self.assertIn("placeholder", result["reason"])
        self.assertIn("data source", result["reason"])

    def test_every_computed_plot_returns_metadata(self):
        state = default_session_state("s1")
        pending = self.resolve_full("For U1, plot risk contribution", state)
        approved = self.resolve_full("yes", pending["session_state"])
        payload = approved["fallback_result"]["plot_payload"]
        for key in (
            "plot_id",
            "chart_type",
            "bar_mode",
            "universe",
            "ticker_count",
            "tickers_used",
            "x_axis",
            "y_axis",
            "unit",
            "data_source",
            "fallback_used",
            "confidence",
            "limitations",
        ):
            self.assertIn(key, payload)

    def test_audit_logger_records_blocked_render_reason(self):
        logger = AuditLogger()
        row = logger.log(
            {
                "session_id": "s1",
                "user_message": "plot centrality",
                "intent": "plot_only",
                "resolved_universe": "U1",
                "tool_called": "plot_validation",
                "plot_id": "plot_61_eigenvector_centrality_by_ticker",
                "data_source": None,
                "status": "blocked",
                "failure_reason": "institutional graph data missing",
            }
        )
        self.assertEqual(row["status"], "blocked")
        self.assertIn("graph", row["failure_reason"])

    def test_missing_data_heatmap_includes_pre_inception_periods(self):
        from api import analytics_router

        dates = pd.date_range("2024-01-01", "2024-01-10", freq="B")
        prices = pd.DataFrame(
            {
                "AAPL": [100, 101, 102, 103, 104, 105, 106, 107],
                "NEW": [None, None, None, 50, 51, None, 53, 54],
            },
            index=dates,
        )
        raw = {
            "AAPL": prices["AAPL"].dropna(),
            "NEW": prices["NEW"].dropna(),
        }
        with patch.object(analytics_router, "get_portfolio_prices", return_value=(prices, False)), patch.object(
            analytics_router, "extract_prices_from_db", return_value=raw
        ):
            result = analytics_router.get_eda_analytics("AAPL,NEW", "2024-01-01", "2024-01-10")
        metadata = result["missing_data_metadata"]
        self.assertEqual(metadata["encoding"]["pre_inception"], -1)
        self.assertTrue(any(row["NEW"] == -1 for row in result["missing_data"]))
        self.assertTrue(any(row["NEW"] == 0 for row in result["missing_data"]))


if __name__ == "__main__":
    unittest.main()
