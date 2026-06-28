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
from src.memory.fallback_computation import compute_hhi_bundle, compute_risk_contribution, compute_risk_return_scatter
from src.memory.intent_lock import build_intent_lock
from src.memory.missing_data_resolver import MissingDataResolver
from src.memory.memory_store import InProcessSessionMemoryStore
from src.memory.plot_validation import validate_plot_payload
from src.memory.response_contract import build_response_contract
from src.memory.session_state import U1_TECH_TICKERS, default_session_state
from src.memory.todo_state import TODO_STATUS_VALUES


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
        self.assertEqual(payload["chart_tier"], "free")
        self.assertEqual(payload["component"], "BarChart")
        self.assertFalse(payload["requires_premium"])
        self.assertTrue(payload["fallback_used"])
        self.assertFalse(payload["optimizer_called"])
        self.assertFalse(payload["advisory_allocation_generated"])
        self.assertTrue(all(row["allocation_percent"] == 5.0 for row in payload["data"]))
        self.assertFalse(resolved["router_plan"]["execution"]["needs_optimizer"])

    def test_planner_marks_safe_same_prompt_proxy_as_computed_fallback(self):
        state = default_session_state("s1")
        prompt = (
            "For U1 tickers, generate only the Ticker Concentration Plot as a horizontal bar chart. "
            "Use all U1 tickers. If current weights are missing, use equal-weight current allocation "
            "proxy for this test and clearly label it as equal-weight proxy. Do not run optimizer."
        )

        resolved = self.resolve_full(prompt, state)
        plan = resolved["session_state"]["current_plan"]
        todos = {todo["name"]: todo for todo in plan["todos"]}

        self.assertEqual(plan["user_intent"], "ticker_concentration_plot")
        self.assertEqual(plan["response_scope"], "plot_only")
        self.assertEqual(plan["universe"], "U1")
        self.assertEqual(plan["requested_plot_id"], "plot_42_ticker_concentration_plot")
        self.assertTrue(plan["ready_to_execute"])
        self.assertTrue(plan["ready_to_render"])
        self.assertIsNone(plan["blocked_step"])
        self.assertIsNone(plan["pending_user_choice"])
        self.assertEqual(todos["resolve_universe"]["status"], "done")
        self.assertEqual(todos["resolve_tickers"]["status"], "done")
        self.assertEqual(todos["resolve_weights"]["status"], "computed_by_fallback")
        self.assertEqual(todos["resolve_weights"]["fallback_method"], "equal_weight_proxy")
        self.assertEqual(todos["resolve_weights"]["resolved_data"]["weight_count"], 20)
        self.assertEqual(todos["generate_plot_payload"]["status"], "done")
        self.assertEqual(todos["validate_plot"]["status"], "done")
        self.assertEqual(set(TODO_STATUS_VALUES), {
            "pending",
            "done",
            "blocked",
            "computed_by_fallback",
            "requires_user_input",
            "skipped_by_intent_lock",
            "failed_validation",
        })

    def test_planner_auto_resolves_missing_weights_with_proxy(self):
        state = default_session_state("s1")
        resolved = self.resolve_full("For U1 tickers, show ticker concentration", state)
        plan = resolved["session_state"]["current_plan"]
        todos = {todo["name"]: todo for todo in plan["todos"]}

        self.assertEqual(plan["user_intent"], "ticker_concentration_plot")
        self.assertIsNone(plan["blocked_step"])
        self.assertIsNone(plan["pending_user_choice"])
        self.assertTrue(plan["ready_to_execute"])
        self.assertTrue(plan["ready_to_render"])
        self.assertEqual(todos["resolve_weights"]["status"], "computed_by_fallback")
        self.assertEqual(todos["resolve_weights"]["fallback_method"], "equal_weight_proxy")

    def test_pending_action_does_not_override_list_u1_tickers_command(self):
        state = default_session_state("s1")
        pending = self.resolve_full("For U1 tickers, show ticker concentration", state)

        resolved = self.resolver.resolve("list the tickers of U1", pending["session_state"])
        response = build_direct_context_response(resolved)
        plan = resolved["session_state"]["current_plan"]

        self.assertIsNone(resolved["pending_action"])
        self.assertIsNotNone(response)
        self.assertIn("source: universe_registry", response)
        self.assertIn("ticker_count: 20", response)
        self.assertIn(", ".join(U1_TECH_TICKERS), response)
        self.assertNotIn("current_weights", resolved["session_state"]["missing_inputs"])
        self.assertEqual(plan["user_intent"], "list_universe_tickers")
        self.assertEqual(plan["response_scope"], "data_lookup")
        self.assertTrue(plan["ready_to_execute"])
        self.assertIsNone(plan["pending_user_choice"])
        self.assertNotIn("resolve_weights", {todo["name"] for todo in plan["todos"]})

    def test_fetch_u1_tickers_does_not_require_weights(self):
        state = default_session_state("s1")
        pending = self.resolve_full("For U1 tickers, show ticker concentration", state)

        resolved = self.resolve_full("fetch the u1 tickers", pending["session_state"])
        response = build_direct_context_response(resolved)

        self.assertIsNone(resolved["pending_action"])
        self.assertEqual(resolved["fallback_result"]["status"], "not_applicable")
        self.assertIsNotNone(response)
        self.assertIn("universe: U1", response)
        self.assertIn("ticker_count: 20", response)
        self.assertNotIn("Do you want me to use an equal-weight proxy", response)

    def test_calculate_real_values_explains_holdings_required(self):
        state = default_session_state("s1")
        pending = self.resolve_full("For U1 tickers, show ticker concentration", state)

        resolved = self.resolver.resolve("calculate the real values", pending["session_state"])
        response = build_direct_context_response(resolved)
        plan = resolved["session_state"]["current_plan"]

        self.assertIsNone(resolved["pending_action"])
        self.assertIsNotNone(response)
        self.assertIn("cannot calculate real current allocation weights from price data alone", response)
        self.assertIn("shares", response)
        self.assertIn("invested amount", response)
        self.assertEqual(plan["user_intent"], "explain_real_current_weights")
        self.assertEqual(plan["response_scope"], "data_lookup")
        self.assertTrue(plan["ready_to_execute"])
        self.assertIsNone(plan["pending_user_choice"])

    def test_proxy_followup_does_not_require_pending_action(self):
        state = default_session_state("s1")
        pending = self.resolve_full("For U1 tickers, show ticker concentration", state)

        yes_resolved = self.resolve_full("use proxy", pending["session_state"])
        self.assertIsNone(yes_resolved["pending_status"])
        self.assertIsNone(yes_resolved["pending_action"])

        pending_again = self.resolve_full("For U1 tickers, show ticker concentration", state)
        new_command = self.resolver.resolve("fetch the U1 tickers", pending_again["session_state"])
        self.assertIsNone(new_command["pending_status"])
        self.assertIsNone(new_command["pending_action"])

    def test_planner_rolls_current_plan_to_last_plan_on_explicit_command(self):
        state = default_session_state("s1")
        pending = self.resolve_full("For U1 tickers, show ticker concentration", state)
        previous_plan_id = pending["session_state"]["current_plan"]["plan_id"]

        resolved = self.resolver.resolve("list the tickers of U1", pending["session_state"])

        self.assertEqual(resolved["session_state"]["last_plan"]["plan_id"], previous_plan_id)
        self.assertEqual(resolved["session_state"]["current_plan"]["user_intent"], "list_universe_tickers")
        self.assertNotEqual(resolved["session_state"]["current_plan"]["plan_id"], previous_plan_id)

    def test_missing_current_weights_auto_use_equal_weight_proxy(self):
        state = default_session_state("s1")
        resolved = self.resolve_full("For U1 tickers, show ticker concentration", state)
        weights = resolved["session_state"]["active_weights"]
        self.assertIsNone(resolved["pending_action"])
        self.assertEqual(weights["type"], "equal_weight_proxy")
        self.assertEqual(weights["source"], "automatic_default_missing_current_weights")
        self.assertEqual(resolved["session_state"]["missing_inputs"], [])
        self.assertEqual(resolved["fallback_result"]["status"], "success")
        self.assertIn("equal_weight_proxy", build_direct_context_response(resolved))
        self.assertIsNone(build_missing_input_response(resolved))

    def test_yes_after_proxy_prompt_executes_pending_action(self):
        state = default_session_state("s1")
        pending_resolved = self.resolve_full("For U1 tickers, show ticker concentration", state)
        yes_resolved = self.resolve_full("yes", pending_resolved["session_state"])

        weights = yes_resolved["session_state"]["active_weights"]
        self.assertEqual(weights["type"], "equal_weight_proxy")
        self.assertFalse(weights["approved_by_user"])
        self.assertEqual(len(weights["weights"]), 20)
        self.assertAlmostEqual(sum(weights["weights"].values()), 100.0)
        self.assertIsNone(build_pending_execution_response(yes_resolved))
        self.assertEqual(yes_resolved["fallback_result"]["status"], "success")
        self.assertTrue(yes_resolved["fallback_result"]["plot_payload"]["proxy_declared"])

    def test_use_them_and_plot_executes_pending_action(self):
        state = default_session_state("s1")
        pending_resolved = self.resolve_full("For U1 tickers, show ticker concentration", state)
        use_resolved = self.resolve_full("use them and plot", pending_resolved["session_state"])

        self.assertIsNone(use_resolved["pending_status"])
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
        self.assertEqual(resolved["session_state"]["missing_inputs"], [])
        self.assertEqual(resolved["session_state"]["active_weights"]["type"], "equal_weight_proxy")
        self.assertEqual(resolved["fallback_result"]["status"], "success")

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

    def test_u1_risk_contribution_missing_weights_uses_auto_proxy(self):
        state = default_session_state("s1")
        resolved = self.resolve_full("For U1, plot risk contribution", state)
        self.assertIsNone(resolved["pending_action"])
        self.assertEqual(resolved["session_state"]["missing_inputs"], [])
        self.assertEqual(resolved["session_state"]["active_weights"]["type"], "equal_weight_proxy")
        self.assertEqual(resolved["fallback_result"]["status"], "success")

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

    def test_plot_validation_premium_disabled_uses_fallback_state(self):
        request = {
            "requested_plot_id": "return_range_by_ticker",
            "requested_chart_type": "rangeBar",
            "required_fields": ["ticker", "min_return", "max_return"],
        }
        payload = {
            "plot_id": "return_range_by_ticker",
            "chart_type": "rangeBar",
            "chart_tier": "premium",
            "component": "BarChartPremium",
            "requires_premium": True,
            "premium_enabled": False,
            "fallback_chart": "standard_min_max_bar",
            "data": [{"ticker": "AAPL", "min_return": -2.3, "max_return": 3.1}],
            "data_source": "historical_returns",
        }
        with patch.dict("os.environ", {"ENABLE_MUI_PREMIUM_CHARTS": "false"}):
            result = validate_plot_payload(request, payload)
        self.assertTrue(result["can_render"])
        self.assertEqual(result["status"], "premium_unavailable")
        self.assertIn("fallback", " ".join(result["warnings"]).lower())

    def test_plot_validation_blocks_invalid_premium_range_shapes(self):
        request = {
            "requested_plot_id": "return_range_by_ticker",
            "requested_chart_type": "rangeBar",
            "required_fields": ["ticker", "min_return", "max_return"],
        }
        payload = {
            "plot_id": "return_range_by_ticker",
            "chart_type": "rangeBar",
            "chart_tier": "premium",
            "component": "BarChartPremium",
            "requires_premium": True,
            "fallback_chart": "standard_min_max_bar",
            "data": [{"ticker": "AAPL", "min_return": -2.3}],
            "data_source": "historical_returns",
        }
        result = validate_plot_payload(request, payload)
        self.assertFalse(result["can_render"])
        self.assertIn("max_return", result["reason"])

    def test_plot_validation_blocks_missing_mirrored_weight_series(self):
        request = {
            "requested_plot_id": "current_vs_advisory_mirrored_bar",
            "requested_chart_type": "mirroredBar",
            "required_fields": ["ticker", "current_weight", "advisory_weight"],
        }
        payload = {
            "plot_id": "current_vs_advisory_mirrored_bar",
            "chart_type": "mirroredBar",
            "chart_tier": "premium",
            "component": "BarChartPremium",
            "requires_premium": True,
            "fallback_chart": "grouped_bar",
            "data": [{"ticker": "AAPL", "current_weight": 20}],
            "data_source": "current_and_advisory_weights",
        }
        result = validate_plot_payload(request, payload)
        self.assertFalse(result["can_render"])
        self.assertIn("advisory_weight", result["reason"])

    def test_return_range_by_ticker_premium_payload_is_computed_without_optimizer(self):
        state = default_session_state("s1")
        resolved = self.resolve_full(
            "For U1 tickers, plot return range by ticker using a premium range bar chart. Do not run optimizer.",
            state,
        )
        payload = resolved["fallback_result"]["plot_payload"]

        self.assertEqual(resolved["fallback_result"]["status"], "success")
        self.assertEqual(payload["plot_id"], "return_range_by_ticker")
        self.assertEqual(payload["chart_type"], "rangeBar")
        self.assertEqual(payload["chart_tier"], "premium")
        self.assertTrue(payload["requires_premium"])
        self.assertEqual(payload["fallback_chart"], "standard_min_max_bar")
        self.assertFalse(payload["optimizer_called"])
        self.assertEqual(len(payload["data"]), 20)
        self.assertTrue(all({"ticker", "min_return", "max_return"} <= set(row) for row in payload["data"]))

    def test_adjusted_close_line_plot_does_not_call_optimizer(self):
        state = default_session_state("s1")
        resolved = self.resolve_full(
            "For AAPL, generate only the historical adjusted close line plot from 2024-01-01 to 2024-12-31. "
            "Do not run optimizer. Do not generate advisory allocation.",
            state,
        )
        payload = resolved["fallback_result"]["plot_payload"]

        self.assertEqual(resolved["fallback_result"]["status"], "success")
        self.assertEqual(payload["plot_id"], "historical_adjusted_close")
        self.assertEqual(payload["plot_type"], "line")
        self.assertEqual(payload["chart_type"], "line")
        self.assertEqual(payload["tickers_used"], ["AAPL"])
        self.assertEqual(payload["x_axis"], "date")
        self.assertEqual(payload["y_axis"], "adjusted_close")
        self.assertFalse(payload["connect_nulls"])
        self.assertFalse(payload["optimizer_called"])
        self.assertFalse(payload["advisory_allocation_generated"])

    def test_multi_ticker_price_prompt_uses_normalized_line_by_default(self):
        state = default_session_state("s1")
        resolved = self.resolve_full(
            "Compare AAPL MSFT NVDA with a line chart from 2024-01-01 to 2024-12-31. Do not run optimizer.",
            state,
        )
        payload = resolved["fallback_result"]["plot_payload"]

        self.assertEqual(resolved["fallback_result"]["status"], "success")
        self.assertEqual(payload["plot_id"], "normalized_price_comparison")
        self.assertEqual(payload["fallback_method"], "normalize_adjusted_close_to_100")
        self.assertEqual(set(payload["tickers_used"]), {"AAPL", "MSFT", "NVDA"})
        first_points = {series["name"]: series["data"][0]["y"] for series in payload["series"]}
        for value in first_points.values():
            self.assertAlmostEqual(value, 100.0)
        self.assertFalse(payload["optimizer_called"])

    def test_regime_plot_only_returns_instability_line_without_allocation(self):
        state = default_session_state("s1")
        state["active_universe"] = "U1"
        state["active_tickers"] = U1_TECH_TICKERS
        state["active_ticker_count"] = len(U1_TECH_TICKERS)
        resolved = self.resolve_full(
            "Is my portfolio calm, elevated, or crisis? Show plot only. Do not generate allocation.",
            state,
        )
        payload = resolved["fallback_result"]["plot_payload"]

        self.assertEqual(payload["plot_id"], "instability_index_over_time")
        self.assertEqual(payload["plot_type"], "line")
        self.assertFalse(payload["optimizer_called"])
        self.assertFalse(payload["advisory_allocation_generated"])
        self.assertEqual(len(payload["tickers_used"]), 20)
        self.assertEqual(payload["fallback_chart"], "basic_instability_line")
        self.assertIn("optimizer", resolved["session_state"]["last_modules_skipped"])

    def test_line_validation_blocks_unsorted_dates_and_optimizer_leakage(self):
        request = {
            "requested_plot_id": "historical_adjusted_close",
            "requested_chart_type": "line",
            "requested_tickers": ["AAPL"],
        }
        payload = {
            "plot_id": "historical_adjusted_close",
            "plot_type": "line",
            "chart_type": "line",
            "tickers": ["AAPL"],
            "tickers_used": ["AAPL"],
            "x_axis": "date",
            "y_axis": "adjusted_close",
            "required_fields": ["date", "adjusted_close"],
            "data": [
                {"date": "2024-01-03", "ticker": "AAPL", "adjusted_close": 101},
                {"date": "2024-01-02", "ticker": "AAPL", "adjusted_close": 100},
            ],
            "data_source": "historical_price_database",
            "optimizer_called": True,
        }
        result = validate_plot_payload(request, payload)

        self.assertFalse(result["can_render"])
        self.assertIn("sorted ascending", result["reason"])
        self.assertIn("optimizer", result["reason"])

    def test_risk_return_scatter_computes_from_adjusted_close_without_optimizer(self):
        state = default_session_state("s1")
        resolved = self.resolve_full(
            "For U1, generate only the risk-return scatter plot. Use adjusted close price history. "
            "Do not run optimizer. Do not generate advisory allocation.",
            state,
        )
        payload = resolved["fallback_result"]["plot_payload"]

        self.assertEqual(resolved["fallback_result"]["status"], "success")
        self.assertEqual(payload["plot_id"], "risk_return_scatter")
        self.assertEqual(payload["plot_type"], "scatter")
        self.assertEqual(payload["chart_type"], "scatter")
        self.assertEqual(payload["x_axis"], "annualized_volatility_percent")
        self.assertEqual(payload["y_axis"], "annualized_return_percent")
        self.assertEqual(payload["point_id"], "ticker")
        self.assertEqual(payload["point_count"], 20)
        self.assertEqual(payload["tickers_used"], U1_TECH_TICKERS)
        self.assertEqual(payload["fallback_method"], "compute_return_and_volatility_from_adjusted_close")
        self.assertFalse(payload["optimizer_called"])
        self.assertFalse(payload["advisory_allocation_generated"])

    def test_risk_return_scatter_explanation_words_do_not_become_tickers(self):
        state = default_session_state("s1")
        resolved = self.resolve_full(
            "Generate only a risk-return scatter plot for U1 using adjusted close price history. "
            "Do not run optimizer. Do not generate advisory allocation. Return the plot_id, "
            "chart_type, point_count, x/y fields, and explain what the dots mean.",
            state,
        )
        plot_request = resolved["resolved_context"]["plot_request"]
        payload = resolved["fallback_result"]["plot_payload"]

        self.assertFalse(plot_request["requested_subset_explicit"])
        self.assertEqual(resolved["session_state"]["active_tickers"], U1_TECH_TICKERS)
        self.assertEqual(payload["tickers_used"], U1_TECH_TICKERS)
        self.assertEqual(payload["point_count"], 20)
        self.assertNotIn("DOTS", payload["tickers_used"])
        self.assertNotIn("MEAN", payload["tickers_used"])

    def test_risk_return_scatter_does_not_require_current_weights(self):
        state = default_session_state("s1")
        resolved = self.resolve_full("For U1, show risk-return scatter", state)

        self.assertEqual(resolved["fallback_result"]["status"], "success")
        self.assertNotIn("current_weights", resolved["session_state"]["missing_inputs"])
        self.assertIsNone(resolved["pending_action"])

    def test_bubble_risk_return_auto_uses_weight_proxy_for_size(self):
        state = default_session_state("s1")
        resolved = self.resolve_full("For U1, show risk-return bubble scatter", state)

        self.assertIsNone(resolved["pending_action"])
        self.assertEqual(resolved["session_state"]["missing_inputs"], [])
        self.assertEqual(resolved["session_state"]["active_weights"]["type"], "equal_weight_proxy")
        self.assertEqual(resolved["fallback_result"]["status"], "success")

    def test_after_yes_bubble_risk_return_uses_weight_size_axis_without_optimizer(self):
        state = default_session_state("s1")
        pending = self.resolve_full("For U1, show risk-return bubble scatter", state)
        approved = self.resolve_full("yes", pending["session_state"])
        payload = approved["fallback_result"]["plot_payload"]

        self.assertEqual(approved["fallback_result"]["status"], "success")
        self.assertEqual(payload["plot_id"], "bubble_risk_return")
        self.assertEqual(payload["chart_type"], "bubble_scatter")
        self.assertEqual(payload["size_axis"], "bubble_size_value")
        self.assertTrue(all(row["bubble_size_value"] > 0 for row in payload["data"]))
        self.assertFalse(payload["optimizer_called"])

    def test_scatter_regression_requires_and_uses_two_tickers(self):
        state = default_session_state("s1")
        resolved = self.resolve_full(
            "For AAPL MSFT, show regression scatter. Do not run optimizer.",
            state,
        )
        payload = resolved["fallback_result"]["plot_payload"]

        self.assertEqual(resolved["fallback_result"]["status"], "success")
        self.assertEqual(payload["plot_id"], "scatter_with_regression_line")
        self.assertEqual(payload["chart_type"], "scatter_regression")
        self.assertTrue(payload["regression_used"])
        self.assertEqual(payload["tickers_used"], ["AAPL", "MSFT"])
        self.assertIn("regression_line", payload)
        self.assertGreaterEqual(payload["point_count"], 3)

    def test_ownership_overlap_scatter_blocks_without_graph_data(self):
        state = default_session_state("s1")
        state["active_universe"] = "U1"
        state["active_tickers"] = U1_TECH_TICKERS
        resolved = self.resolve_full("Show ownership overlap vs correlation scatter for U1", state)

        self.assertEqual(resolved["fallback_result"]["status"], "unavailable")
        self.assertIn("institutional_graph_data", resolved["fallback_result"]["missing_inputs"])
        self.assertEqual(resolved["session_state"]["current_plan"]["blocked_step"], "resolve_graph_data")

    def test_scatter_validation_blocks_too_few_points_and_optimizer_leakage(self):
        request = {
            "requested_plot_id": "risk_return_scatter",
            "requested_chart_type": "scatter",
            "required_fields": ["ticker", "annualized_volatility_percent", "annualized_return_percent"],
        }
        payload = {
            "plot_id": "risk_return_scatter",
            "plot_type": "scatter",
            "chart_type": "scatter",
            "x_axis": "annualized_volatility_percent",
            "y_axis": "annualized_return_percent",
            "point_id": "ticker",
            "point_count": 1,
            "tickers": ["AAPL"],
            "data": [{"ticker": "AAPL", "annualized_volatility_percent": 18.0, "annualized_return_percent": 12.0}],
            "series": [{"name": "Technology", "data": [{"x": 18.0, "y": 12.0, "id": "AAPL"}]}],
            "data_source": "historical_price_database",
            "optimizer_called": True,
        }
        result = validate_plot_payload(request, payload)

        self.assertFalse(result["can_render"])
        self.assertIn("at least two valid points", result["reason"])
        self.assertIn("optimizer", result["reason"])

    def test_scatter_computation_returns_metadata_bundle(self):
        result = compute_risk_return_scatter(["AAPL", "MSFT"], "2024-01-01", "2024-12-31")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["point_count"], 2)
        self.assertEqual(result["x_axis"], "annualized_volatility_percent")
        self.assertEqual(result["y_axis"], "annualized_return_percent")
        self.assertEqual(result["point_id"], "ticker")
        self.assertFalse(result.get("optimizer_called", False))

    def test_sector_allocation_donut_computes_from_ticker_weights(self):
        state = default_session_state("s1")
        state["active_tickers"] = ["AAPL", "MSFT", "JPM"]
        state["active_ticker_count"] = 3
        state["active_weights"] = {
            "type": "actual_current_weights",
            "weights": {"AAPL": 50.0, "MSFT": 30.0, "JPM": 20.0},
            "source": "test",
        }

        resolved = self.resolve_full("Show sector allocation donut. Do not run optimizer.", state)
        payload = resolved["fallback_result"]["plot_payload"]

        self.assertEqual(resolved["fallback_result"]["status"], "success")
        self.assertEqual(payload["plot_id"], "sector_allocation_donut")
        self.assertEqual(payload["plot_type"], "pie")
        self.assertEqual(payload["chart_type"], "donut")
        self.assertEqual(payload["category_field"], "sector")
        self.assertEqual(payload["value_field"], "weight_percent")
        self.assertAlmostEqual(sum(row["weight_percent"] for row in payload["data"]), 100.0)
        self.assertFalse(payload["optimizer_called"])
        self.assertFalse(payload["advisory_allocation_generated"])

    def test_ticker_allocation_donut_falls_back_when_too_many_slices(self):
        state = default_session_state("s1")
        state["active_universe"] = "U1"
        state["active_tickers"] = U1_TECH_TICKERS
        state["active_ticker_count"] = len(U1_TECH_TICKERS)
        state["active_weights"] = {
            "type": "actual_current_weights",
            "weights": {ticker: 5.0 for ticker in U1_TECH_TICKERS},
            "source": "test",
        }

        resolved = self.resolve_full("Show ticker allocation donut. Do not run optimizer.", state)
        payload = resolved["fallback_result"]["plot_payload"]

        self.assertEqual(resolved["fallback_result"]["status"], "success")
        self.assertEqual(payload["plot_id"], "ticker_allocation_donut")
        self.assertEqual(payload["plot_type"], "bar")
        self.assertEqual(payload["chart_type"], "bar")
        self.assertTrue(payload["fallback_rendered"])
        self.assertEqual(payload["fallback_chart"], "ticker_concentration_bar")
        self.assertEqual(payload["slice_count"], 20)
        self.assertTrue(resolved["validation_result"]["can_render"])

    def test_sector_ticker_nested_donut_blocks_missing_sector_mapping(self):
        state = default_session_state("s1")
        state["active_tickers"] = ["ZZZZ"]
        state["active_ticker_count"] = 1
        state["active_weights"] = {
            "type": "actual_current_weights",
            "weights": {"ZZZZ": 100.0},
            "source": "test",
        }

        resolved = self.resolve_full("Show sector ticker nested donut. Do not run optimizer.", state)

        self.assertEqual(resolved["fallback_result"]["status"], "unavailable")
        self.assertIn("sector_mapping", resolved["fallback_result"]["missing_inputs"])
        self.assertEqual(resolved["session_state"]["current_plan"]["blocked_step"], "resolve_sector_mapping")

    def test_risk_contribution_donut_computes_without_optimizer(self):
        state = default_session_state("s1")
        state["active_tickers"] = ["AAPL", "MSFT", "NVDA"]
        state["active_ticker_count"] = 3
        state["active_date_range"] = {"start": "2024-01-01", "end": "2024-12-31"}
        state["active_weights"] = {
            "type": "actual_current_weights",
            "weights": {"AAPL": 40.0, "MSFT": 35.0, "NVDA": 25.0},
            "source": "test",
        }

        resolved = self.resolve_full("Show risk contribution donut. Do not run optimizer.", state)
        payload = resolved["fallback_result"]["plot_payload"]

        self.assertEqual(resolved["fallback_result"]["status"], "success")
        self.assertEqual(payload["plot_id"], "risk_contribution_donut")
        self.assertEqual(payload["plot_type"], "pie")
        self.assertIn("covariance_matrix", payload)
        self.assertAlmostEqual(sum(row["risk_contribution_percent"] for row in payload["data"]), 100.0, places=4)
        self.assertFalse(payload["optimizer_called"])

    def test_pie_validation_blocks_bad_percent_total_and_time_series(self):
        request = {
            "requested_plot_id": "sector_allocation_donut",
            "requested_chart_type": "donut",
        }
        payload = {
            "plot_id": "sector_allocation_donut",
            "plot_type": "pie",
            "chart_type": "donut",
            "category_field": "sector",
            "value_field": "weight_percent",
            "unit": "%",
            "data": [{"sector": "Technology", "weight_percent": 80.0}],
            "series": [{"name": "sector_weight_percent", "data": [{"id": "Technology", "value": 80.0}]}],
            "total_value": 80.0,
            "data_source": "test",
        }
        result = validate_plot_payload(request, payload)
        self.assertFalse(result["can_render"])
        self.assertIn("sum close to 100", result["reason"])

        payload["data"] = [{"date": "2024-01-01", "sector": "Technology", "weight_percent": 100.0}]
        payload["total_value"] = 100.0
        payload["category_field"] = "date"
        payload["time_series_requested"] = True
        result = validate_plot_payload(request, payload)
        self.assertFalse(result["can_render"])
        self.assertIn("time-series", result["reason"])

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

    def test_response_contract_exposes_planner_metadata(self):
        state = default_session_state("s1")
        resolved = self.resolve_full(
            "For U1 tickers, generate only the Ticker Concentration Plot as a horizontal bar chart. "
            "Use all U1 tickers. If current weights are missing, use equal-weight proxy for this test.",
            state,
        )
        contract = build_response_contract("plot ticker concentration", resolved)

        self.assertEqual(contract["planner"]["user_intent"], "ticker_concentration_plot")
        self.assertTrue(contract["planner"]["ready_to_render"])
        self.assertEqual(contract["plots"][0]["planner"]["plan_id"], contract["planner"]["plan_id"])

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
