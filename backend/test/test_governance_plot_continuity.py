import json
import os
import sys
import unittest


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.memory.governance_plot_continuity import (
    build_systemic_risk_weight_plot,
    is_short_followup,
    resolve_governance_plot_followup,
    store_latest_governance_run,
)


class GovernancePlotContinuityTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "session_id": "healthcare-session",
            "active_universe": "Healthcare",
            "active_tickers": [],
            "latest_governance_run": None,
            "pending_plot_request": None,
        }
        self.payload = {
            "status": "success",
            "target_date": "2026-06-20",
            "valid_tickers": ["JNJ", "PFE", "UNH"],
            "systemic_risk": {
                "method": "institutional_overlap",
                "scores": {"JNJ": 0.31, "PFE": 0.47, "UNH": 0.22},
            },
            "optimization": {
                "weights": {"JNJ": 0.55, "PFE": 0.0, "UNH": 0.45},
                "expected_annualized_return": 0.1234,
                "expected_cvar_95": 0.0275,
                "instability_index": 0.18,
                "lambda_t": 0.7,
                "hhi": 0.505,
                "effective_number_of_holdings": 1.98,
            },
        }

    def test_successful_governance_payload_is_committed_to_session_state(self):
        state = store_latest_governance_run(self.state, json.dumps(self.payload))

        self.assertEqual(state["active_tickers"], ["JNJ", "PFE", "UNH"])
        self.assertEqual(state["latest_governance_run"]["universe"], "Healthcare")
        self.assertEqual(state["latest_governance_run"]["weights"]["JNJ"], 0.55)
        self.assertEqual(state["latest_governance_run"]["systemic_risk_scores"]["PFE"], 0.47)
        self.assertEqual(state["latest_governance_run"]["regime"], "calm")
        self.assertEqual(state["latest_governance_run"]["graph_penalty"], 0.7)
        self.assertEqual(state["latest_governance_run"]["annualized_return"], 0.1234)
        self.assertEqual(state["latest_governance_run"]["expected_cvar"], 0.0275)
        self.assertEqual(state["latest_governance_run"]["hhi"], 0.505)
        self.assertEqual(state["latest_governance_run"]["effective_holdings"], 1.98)

    def test_axis_message_sets_pending_request_and_builds_expected_scatter(self):
        state = store_latest_governance_run(self.state, self.payload)

        resolution = resolve_governance_plot_followup(
            state, "systemic risk score vs portfolio weight"
        )

        self.assertEqual(
            resolution["state"]["pending_plot_request"],
            {
                "chart_type": "scatter",
                "x": "systemic_risk_score",
                "y": "portfolio_weight",
                "source": "latest_governance_run",
                "label": "ticker",
            },
        )
        plot = resolution["plot_spec"]
        self.assertEqual(plot["title"], "Systemic Risk Score vs Portfolio Weight")
        self.assertEqual(plot["x_label"], "Systemic Risk Score")
        self.assertEqual(plot["y_label"], "Portfolio Weight (%)")
        points = plot["series"][0]["data"]
        self.assertEqual([point["id"] for point in points], ["JNJ", "PFE", "UNH"])
        self.assertEqual(points[1]["y"], 0.0)
        self.assertNotIn("Beta", json.dumps(plot))
        self.assertNotIn("Forward P/E", json.dumps(plot))

    def test_short_followup_executes_existing_pending_governance_plot(self):
        state = store_latest_governance_run(self.state, self.payload)
        state["pending_plot_request"] = {
            "chart_type": "scatter",
            "x": "systemic_risk_score",
            "y": "portfolio_weight",
            "source": "latest_governance_run",
            "label": "ticker",
        }

        resolution = resolve_governance_plot_followup(state, "plot the scatter plot")

        self.assertEqual(resolution["action"], "plot")
        self.assertEqual(resolution["plot_spec"]["data_source"], "latest_governance_run")

    def test_generic_scatter_request_with_governance_state_asks_only_for_axes(self):
        state = store_latest_governance_run(self.state, self.payload)

        resolution = resolve_governance_plot_followup(state, "plot scatter plot")

        self.assertEqual(resolution["action"], "clarify_axes")
        self.assertIn("which governance metrics", resolution["response"].lower())

    def test_no_governance_state_does_not_capture_generic_scatter_request(self):
        self.assertIsNone(resolve_governance_plot_followup(self.state, "plot scatter plot"))

    def test_short_followup_vocabulary(self):
        for message in ("plot it", "show it", "do it", "use that", "same", "plot that"):
            with self.subTest(message=message):
                self.assertTrue(is_short_followup(message))
        self.assertFalse(is_short_followup("plot beta vs forward P/E for JNJ"))

    def test_plot_uses_only_tickers_with_risk_and_weight_values(self):
        run = {
            "systemic_risk_scores": {"JNJ": 0.31, "PFE": 0.47, "MRK": 0.4},
            "weights": {"JNJ": 0.55, "PFE": None, "UNH": 0.45},
            "target_date": "2026-06-20",
        }

        plot = build_systemic_risk_weight_plot(run)

        self.assertEqual([point["id"] for point in plot["series"][0]["data"]], ["JNJ"])


if __name__ == "__main__":
    unittest.main()
