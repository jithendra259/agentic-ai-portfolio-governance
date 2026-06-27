import os
import sys
import unittest
from unittest.mock import patch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import ChatRequest, _assistant_config, _resolve_previous_weights
import src.orchestrator.chatbot_orchestrator as orchestrator


class PreviousWeightResolutionTests(unittest.TestCase):
    def test_normalizes_valid_current_weights(self):
        resolved = {
            "session_state": {
                "active_weights": {
                    "type": "user_supplied",
                    "weights": {"aapl": 60, "MSFT": 40},
                    "approved_by_user": True,
                }
            }
        }

        self.assertEqual(_resolve_previous_weights(resolved), {"AAPL": 0.6, "MSFT": 0.4})

    def test_rejects_unapproved_equal_weight_proxy(self):
        resolved = {
            "session_state": {
                "active_weights": {
                    "type": "equal_weight_proxy",
                    "weights": {"AAPL": 0.5, "MSFT": 0.5},
                    "approved_by_user": False,
                }
            }
        }

        self.assertEqual(_resolve_previous_weights(resolved), {})

    def test_rejects_negative_nonfinite_or_empty_weights(self):
        for weights in (
            {"AAPL": -0.1, "MSFT": 1.1},
            {"AAPL": float("nan")},
            {},
        ):
            with self.subTest(weights=weights):
                resolved = {"session_state": {"active_weights": {"weights": weights}}}
                self.assertEqual(_resolve_previous_weights(resolved), {})

    def test_assistant_config_includes_resolved_previous_weights(self):
        request = ChatRequest(session_id="session-1", user_message="optimize", model="test-model")
        resolved = {
            "session_state": {
                "active_weights": {
                    "type": "user_supplied",
                    "weights": {"AAPL": 3, "MSFT": 2},
                    "approved_by_user": True,
                }
            }
        }

        config = _assistant_config(request, resolved)

        self.assertEqual(config["configurable"]["thread_id"], "session-1")
        self.assertEqual(config["configurable"]["override_model"], "test-model")
        self.assertEqual(config["configurable"]["previous_weights"], {"AAPL": 0.6, "MSFT": 0.4})


class GovernanceCacheContractTests(unittest.TestCase):
    def test_configured_weights_bypass_cache_and_are_forwarded(self):
        with (
            patch.object(orchestrator, "memory_manager") as memory,
            patch.object(orchestrator, "run_full_governance_pipeline") as pipeline,
        ):
            pipeline.invoke.return_value = '{"status":"success"}'

            orchestrator.governance_pipeline_with_cache.func(
                tickers=["AAPL", "MSFT"],
                target_date="2026-06-20",
                config={"configurable": {"previous_weights": {"AAPL": 0.6, "MSFT": 0.4}}},
            )

        memory.retrieve_cached_plan.assert_not_called()
        memory.cache_governance_plan.assert_not_called()
        self.assertEqual(
            pipeline.invoke.call_args.args[0]["previous_weights"],
            {"AAPL": 0.6, "MSFT": 0.4},
        )

    def test_explicit_weights_override_configured_weights(self):
        with (
            patch.object(orchestrator, "memory_manager") as memory,
            patch.object(orchestrator, "run_full_governance_pipeline") as pipeline,
        ):
            pipeline.invoke.return_value = '{"status":"success"}'

            orchestrator.governance_pipeline_with_cache.func(
                tickers=["AAPL", "MSFT"],
                target_date="2026-06-20",
                previous_weights={"AAPL": 0.5, "MSFT": 0.5},
                config={"configurable": {"previous_weights": {"AAPL": 0.6, "MSFT": 0.4}}},
            )

        self.assertEqual(
            pipeline.invoke.call_args.args[0]["previous_weights"],
            {"AAPL": 0.5, "MSFT": 0.5},
        )
        memory.retrieve_cached_plan.assert_not_called()
        memory.cache_governance_plan.assert_not_called()

    def test_unweighted_request_uses_versioned_cache_key(self):
        with (
            patch.object(orchestrator, "memory_manager") as memory,
            patch.object(orchestrator, "run_full_governance_pipeline") as pipeline,
        ):
            memory.compute_query_hash.return_value = "hash-v2"
            memory.retrieve_cached_plan.return_value = None
            pipeline.invoke.return_value = '{"status":"success"}'

            orchestrator.governance_pipeline_with_cache.func(
                tickers=["AAPL", "MSFT"],
                target_date="2026-06-20",
            )

        self.assertIn(
            orchestrator.GOVERNANCE_CACHE_VERSION,
            memory.compute_query_hash.call_args.kwargs["risk_tolerance"],
        )
        memory.retrieve_cached_plan.assert_called_once_with("hash-v2")
        memory.cache_governance_plan.assert_called_once()


if __name__ == "__main__":
    unittest.main()
