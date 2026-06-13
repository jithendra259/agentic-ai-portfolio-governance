import os
import sys
import unittest


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.memory.context_resolver import ContextResolver
from src.memory.continuity import recover_state_from_chat_history
from src.memory.session_state import U1_TECH_TICKERS, default_session_state


class MemoryContinuityTests(unittest.TestCase):
    def test_recover_state_from_response_contract_metadata(self):
        state = default_session_state("session-1")
        messages = [
            {
                "role": "assistant",
                "content": "Rendered the U1 risk contribution plot.",
                "metadata": {
                    "response_contract": {
                        "resolved_context": {
                            "universe": "U1",
                            "tickers": U1_TECH_TICKERS,
                            "ticker_count": len(U1_TECH_TICKERS),
                            "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
                        }
                    }
                },
                "created_at": "2026-06-13T06:00:00+00:00",
            }
        ]

        recovered = recover_state_from_chat_history(state, messages)

        self.assertEqual(recovered["active_universe"], "U1")
        self.assertEqual(recovered["active_tickers"], U1_TECH_TICKERS)
        self.assertEqual(recovered["active_ticker_count"], len(U1_TECH_TICKERS))
        self.assertEqual(recovered["active_date_range"], {"start": "2024-01-01", "end": "2024-12-31"})
        self.assertGreaterEqual(len(recovered["continuity_memory"]["anchors"]), 3)

    def test_context_resolver_uses_recovered_history_for_follow_up_plan(self):
        history = [
            {
                "role": "assistant",
                "content": "I have U1 in context.",
                "metadata": {
                    "response_contract": {
                        "resolved_context": {
                            "universe": "U1",
                            "tickers": U1_TECH_TICKERS,
                            "ticker_count": len(U1_TECH_TICKERS),
                        }
                    }
                },
                "created_at": "2026-06-13T06:00:00+00:00",
            }
        ]

        resolved = ContextResolver().resolve(
            "now plot risk contribution",
            default_session_state("session-1"),
            chat_history_last_25=history,
        )

        self.assertEqual(resolved["session_state"]["active_universe"], "U1")
        self.assertEqual(resolved["session_state"]["active_tickers"], U1_TECH_TICKERS)
        self.assertEqual(resolved["session_state"]["current_plan"]["universe"], "U1")
        self.assertEqual(resolved["session_state"]["last_plot_id"], "plot_55_risk_contribution_by_ticker")
        self.assertEqual(
            resolved["resolved_context"]["continuity_memory"]["top"]["universe"],
            "U1",
        )


if __name__ == "__main__":
    unittest.main()
