import os
import sys
import unittest


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.memory.conversation_memory import (
    conversation_prompt_block,
    update_for_assistant_response,
    update_for_user_message,
)
from src.memory.session_state import default_session_state


class ConversationMemoryTests(unittest.TestCase):
    def test_user_update_tracks_strategy_tickers_and_resolves_pronoun(self):
        state = default_session_state("thread-1")

        state = update_for_user_message(state, "Explain Adaptive G-CVaR for RELIANCE.NS, TCS.NS, INFY.NS")
        state = update_for_assistant_response(state, "Adaptive G-CVaR is a graph-aware CVaR method.")
        state = update_for_user_message(state, "Compare it with Standard CVaR")

        conversation = state["conversation_state"]
        self.assertEqual(conversation["current_strategy"], "Adaptive G-CVaR")
        self.assertEqual(conversation["comparison_target"], "Standard CVaR")
        self.assertEqual(conversation["selected_tickers"], ["RELIANCE.NS", "TCS.NS", "INFY.NS"])
        self.assertIn("Reference resolution", conversation["resolved_question"])
        self.assertIn("Selected tickers are RELIANCE.NS, TCS.NS, INFY.NS", conversation["summary"])

    def test_prompt_block_contains_compact_thread_memory(self):
        state = default_session_state("thread-1")
        state = update_for_user_message(state, "Plot RELIANCE.NS and TCS.NS")

        block = conversation_prompt_block(state)

        self.assertIn("### CONVERSATION MEMORY ###", block)
        self.assertIn("Selected tickers: RELIANCE.NS, TCS.NS", block)
        self.assertIn("resolve words like it", block)

    def test_vague_values_followup_reuses_active_metrics_and_tickers(self):
        state = default_session_state("thread-1")
        state = update_for_user_message(state, "RELIANCE.NS, INFY.NS, TCS.NS")
        state = update_for_user_message(state, "compare their revenue / net income")
        state = update_for_assistant_response(state, "Revenue: RELIANCE > TCS > INFY. Net income: RELIANCE > TCS > INFY.")
        state = update_for_user_message(state, "give me the values comparison")

        conversation = state["conversation_state"]
        self.assertEqual(conversation["selected_tickers"], ["RELIANCE.NS", "INFY.NS", "TCS.NS"])
        self.assertEqual(conversation["current_metrics"], ["revenue", "net income"])
        self.assertIn("Use active metrics: revenue, net income", conversation["resolved_question"])
        self.assertIn("Use selected tickers: RELIANCE.NS, INFY.NS, TCS.NS", conversation["resolved_question"])


if __name__ == "__main__":
    unittest.main()
