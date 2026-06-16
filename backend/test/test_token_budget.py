import json
import unittest

import numpy as np

from src.agents.token_budget import (
    cap_tool_output,
    check_token_budget,
    estimate_tokens,
    update_token_ledger,
)


class TokenBudgetTests(unittest.TestCase):
    def test_update_token_ledger_prefers_provider_usage_metadata(self):
        class Response:
            usage_metadata = {"input_tokens": 12, "output_tokens": 8}

        update = update_token_ledger({"total_tokens_used": 5}, response=Response())

        self.assertEqual(update["total_tokens_used"], 25)
        self.assertEqual(update["remaining_token_budget"], 99975)
        self.assertGreater(update["estimated_cost_usd"], 0)

    def test_budget_check_blocks_projected_context_over_limit(self):
        check = check_token_budget(
            {"total_tokens_used": 90, "max_token_budget": 100},
            "x" * 80,
        )

        self.assertFalse(check.allowed)
        self.assertEqual(check.remaining_token_budget, 10)
        self.assertIn("exceeded", check.reason)

    def test_cap_tool_output_returns_json_safe_summary_for_large_objects(self):
        output = {"rows": [{"value": np.float64(i)} for i in range(50)]}

        capped = cap_tool_output(output, max_chars=80)

        json.dumps(capped)
        self.assertEqual(capped["status"], "truncated")
        self.assertGreater(capped["truncated_chars"], 0)

    def test_estimate_tokens_never_returns_zero_for_nonempty_values(self):
        self.assertEqual(estimate_tokens("a"), 1)


if __name__ == "__main__":
    unittest.main()
