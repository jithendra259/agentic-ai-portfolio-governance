import json
import unittest

import pandas as pd

from src.orchestrator.langgraph_dag import SupervisoryOrchestrator


class FakeAgent1:
    def execute(self, universe_id, target_date_str):
        return {
            "returns_df": pd.DataFrame([{"AAPL": 0.01, "MSFT": 0.02}]),
            "covariance_matrix": pd.DataFrame([{"AAPL": 0.03, "MSFT": 0.01}]),
            "instability_index": 0.42,
            "raw_instability_index": 0.40,
            "retained_assets": ["AAPL", "MSFT"],
            "dropped_assets": [],
            "mean_volatility": 0.2,
            "mean_correlation": 0.3,
            "mean_drawdown": -0.1,
        }


class LangGraphDagSanitizationTests(unittest.TestCase):
    def test_agent_1_return_state_is_json_safe(self):
        orchestrator = object.__new__(SupervisoryOrchestrator)
        orchestrator.agent1 = FakeAgent1()

        state = orchestrator.run_agent_1({"universe_id": "U1", "target_date": "2025-12-31"})

        json.dumps(state)
        self.assertEqual(state["returns_df"][0]["AAPL"], 0.01)
        self.assertEqual(state["covariance_matrix"][0]["MSFT"], 0.01)


if __name__ == "__main__":
    unittest.main()
