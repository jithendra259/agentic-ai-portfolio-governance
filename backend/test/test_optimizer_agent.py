import os
import sys
import unittest

import numpy as np
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.optimizer_a3 import GCVaROptimizerAgent


class GCVaROptimizerAgentTests(unittest.TestCase):
    def test_notebook_aligned_outputs_include_baselines_and_hitl_flags(self):
        returns_df = pd.DataFrame(
            {
                "AAA": [0.010, 0.011, 0.008, 0.009, 0.010, 0.012],
                "BBB": [0.006, 0.005, 0.004, 0.006, 0.005, 0.004],
                "CCC": [0.002, 0.003, 0.001, 0.002, 0.003, 0.002],
            }
        )
        c_vector = pd.Series({"AAA": 0.9, "BBB": 0.4, "CCC": 0.1}, dtype=float)

        agent = GCVaROptimizerAgent()
        result = agent.execute(returns_df=returns_df, c_vector=c_vector, I_t=0.90)

        self.assertIn("optimal_weights", result)
        self.assertIn("strategy_weights", result)
        self.assertIn("g_cvar", result["strategy_weights"])
        self.assertIn("std_cvar", result["strategy_weights"])
        self.assertIn("mean_variance", result["strategy_weights"])
        self.assertIn("equal_weight", result["strategy_weights"])
        self.assertTrue(np.isclose(result["optimal_weights"].sum(), 1.0))
        self.assertTrue(np.isclose(result["strategy_weights"]["equal_weight"].sum(), 1.0))
        self.assertGreaterEqual(result["lambda_t"], 0.0)
        self.assertTrue(result["hitl_required"])
        self.assertTrue(result["hitl_crisis"])
        self.assertFalse(result["hitl_turnover"])
        self.assertTrue(result["hitl_reasons"])
        self.assertIsInstance(result["solver_status"], str)
        self.assertGreaterEqual(result["solve_time_s"], 0.0)
        self.assertGreaterEqual(result["max_weight_constraint"], 1.0 / 3.0)

    def test_turnover_trigger_fires_when_previous_weights_are_far_away(self):
        returns_df = pd.DataFrame(
            {
                "AAA": [0.010, 0.011, 0.008, 0.009, 0.010, 0.012],
                "BBB": [0.006, 0.005, 0.004, 0.006, 0.005, 0.004],
                "CCC": [0.002, 0.003, 0.001, 0.002, 0.003, 0.002],
            }
        )
        c_vector = pd.Series({"AAA": 0.9, "BBB": 0.4, "CCC": 0.1}, dtype=float)
        previous = pd.Series({"AAA": 1.0, "BBB": 0.0, "CCC": 0.0}, dtype=float)

        agent = GCVaROptimizerAgent(tau_crisis=0.95, tau_turnover=0.10)
        result = agent.execute(
            returns_df=returns_df,
            c_vector=c_vector,
            I_t=0.60,
            previous_weights=previous,
        )

        self.assertTrue(result["turnover"]["g_cvar"] > 0.10)
        self.assertTrue(result["hitl_required"])
        self.assertFalse(result["hitl_crisis"])
        self.assertTrue(result["hitl_turnover"])


if __name__ == "__main__":
    unittest.main()
