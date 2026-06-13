import unittest

from src.agents.agentic_task_executor import AgenticTaskExecutor, BlackboardState
from src.memory.calculation_scratchpad import CalculationScratchpad


class CalculationScratchpadTests(unittest.TestCase):
    def test_records_exact_calculation_and_compact_summary(self):
        scratchpad = CalculationScratchpad()

        record = scratchpad.record(
            label="portfolio_cvar_95",
            formula="mean(losses[losses >= var_95])",
            inputs={"var_95": "0.03125", "tail_count": 12},
            result="0.0448123456789",
            source="risk_engine",
        )

        self.assertEqual(record["calculation_id"], "calc_001")
        self.assertEqual(scratchpad.get("calc_001")["result"], "0.0448123456789")
        self.assertEqual(
            scratchpad.compact_summary(),
            [
                {
                    "calculation_id": "calc_001",
                    "label": "portfolio_cvar_95",
                    "formula": "mean(losses[losses >= var_95])",
                    "result": "0.0448123456789",
                    "source": "risk_engine",
                }
            ],
        )

    def test_blackboard_round_trip_preserves_scratchpad(self):
        blackboard = BlackboardState("req-1", "Compute portfolio risk")
        blackboard.record_calculation(
            label="annualized_volatility",
            formula="std(daily_returns) * sqrt(252)",
            inputs={"std": "0.0123"},
            result="0.195255",
            source="quant_agent",
        )

        restored = AgenticTaskExecutor._dict_to_blackboard(
            object.__new__(AgenticTaskExecutor),
            blackboard.to_dict(),
        )

        self.assertEqual(restored.calculation_scratchpad.get("calc_001")["result"], "0.195255")
        self.assertEqual(
            restored.get_summary_context()["calculation_scratchpad"][0]["label"],
            "annualized_volatility",
        )


if __name__ == "__main__":
    unittest.main()
