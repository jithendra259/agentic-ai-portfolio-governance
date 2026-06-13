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

    def test_save_financial_metric_tool_records_exact_value(self):
        blackboard = BlackboardState("req-1", "Compute beta")
        executor = object.__new__(AgenticTaskExecutor)

        observation = executor._save_financial_metric(
            {
                "tool": "save_financial_metric",
                "parameters": {
                    "metric_name": "tech_portfolio_beta",
                    "exact_value": "1.247",
                    "context": "CAPM beta from covariance / benchmark variance",
                    "formula": "cov(portfolio, benchmark) / var(benchmark)",
                },
            },
            blackboard,
        )

        self.assertEqual(observation["status"], "success")
        self.assertEqual(observation["metric_name"], "tech_portfolio_beta")
        self.assertEqual(
            blackboard.calculation_scratchpad.get("calc_001")["result"],
            "1.247",
        )

    def test_generate_thought_injects_exact_scratchpad_rules(self):
        blackboard = BlackboardState("req-1", "Compute required return")
        blackboard.record_calculation(
            label="tech_portfolio_beta",
            formula="cov(portfolio, benchmark) / var(benchmark)",
            inputs={"portfolio": "TECH_U1", "benchmark": "SPY"},
            result="1.247",
            source="save_financial_metric",
        )
        task = type("Task", (), {"name": "required_return"})()
        executor = object.__new__(AgenticTaskExecutor)

        thought = executor._generate_thought(task, blackboard)

        self.assertIn("tech_portfolio_beta", thought)
        self.assertIn("1.247", thought)
        self.assertIn("Before calculating any metric, check the Scratchpad", thought)


if __name__ == "__main__":
    unittest.main()
