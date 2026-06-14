import unittest

from src.agents.htn_plan_parser import (
    PlanValidationError,
    build_sequential_fallback_plan,
    extract_task_descriptions,
    parse_execution_plan,
    validate_execution_plan,
)


class HTNPlanParserTests(unittest.TestCase):
    def test_parse_markdown_wrapped_json_plan(self):
        raw = """```json
        {
          "goal": "Analyze tech portfolio",
          "nodes": [
            {"task_id": "task_1", "description": "Fetch prices", "dependencies": []},
            {"task_id": "task_2", "description": "Compute beta", "dependencies": ["task_1"]}
          ]
        }
        ```"""

        plan = parse_execution_plan(raw)

        self.assertEqual(plan.goal, "Analyze tech portfolio")
        self.assertEqual(plan.nodes[1].dependencies, ["task_1"])

    def test_parse_json_with_trailing_commas(self):
        raw = """
        {
          "goal": "Analyze risk",
          "nodes": [
            {"task_id": "task_1", "description": "Fetch returns", "dependencies": [],},
          ],
        }
        """

        plan = parse_execution_plan(raw)

        self.assertEqual(plan.nodes[0].task_id, "task_1")

    def test_validate_execution_plan_rejects_cycles(self):
        plan = parse_execution_plan(
            {
                "goal": "Bad plan",
                "nodes": [
                    {"task_id": "task_1", "description": "A", "dependencies": ["task_2"]},
                    {"task_id": "task_2", "description": "B", "dependencies": ["task_1"]},
                ],
            }
        )

        with self.assertRaisesRegex(PlanValidationError, "Circular dependency"):
            validate_execution_plan(plan)

    def test_build_sequential_fallback_plan(self):
        plan = build_sequential_fallback_plan(
            "Analyze AAPL portfolio",
            ["Fetch historical prices", "Compute risk metrics", "Summarize recommendation"],
        )

        self.assertEqual([node.task_id for node in plan.nodes], ["task_1", "task_2", "task_3"])
        self.assertEqual(plan.nodes[1].dependencies, ["task_1"])
        self.assertEqual(plan.nodes[2].dependencies, ["task_2"])

    def test_extract_task_descriptions_from_invalid_plan_payload(self):
        descriptions = extract_task_descriptions(
            {
                "goal": "Malformed plan",
                "nodes": [
                    {"task_id": "task_1", "description": "Fetch MongoDB prices"},
                    {"task_id": "task_2", "description": "Compute portfolio beta"},
                ],
            }
        )

        self.assertEqual(descriptions, ["Fetch MongoDB prices", "Compute portfolio beta"])


if __name__ == "__main__":
    unittest.main()
