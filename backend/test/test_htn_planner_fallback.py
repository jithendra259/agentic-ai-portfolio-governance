import unittest

from src.agents.agentic_task_executor import HTNPlanner


class SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("no responses left")
        return self.responses.pop(0)


class HTNPlannerFallbackTests(unittest.TestCase):
    def test_llm_planner_retries_after_invalid_dag(self):
        llm = SequenceLLM(
            [
                {
                    "goal": "Analyze custom portfolio",
                    "nodes": [
                        {"task_id": "task_1", "description": "A", "dependencies": ["task_2"]},
                        {"task_id": "task_2", "description": "B", "dependencies": ["task_1"]},
                    ],
                },
                {
                    "goal": "Analyze custom portfolio",
                    "nodes": [
                        {"task_id": "task_1", "description": "Fetch data", "dependencies": []},
                        {"task_id": "task_2", "description": "Compute metrics", "dependencies": ["task_1"]},
                    ],
                },
            ]
        )
        planner = HTNPlanner(llm_client=llm)

        tasks = planner.decompose_goal("Analyze custom portfolio with private constraints")

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[1].depends_on, [tasks[0].task_id])
        self.assertIn("previous plan failed validation", llm.prompts[1])

    def test_llm_planner_degrades_to_sequential_plan_after_retries(self):
        llm = SequenceLLM(
            [
                "not json",
                {
                    "goal": "Bad",
                    "nodes": [
                        {
                            "task_id": "task_1",
                            "description": "Fetch local MongoDB portfolio prices",
                            "dependencies": ["missing_task"],
                        },
                        {
                            "task_id": "task_2",
                            "description": "Compute governance risk metrics",
                            "dependencies": ["task_1"],
                        },
                    ],
                },
            ]
        )
        planner = HTNPlanner(llm_client=llm)

        tasks = planner.decompose_goal("Handle a custom governance request")

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].depends_on, [])
        for index in range(1, len(tasks)):
            self.assertEqual(tasks[index].depends_on, [tasks[index - 1].task_id])
        self.assertTrue(all(task.metadata.get("decomposition_method") == "sequential_fallback" for task in tasks))
        self.assertEqual(tasks[0].description, "Fetch local MongoDB portfolio prices")
        self.assertEqual(tasks[1].description, "Compute governance risk metrics")


if __name__ == "__main__":
    unittest.main()
