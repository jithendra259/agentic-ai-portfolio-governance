import unittest

from src.agents.state_reducers import merge_blackboard_state, merge_dict_state, merge_unique_values
from src.agents.agentic_task_executor import AgenticTaskExecutor, BlackboardState, ReActStep, ActionType


class AgenticStateReducerTests(unittest.TestCase):
    def test_merge_unique_values_appends_without_duplicates(self):
        merged = merge_unique_values(["executor"], ["critic", "executor"])

        self.assertEqual(merged, ["executor", "critic"])

    def test_merge_dict_state_deep_merges_nested_values(self):
        existing = {
            "scratchpad": {"beta": {"value": "1.247", "source": "executor"}},
            "status": "analyzed",
        }
        update = {
            "scratchpad": {"sharpe": {"value": "1.83", "source": "critic"}},
            "status": "verified",
        }

        merged = merge_dict_state(existing, update)

        self.assertEqual(merged["scratchpad"]["beta"]["value"], "1.247")
        self.assertEqual(merged["scratchpad"]["sharpe"]["value"], "1.83")
        self.assertEqual(merged["status"], "verified")

    def test_merge_blackboard_state_preserves_parallel_agent_updates(self):
        existing = {
            "tasks": {"t1": {"task_id": "t1", "status": "in_progress"}},
            "completed_tasks": ["t0"],
            "react_history": [{"step_number": 1, "thought": "executor calculated beta"}],
            "calculation_scratchpad": [{"calculation_id": "calc_001", "label": "beta", "result": "1.247"}],
        }
        update = {
            "tasks": {"t2": {"task_id": "t2", "status": "pending"}},
            "completed_tasks": ["t1"],
            "react_history": [{"step_number": 2, "thought": "critic verified beta"}],
            "calculation_scratchpad": [{"calculation_id": "calc_002", "label": "sharpe", "result": "1.83"}],
        }

        merged = merge_blackboard_state(existing, update)

        self.assertEqual(set(merged["tasks"]), {"t1", "t2"})
        self.assertEqual(merged["completed_tasks"], ["t0", "t1"])
        self.assertEqual(len(merged["react_history"]), 2)
        self.assertEqual(len(merged["calculation_scratchpad"]), 2)

    def test_blackboard_delta_only_returns_new_append_only_records(self):
        original = BlackboardState("req-1", "Analyze AAPL")
        original.add_react_step(ReActStep(1, ActionType.THINK, "first thought"))
        original_dict = original.to_dict()

        updated = AgenticTaskExecutor._dict_to_blackboard(
            object.__new__(AgenticTaskExecutor),
            original_dict,
        )
        updated.add_react_step(ReActStep(2, ActionType.THINK, "second thought"))

        delta = AgenticTaskExecutor._blackboard_delta(
            object.__new__(AgenticTaskExecutor),
            original_dict,
            updated,
        )

        self.assertEqual(len(delta["react_history"]), 1)
        self.assertEqual(delta["react_history"][0]["thought"], "second thought")


if __name__ == "__main__":
    unittest.main()
