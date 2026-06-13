import unittest

from src.agents.agentic_task_executor import (
    AgenticTaskExecutor,
    BlackboardState,
    SafetyGuardrails,
    TaskNode,
    TaskStatus,
)
from src.guardrails.loop_detector import SemanticLoopDetector


class LoopingExecutor(AgenticTaskExecutor):
    def __init__(self):
        self.allowed_tools = {"web_search"}
        self.guardrails = SafetyGuardrails()
        self.semantic_loop_detector = SemanticLoopDetector(similarity_threshold=0.82)
        self.execute_count = 0
        self._actions = iter(
            [
                {"tool": "web_search", "parameters": {"query": "AAPL stock price today"}},
                {"tool": "web_search", "parameters": {"query": "Apple Inc current share value"}},
                {"tool": "web_search", "parameters": {"query": "What is the price of Apple stock right now?"}},
            ]
        )

    def _generate_thought(self, task, blackboard):
        return "search for stock price"

    def _decide_action(self, task, blackboard, thought):
        return next(self._actions)

    async def _execute_action(self, action, task, blackboard):
        self.execute_count += 1
        return {"status": "retry", "data": action["parameters"]["query"]}

    def _is_task_complete(self, task, observation):
        return False


class AgenticSemanticLoopIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_executor_blocks_third_semantically_repeated_tool_call(self):
        blackboard = BlackboardState("req-1", "Find Apple stock price")
        task = TaskNode(
            "task-1",
            "find_price",
            "Find Apple stock price",
            status=TaskStatus.IN_PROGRESS,
            tool_name="web_search",
        )
        blackboard.add_task(task)
        blackboard.active_task_id = "task-1"

        executor = LoopingExecutor()
        state = {"blackboard": blackboard.to_dict()}

        result = await executor._executor_node(state)

        final_blackboard = result["blackboard"]
        loop_steps = [
            step
            for step in final_blackboard["react_history"]
            if isinstance(step.get("observation"), dict)
            and step["observation"].get("status") == "loop_detected"
        ]

        self.assertEqual(executor.execute_count, 2)
        self.assertIn("task-1", final_blackboard["failed_tasks"])
        self.assertEqual(loop_steps[-1]["observation"]["tool"], "web_search")


if __name__ == "__main__":
    unittest.main()
