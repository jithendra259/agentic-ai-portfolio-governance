from collections import deque
import unittest

from src.guardrails.loop_detector import SemanticLoopDetector


class SemanticLoopDetectorTests(unittest.TestCase):
    def test_semantic_loop_detection_for_rephrased_stock_queries(self):
        detector = SemanticLoopDetector(similarity_threshold=0.82)
        history = deque(
            [
                {"action": {"tool": "web_search", "parameters": {"query": "AAPL stock price today"}}},
                {"action": {"tool": "web_search", "parameters": {"query": "Apple Inc current share value"}}},
            ],
            maxlen=100,
        )

        result = detector.detect_loop(
            {"tool": "web_search", "parameters": {"query": "What is the price of Apple stock right now?"}},
            history,
        )

        self.assertTrue(result.detected)
        self.assertEqual(result.tool, "web_search")
        self.assertGreaterEqual(result.max_similarity, 0.82)

    def test_semantic_loop_detection_ignores_different_tools(self):
        detector = SemanticLoopDetector(similarity_threshold=0.82)
        history = deque(
            [
                {"action": {"tool": "web_search", "parameters": {"query": "AAPL stock price today"}}},
                {"action": {"tool": "web_search", "parameters": {"query": "Apple Inc current share value"}}},
            ],
            maxlen=100,
        )

        result = detector.detect_loop(
            {"tool": "code_executor", "parameters": {"code": "print('hello')"}},
            history,
        )

        self.assertFalse(result.detected)


if __name__ == "__main__":
    unittest.main()
