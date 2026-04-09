import importlib
import os
import sys
import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage, ToolMessage


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class ChatbotOrchestratorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(
            os.environ,
            {
                "MONGO_URI": "",
                "PORTFOLIO_OLLAMA_MODEL": "mistral:latest",
                "PORTFOLIO_OLLAMA_FALLBACK_MODEL": "mistral:latest",
            },
            clear=False,
        ):
            cls.module = importlib.import_module("src.orchestrator.chatbot_orchestrator")

    def test_route_after_tool_loops_back_to_chatbot_for_universe_lookup(self):
        state = {
            "messages": [
                ToolMessage(
                    content="Here are the stocks in universe U1 found in the database:\nAAPL: Apple Inc.",
                    name="get_stocks_by_universe",
                    tool_call_id="lookup-1",
                )
            ]
        }

        route = self.module._route_after_tool(state)

        self.assertEqual(route, "chatbot")

    def test_route_after_tool_finalizes_snapshot_output(self):
        state = {
            "messages": [
                ToolMessage(
                    content="Ticker: BAC\n- Company: Bank of America Corporation",
                    name="get_stock_database_snapshot",
                    tool_call_id="snapshot-1",
                )
            ]
        }

        route = self.module._route_after_tool(state)

        self.assertEqual(route, "finalize_governance")

    def test_finalize_governance_formats_snapshot_explanations_from_tool_path(self):
        snapshot_text = (
            "MongoDB Stock Snapshot\n\n"
            "Ticker: BAC\n"
            "- Company: Bank of America Corporation\n"
            "- Universes: U2\n"
            "- Sector: Financial Services\n"
            "- Industry: Banks - Diversified\n"
            "- Country: United States\n"
            "- Historical price coverage: 2005-01-03 to 2025-12-30\n"
            "- Most recent stored close: 54.97 on 2025-12-30\n"
            "- Key stats:\n"
            "  - trailing_pe: 12.328085\n"
            "  - forward_pe: 9.473024\n"
            "  - profit_margin: 0.28401\n"
            "  - return_on_equity: 0.10217\n"
            "  - dividend_yield: 2.38\n"
            "  - beta: 1.263\n"
            "- Business summary: Bank of America Corporation provides financial products and services.\n"
        )
        state = {
            "messages": [
                HumanMessage(content="Describe the company for BAC"),
                ToolMessage(
                    content=snapshot_text,
                    name="get_stock_database_snapshot",
                    tool_call_id="snapshot-2",
                ),
            ]
        }

        response = self.module.finalize_governance_node(state)
        content = response["messages"][0].content

        self.assertIn("Bank of America Corporation (BAC) is", content)
        self.assertNotIn("MongoDB Stock Snapshot", content)

    def test_tools_include_price_series_analysis_tool(self):
        tool_names = {tool.name for tool in self.module.tools}
        self.assertIn("get_price_series_for_analysis", tool_names)

    def test_system_prompt_includes_two_step_statistical_analysis_rules(self):
        self.assertIn("get_price_series_for_analysis", self.module.SYSTEM_PROMPT)
        self.assertIn("correlation heatmap", self.module.SYSTEM_PROMPT)
        self.assertIn("Never tell the user you cannot do this analysis", self.module.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
