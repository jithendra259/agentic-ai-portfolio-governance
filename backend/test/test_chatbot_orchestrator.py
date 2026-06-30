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

    def test_route_after_plot_tool_continues_when_computed_stats_requested(self):
        state = {
            "messages": [
                HumanMessage(content="Compare RELIANCE.NS and TCS.NS volatility, CAGR, and max drawdown."),
                ToolMessage(
                    content="Historical Price Plot\nPlot generated successfully.",
                    name="plot_historical_prices",
                    tool_call_id="plot-1",
                ),
            ]
        }

        route = self.module._route_after_tool(state)

        self.assertEqual(route, "chatbot")

    def test_route_after_plot_tool_finalizes_simple_price_chart(self):
        state = {
            "messages": [
                HumanMessage(content="Plot RELIANCE.NS and TCS.NS prices from 2023 to 2024."),
                ToolMessage(
                    content="Historical Price Plot\nPlot generated successfully.",
                    name="plot_historical_prices",
                    tool_call_id="plot-2",
                ),
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

        config = {"configurable": {"thread_id": "test"}}
        response = self.module.finalize_governance_node(state, config=config)
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

    def test_system_prompt_forbids_unregistered_attachment_chart_links(self):
        self.assertIn("Never output attachment://", self.module.SYSTEM_PROMPT)
        self.assertIn("plot://", self.module.SYSTEM_PROMPT)

    def test_assembled_prompt_includes_chatbot_conversation_markdown_guidance(self):
        prompt = self.module.assemble_system_prompt({"messages": [], "session_state": {}})

        self.assertIn("### CHATBOT CONVERSATION GUIDANCE ###", prompt)
        self.assertIn("Risk-profile follow-ups", prompt)
        self.assertIn("Do not list universes", prompt)

    def test_resolve_model_prefers_configured_ashna_when_ollama_has_no_models(self):
        model = self.module._resolve_ollama_model(
            ["ashnaai", "qwen3-coder-next:cloud", "qwen3:1.7b"],
            [],
        )

        self.assertEqual(model, "ashnaai")

    def test_unavailable_ollama_error_is_retryable(self):
        error = Exception("Failed to connect to Ollama. Is the Ollama server running?")

        self.assertTrue(self.module._is_ollama_unavailable_error(error))

    def test_user_visible_response_summarizes_scratchpad_only_json(self):
        raw = (
            '{"scratchpad_save": true, "metric_name": "annualised_volatility_BTC-USD", '
            '"exact_value": 0.402697, "context": "BTC-USD vs GLD 2023-01-01 to 2024-12-31"}'
        )

        content = self.module._sanitize_user_visible_response(raw)

        self.assertIn("Saved exact metric", content)
        self.assertIn("annualised_volatility_BTC-USD", content)
        self.assertIn("0.402697", content)
        self.assertNotIn("scratchpad_save", content)

    def test_user_visible_response_uses_scratchpad_metrics_when_json_leaks(self):
        raw = '{"scratchpad_save": true, "metric_name": "annualised_volatility_BTC-USD", "exact_value": 0.402697}'
        scratchpad = {
            "annualised_volatility_BTC-USD": {"exact_value": 0.402697, "context": "BTC-USD vs GLD"},
            "max_drawdown_GLD": {"exact_value": -11.3474, "context": "BTC-USD vs GLD"},
        }

        content = self.module._sanitize_user_visible_response(raw, scratchpad=scratchpad)

        self.assertIn("Exact metrics available", content)
        self.assertIn("annualised_volatility_BTC-USD", content)
        self.assertIn("max_drawdown_GLD", content)
        self.assertIn("-11.3474", content)
        self.assertNotIn("scratchpad_save", content)

    def test_user_visible_response_removes_unresolved_attachment_chart_links(self):
        raw = (
            "Here is the chart:\n\n"
            "![MSFT Candlestick Chart 2024](attachment://chart.png)\n\n"
            "Actual chart: ![MSFT](plot://plot-123)"
        )

        content = self.module._sanitize_user_visible_response(raw)

        self.assertNotIn("attachment://", content)
        self.assertIn("plot://plot-123", content)
        self.assertIn("Chart rendering is still pending", content)

    def test_loop_blocked_with_scratchpad_routes_to_final_response(self):
        state = {
            "route_status": "loop_blocked",
            "scratchpad": {
                "annualised_volatility_7203.T": {
                    "exact_value": 0.3229,
                    "context": "7203.T 2023-01-01 to 2024-12-31",
                }
            },
        }

        route = self.module._route_after_interceptor(state)

        self.assertEqual(route, "finalize_governance")

    def test_finalize_loop_blocked_uses_scratchpad_metrics(self):
        state = {
            "route_status": "loop_blocked",
            "scratchpad": {
                "annualised_volatility_7203.T": {
                    "exact_value": 0.3229,
                    "context": "7203.T 2023-01-01 to 2024-12-31",
                }
            },
            "messages": [
                HumanMessage(content="Analyze 7203.T Toyota"),
            ],
        }

        response = self.module.finalize_governance_node(state, config={"configurable": {"thread_id": "test"}})
        content = response["messages"][0].content

        self.assertIn("Exact metrics available", content)
        self.assertIn("annualised_volatility_7203.T", content)
        self.assertNotIn("CRITICAL SYSTEM OVERRIDE", content)

    @patch('src.orchestrator.chatbot_orchestrator.plot_us_economic_indicators.func')
    def test_classify_and_route_routes_recession_bands_deterministically(self, mock_plot_func):
        state = {
            "messages": [
                HumanMessage(content="show me recession bands and unemployment")
            ]
        }
        config = {"configurable": {"thread_id": "test_thread"}}
        response = self.module.classify_and_route_node(state, config=config)
        
        self.assertEqual(response["route_status"], "end")
        self.assertEqual(response["messages"][0].content, "Here is the US unemployment rate comparison with GDP per capita, including the shaded recession bands and dual Y-axes.")
        mock_plot_func.assert_called_once_with(config=config)


if __name__ == "__main__":
    unittest.main()
