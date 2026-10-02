import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from langchain_core.messages import AIMessage, HumanMessage
from src.agents.specialized_domain_agents import (
    classify_agent_for_query,
    SPECIALIZED_AGENTS_METADATA,
)
from src.orchestrator.chatbot_orchestrator import (
    _build_graph,
    supervisor_router,
    market_data_agent_tools,
    econometrics_agent_tools,
    risk_volatility_agent_tools,
    portfolio_optimizer_agent_tools,
    network_graph_rag_agent_tools,
    technical_analysis_agent_tools,
    governance_audit_agent_tools,
)


class TestSpecializedDomainAgents(unittest.TestCase):
    """Test suite verifying one-by-one specialized domain agents and LangGraph orchestration."""

    def test_all_seven_domain_agents_registered_in_metadata(self):
        expected_agents = {
            "market_data_agent",
            "econometrics_agent",
            "risk_volatility_agent",
            "portfolio_optimizer_agent",
            "network_graph_rag_agent",
            "technical_analysis_agent",
            "governance_audit_agent",
        }
        self.assertTrue(expected_agents.issubset(set(SPECIALIZED_AGENTS_METADATA.keys())))
        for agent_key in expected_agents:
            meta = SPECIALIZED_AGENTS_METADATA[agent_key]
            self.assertIn("name", meta)
            self.assertIn("role", meta)
            self.assertIn("tools", meta)
            self.assertGreater(len(meta["tools"]), 0)

    def test_tool_subsets_are_compact_and_isolated(self):
        # Econometrics agent only has 3 focused tools
        self.assertEqual(len(econometrics_agent_tools), 3)
        tool_names = [t.name for t in econometrics_agent_tools]
        self.assertIn("run_stationarity_and_econometric_tests", tool_names)
        self.assertIn("forecast_price_arima", tool_names)

        # Risk volatility agent has GARCH and EDA tools
        risk_tool_names = [t.name for t in risk_volatility_agent_tools]
        self.assertIn("fit_garch_volatility_model", risk_tool_names)
        self.assertIn("calculate_full_stock_eda", risk_tool_names)

        # Portfolio optimizer agent has frontier optimization
        opt_tool_names = [t.name for t in portfolio_optimizer_agent_tools]
        self.assertIn("compute_optimal_portfolio_frontier", opt_tool_names)

        # Network graph agent has institutional holders and Graph RAG
        graph_tool_names = [t.name for t in network_graph_rag_agent_tools]
        self.assertIn("compare_common_institutional_holders", graph_tool_names)
        self.assertIn("retrieve_graph_rag_context", graph_tool_names)

    def test_deterministic_agent_query_classification(self):
        # 1. Econometrics routing
        self.assertEqual(
            classify_agent_for_query("Can you run the ADF test and KPSS test on AAPL?"),
            "econometrics_agent",
        )
        self.assertEqual(
            classify_agent_for_query("Forecast the price using ARIMA with 95% confidence"),
            "econometrics_agent",
        )

        # 2. Risk & Volatility routing
        self.assertEqual(
            classify_agent_for_query("Fit a GARCH(1,1) volatility model on MSFT"),
            "risk_volatility_agent",
        )
        self.assertEqual(
            classify_agent_for_query("Perform full stock EDA and calculate VaR and CVaR for NVDA"),
            "risk_volatility_agent",
        )

        # 3. Portfolio Optimization routing
        self.assertEqual(
            classify_agent_for_query("Optimize a portfolio of AAPL, MSFT, and GOOG on the efficient frontier"),
            "portfolio_optimizer_agent",
        )
        self.assertEqual(
            classify_agent_for_query("Calculate Max Sharpe and Risk Parity weights for my tickers"),
            "portfolio_optimizer_agent",
        )

        # 4. Institutional Network & Graph RAG routing
        self.assertEqual(
            classify_agent_for_query("Who are the common institutional holders between AAPL and MSFT according to 13F?"),
            "network_graph_rag_agent",
        )
        self.assertEqual(
            classify_agent_for_query("Search graph rag methodology and explain systemic risk"),
            "network_graph_rag_agent",
        )

        # 5. Technical Analysis routing
        self.assertEqual(
            classify_agent_for_query("Calculate RSI and MACD indicators for SPY"),
            "technical_analysis_agent",
        )

        # 6. Governance Audit routing
        self.assertEqual(
            classify_agent_for_query("Show my past analysis history and audit previous weights"),
            "governance_audit_agent",
        )

        # 7. Market Data routing
        self.assertEqual(
            classify_agent_for_query("List all available sectors and stocks in financials"),
            "market_data_agent",
        )

        # 8. General conversational fallback
        self.assertEqual(
            classify_agent_for_query("Hello! What are your capabilities?"),
            "chatbot",
        )

    def test_supervisor_router_from_state(self):
        state = {
            "messages": [HumanMessage(content="Run Dickey-Fuller stationarity test on TSLA")]
        }
        self.assertEqual(supervisor_router(state), "econometrics_agent")

        state = {
            "messages": [HumanMessage(content="What are the optimal weights for TSLA and AMZN under Max Sharpe?")]
        }
        self.assertEqual(supervisor_router(state), "portfolio_optimizer_agent")

    def test_langgraph_contains_all_specialized_nodes(self):
        graph_builder = _build_graph()
        node_names = set(graph_builder.nodes.keys())

        expected_nodes = {
            "classify_and_route",
            "memory_manager",
            "chatbot",
            "market_data_agent",
            "econometrics_agent",
            "risk_volatility_agent",
            "portfolio_optimizer_agent",
            "network_graph_rag_agent",
            "technical_analysis_agent",
            "governance_audit_agent",
            "tool_interceptor",
            "tools",
            "capture_tool_state",
            "finalize_governance",
        }
        self.assertTrue(expected_nodes.issubset(node_names))

    def test_langgraph_compilation_with_checkpointer(self):
        from langgraph.checkpoint.memory import MemorySaver

        graph_builder = _build_graph()
        compiled = graph_builder.compile(checkpointer=MemorySaver())
        self.assertIsNotNone(compiled)
        self.assertTrue(hasattr(compiled, "invoke"))
        self.assertTrue(hasattr(compiled, "stream"))
        self.assertTrue(hasattr(compiled, "astream_events"))


if __name__ == "__main__":
    unittest.main()
