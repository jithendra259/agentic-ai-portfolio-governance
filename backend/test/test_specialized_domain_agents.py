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
    stress_testing_agent_tools,
    explainability_agent_tools,
    data_sentinel_agent_tools,
)
from src.agents.advanced_governance_tools import (
    run_portfolio_stress_test,
    generate_executive_governance_memo,
    audit_data_integrity,
)


class TestSpecializedDomainAgents(unittest.TestCase):
    """Test suite verifying one-by-one specialized domain agents and LangGraph orchestration."""

    def test_all_ten_domain_agents_registered_in_metadata(self):
        expected_agents = {
            "market_data_agent",
            "econometrics_agent",
            "risk_volatility_agent",
            "portfolio_optimizer_agent",
            "network_graph_rag_agent",
            "technical_analysis_agent",
            "governance_audit_agent",
            "stress_testing_agent",
            "explainability_agent",
            "data_sentinel_agent",
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

        # Stress testing agent has stress test and volatility tools
        stress_tool_names = [t.name for t in stress_testing_agent_tools]
        self.assertIn("run_portfolio_stress_test", stress_tool_names)
        self.assertIn("fit_garch_volatility_model", stress_tool_names)

        # Explainability agent has memo generator and methodology RAG
        expl_tool_names = [t.name for t in explainability_agent_tools]
        self.assertIn("generate_executive_governance_memo", expl_tool_names)
        self.assertIn("search_methodology_knowledge_base", expl_tool_names)

        # Data sentinel agent has data integrity auditor
        data_tool_names = [t.name for t in data_sentinel_agent_tools]
        self.assertIn("audit_data_integrity", data_tool_names)
        self.assertIn("generate_missing_data_heatmap", data_tool_names)

    def test_deterministic_agent_query_classification(self):
        # 1. Stress Testing routing
        self.assertEqual(
            classify_agent_for_query("Simulate 2008 GFC crisis shock and run stress test on AAPL and MSFT"),
            "stress_testing_agent",
        )
        self.assertEqual(
            classify_agent_for_query("Run scenario analysis and drawdown simulation on my portfolio"),
            "stress_testing_agent",
        )

        # 2. Explainability routing
        self.assertEqual(
            classify_agent_for_query("Generate executive board memo explaining why this allocation has shadow costs"),
            "explainability_agent",
        )
        self.assertEqual(
            classify_agent_for_query("Draft a plain English fiduciary summary for the investment committee"),
            "explainability_agent",
        )

        # 3. Data Sentinel routing
        self.assertEqual(
            classify_agent_for_query("Audit data integrity and check for missing bars or bad ticks"),
            "data_sentinel_agent",
        )
        self.assertEqual(
            classify_agent_for_query("Perform pre-flight data completeness verification for U1"),
            "data_sentinel_agent",
        )

        # 4. Econometrics routing
        self.assertEqual(
            classify_agent_for_query("Can you run the ADF test and KPSS test on AAPL?"),
            "econometrics_agent",
        )
        self.assertEqual(
            classify_agent_for_query("Forecast the price using ARIMA with 95% confidence"),
            "econometrics_agent",
        )

        # 5. Risk & Volatility routing
        self.assertEqual(
            classify_agent_for_query("Fit a GARCH(1,1) volatility model on MSFT"),
            "risk_volatility_agent",
        )
        self.assertEqual(
            classify_agent_for_query("Perform full stock EDA and calculate VaR and CVaR for NVDA"),
            "risk_volatility_agent",
        )

        # 6. Portfolio Optimization routing
        self.assertEqual(
            classify_agent_for_query("Optimize a portfolio of AAPL, MSFT, and GOOG on the efficient frontier"),
            "portfolio_optimizer_agent",
        )

        # 7. Institutional Network & Graph RAG routing
        self.assertEqual(
            classify_agent_for_query("Who are the common institutional holders between AAPL and MSFT according to 13F?"),
            "network_graph_rag_agent",
        )

        # 8. Technical Analysis routing
        self.assertEqual(
            classify_agent_for_query("Calculate RSI and MACD indicators for SPY"),
            "technical_analysis_agent",
        )

        # 9. Governance Audit routing
        self.assertEqual(
            classify_agent_for_query("Show my past analysis history and audit previous weights"),
            "governance_audit_agent",
        )

        # 10. Market Data routing
        self.assertEqual(
            classify_agent_for_query("List all available sectors and stocks in financials"),
            "market_data_agent",
        )

        # 11. General conversational fallback
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
            "messages": [HumanMessage(content="Simulate 2008 GFC stress test on TSLA and NVDA")]
        }
        self.assertEqual(supervisor_router(state), "stress_testing_agent")

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
            "stress_testing_agent",
            "explainability_agent",
            "data_sentinel_agent",
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

    def test_run_specialized_agent_standalone(self):
        from src.agents.specialized_domain_agents import run_specialized_agent

        with patch("src.orchestrator.agent_nodes._invoke_llm_with_fallback") as mock_invoke:
            mock_invoke.return_value = AIMessage(content="GARCH(1,1) persistent volatility estimate is 22.4%.")
            res = run_specialized_agent("risk_volatility_agent", "Fit GARCH on AAPL", portfolio=["AAPL"])
            self.assertIn("GARCH(1,1)", res)
            self.assertIn("22.4%", res)

    def test_advanced_governance_tools_execution(self):
        # 1. Executive memo tool
        memo = generate_executive_governance_memo.invoke({
            "tickers": ["AAPL", "MSFT"],
            "universe": "U1",
            "risk_tolerance": "moderate",
        })
        self.assertIn("EXECUTIVE FIDUCIARY MEMORANDUM", memo)
        self.assertIn("Instability Index", memo)

        # 2. Data sentinel audit tool
        audit = audit_data_integrity.invoke({
            "tickers": ["AAPL", "MSFT"],
            "universe": "U1",
        })
        self.assertIn("Data Sentinel Audit", audit)

        # 3. Stress testing tool
        stress = run_portfolio_stress_test.invoke({
            "tickers": ["AAPL", "MSFT"],
            "scenarios": ["2008_GFC", "2020_COVID"],
        })
        self.assertIn("Macro Stress-Testing Report", stress)
        self.assertIn("2008 Great Financial Crisis", stress)

    def test_prompts_module_and_on_demand_retrieval(self):
        from src.orchestrator.prompts import (
            SUPERVISOR_SYSTEM_PROMPT,
            get_agent_prompt,
            get_chatbot_prompt,
            list_available_prompts,
        )

        self.assertEqual(get_chatbot_prompt(), SUPERVISOR_SYSTEM_PROMPT)
        available = list_available_prompts()
        self.assertIn("chatbot", available)

        for agent_name in [
            "market_data_agent",
            "econometrics_agent",
            "risk_volatility_agent",
            "portfolio_optimizer_agent",
            "network_graph_rag_agent",
            "technical_analysis_agent",
            "governance_audit_agent",
            "stress_testing_agent",
            "explainability_agent",
            "data_sentinel_agent",
        ]:
            self.assertIn(agent_name, available)
            prompt = get_agent_prompt(agent_name)
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 50)

        # Test fallback
        fallback = get_agent_prompt("unknown_agent")
        self.assertEqual(fallback, SUPERVISOR_SYSTEM_PROMPT)

        # Test expanded styles and personas
        from src.orchestrator.prompts import (
            EXECUTIVE_FIDUCIARY_STYLE_PROMPT,
            QUANTITATIVE_RESEARCH_STYLE_PROMPT,
            MATHEMATICAL_FORMULATION_PROMPT,
            ANTI_HALLUCINATION_GUARDRAILS_PROMPT,
            compose_agent_prompt,
        )
        self.assertIn("EXECUTIVE FIDUCIARY", EXECUTIVE_FIDUCIARY_STYLE_PROMPT)
        self.assertIn("QUANTITATIVE RESEARCH", QUANTITATIVE_RESEARCH_STYLE_PROMPT)
        self.assertIn("Sharpe Ratio", MATHEMATICAL_FORMULATION_PROMPT)
        self.assertIn("EVIDENCE BOUNDARY", ANTI_HALLUCINATION_GUARDRAILS_PROMPT)

        # Test compose_agent_prompt
        composed = compose_agent_prompt(
            agent_name="portfolio_optimizer_agent",
            style="executive",
            include_math=True,
            include_guardrails=True,
            portfolio=["AAPL", "MSFT"],
        )
        self.assertIn("Conic Portfolio Optimization", composed)
        self.assertIn("EXECUTIVE FIDUCIARY", composed)
        self.assertIn("LATEX MATHEMATICAL", composed)
        self.assertIn("ANTI-HALLUCINATION", composed)
        self.assertIn("Active Portfolio Securities Context: AAPL, MSFT", composed)

        # Test caveman compression mode in compose_agent_prompt
        caveman_composed = compose_agent_prompt(
            agent_name="risk_volatility_agent",
            caveman_mode="lite",
        )
        self.assertIn("Respond professionally but extremely tight", caveman_composed)


if __name__ == "__main__":
    unittest.main()

