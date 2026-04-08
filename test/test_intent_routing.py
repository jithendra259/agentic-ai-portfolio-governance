import os
import sys
import unittest
from unittest.mock import patch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intent.intent_classifier import IntentClassifier, IntentType, RiskTier
from src.intent.intent_router import IntentRouter


def _stub_result(name):
    class StubTool:
        def invoke(self, payload=None):
            return {"tool": name, "payload": payload or {}}

    return StubTool()


def _func_first_stub(name):
    class FuncFirstTool:
        def func(self, **payload):
            return {"tool": name, "payload": payload, "path": "func"}

        def invoke(self, payload=None):
            raise AssertionError("invoke should not be used when func is available")

    return FuncFirstTool()


def _snapshot_text_stub(snapshot_text):
    class SnapshotTool:
        def func(self, **payload):
            return snapshot_text

    return SnapshotTool()


class IntentClassifierTests(unittest.TestCase):
    def setUp(self):
        self.classifier = IntentClassifier(verbose=False)

    def test_classifies_data_lookup(self):
        result = self.classifier.classify("What sectors are available?")
        self.assertEqual(result.intent, IntentType.LIST_SECTORS)
        self.assertEqual(result.risk_tier, RiskTier.LOW)
        self.assertGreaterEqual(result.confidence, 0.9)

    def test_classifies_bare_sector_queries(self):
        result = self.classifier.classify("sectors")
        self.assertEqual(result.intent, IntentType.LIST_SECTORS)

        result = self.classifier.classify("sectors list")
        self.assertEqual(result.intent, IntentType.LIST_SECTORS)

    def test_classifies_greeting(self):
        result = self.classifier.classify("hi")
        self.assertEqual(result.intent, IntentType.GREETING)
        self.assertEqual(result.risk_tier, RiskTier.LOW)

    def test_classifies_simple_universe_lookup(self):
        result = self.classifier.classify("stocks in U1")
        self.assertEqual(result.intent, IntentType.GET_STOCKS_BY_UNIVERSE)
        self.assertEqual(result.parameters["universe"], "U1")

    def test_classifies_bare_known_sector_as_sector_lookup(self):
        result = self.classifier.classify("Industrials")
        self.assertEqual(result.intent, IntentType.GET_STOCKS_BY_SECTOR)
        self.assertEqual(result.parameters["sector"], "Industrials")

    def test_classifies_conversational_stock_snapshot_request(self):
        result = self.classifier.classify("tell me about the company for jpm")
        self.assertEqual(result.intent, IntentType.STOCK_SNAPSHOT)
        self.assertIn("JPM", result.parameters["tickers"])

    def test_classifies_summarize_stock_snapshot_request(self):
        result = self.classifier.classify("summarize jpm")
        self.assertEqual(result.intent, IntentType.STOCK_SNAPSHOT)
        self.assertIn("JPM", result.parameters["tickers"])

    def test_classifies_tell_me_more_about_ticker_request(self):
        result = self.classifier.classify("tell me more about NVDA")
        self.assertEqual(result.intent, IntentType.STOCK_SNAPSHOT)
        self.assertIn("NVDA", result.parameters["tickers"])

    def test_classifies_explain_ticker_request(self):
        result = self.classifier.classify("explain ticker TD")
        self.assertEqual(result.intent, IntentType.STOCK_SNAPSHOT)
        self.assertIn("TD", result.parameters["tickers"])

    def test_classifies_explain_the_ticker_request(self):
        result = self.classifier.classify("explain the MSFT")
        self.assertEqual(result.intent, IntentType.STOCK_SNAPSHOT)
        self.assertIn("MSFT", result.parameters["tickers"])

    def test_classifies_company_label_explain_request(self):
        result = self.classifier.classify("ASX: ASE Technology Holding Co., Ltd. explain this")
        self.assertEqual(result.intent, IntentType.STOCK_SNAPSHOT)
        self.assertIn("ASX", result.parameters["tickers"])

    def test_classifies_governance_request(self):
        result = self.classifier.classify("Analyze AAPL, MSFT, NVDA for 2008-10-15")
        self.assertEqual(result.intent, IntentType.ANALYZE_PORTFOLIO)
        self.assertEqual(result.parameters["tickers"], ["AAPL", "MSFT", "NVDA"])
        self.assertEqual(result.parameters["target_date"], "2008-10-15")
        self.assertEqual(result.risk_tier, RiskTier.HIGH)

    def test_classifies_graph_rag_query_as_low_risk_network_lookup(self):
        result = self.classifier.classify("show institutional overlap for U1")
        self.assertEqual(result.intent, IntentType.INSTITUTIONAL_NETWORK)
        self.assertEqual(result.parameters["universe"], "U1")
        self.assertEqual(result.risk_tier, RiskTier.LOW)

    def test_incomplete_in_domain_query_becomes_malformed(self):
        result = self.classifier.classify("analyse")
        self.assertEqual(result.intent, IntentType.MALFORMED)
        self.assertEqual(result.explanation, "In-domain query missing parameters. Routing to LLM.")

    def test_sector_explain_query_becomes_conversational_fallback(self):
        result = self.classifier.classify("Industrials explain")
        self.assertEqual(result.intent, IntentType.MALFORMED)
        self.assertEqual(result.parameters["sector"], "Industrials")

    def test_blocks_trade_execution(self):
        result = self.classifier.classify("Buy AAPL immediately")
        self.assertEqual(result.intent, IntentType.INVALID_EXECUTION)
        self.assertEqual(result.confidence, 0.0)

    def test_classifies_methodology_question(self):
        result = self.classifier.classify("How does G-CVaR work?")
        self.assertEqual(result.intent, IntentType.METHODOLOGY_QUESTION)
        self.assertEqual(result.risk_tier, RiskTier.LOW)

    def test_unknown_query_falls_back_to_chatbot_instead_of_out_of_scope(self):
        result = self.classifier.classify("What's the weather today?")
        self.assertEqual(result.intent, IntentType.MALFORMED)
        self.assertEqual(
            result.explanation,
            "No deterministic intent matched. Routing to chatbot for best-effort retrieval.",
        )


class IntentRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = IntentRouter(
            classifier=IntentClassifier(verbose=False),
            handlers={
                "list_available_sectors": _stub_result("list_available_sectors"),
                "get_stocks_by_sector": _stub_result("get_stocks_by_sector"),
                "get_stocks_by_universe": _stub_result("get_stocks_by_universe"),
                "get_universe_overview": _stub_result("get_universe_overview"),
                "get_stock_database_snapshot": _stub_result("get_stock_database_snapshot"),
                "analyze_institutional_network": _stub_result("analyze_institutional_network"),
                "run_historical_cvar_optimization": _stub_result("run_historical_cvar_optimization"),
            },
        )

    def test_routes_sector_lookup_without_hitl(self):
        result = self.router.handle("Show me tech stocks")
        self.assertEqual(result["intent"], IntentType.GET_STOCKS_BY_SECTOR.value)
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["requires_hitl"])
        self.assertEqual(result["result"]["payload"]["sector"], "tech")

    def test_routes_bare_sector_list_without_hitl(self):
        result = self.router.handle("sectors")
        self.assertEqual(result["intent"], IntentType.LIST_SECTORS.value)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["tool"], "list_available_sectors")

    def test_routes_greeting_to_help(self):
        result = self.router.handle("hi")
        self.assertEqual(result["intent"], IntentType.GREETING.value)
        self.assertEqual(result["status"], "success")
        self.assertIn("stocks in U1", result["result"])

    def test_routes_universe_lookup_without_hitl(self):
        result = self.router.handle("What's in U1?")
        self.assertEqual(result["intent"], IntentType.GET_STOCKS_BY_UNIVERSE.value)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["payload"]["universe"], "U1")

    def test_routes_simple_universe_lookup_without_hitl(self):
        result = self.router.handle("stocks in U1")
        self.assertEqual(result["intent"], IntentType.GET_STOCKS_BY_UNIVERSE.value)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["payload"]["universe"], "U1")

    def test_routes_bare_known_sector_without_hitl(self):
        result = self.router.handle("Industrials")
        self.assertEqual(result["intent"], IntentType.GET_STOCKS_BY_SECTOR.value)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["payload"]["sector"], "Industrials")

    def test_routes_tell_me_more_about_ticker_without_hitl(self):
        result = self.router.handle("tell me more about NVDA")
        self.assertEqual(result["intent"], IntentType.STOCK_SNAPSHOT.value)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["payload"]["tickers"], ["NVDA"])

    def test_routes_explain_ticker_without_hitl(self):
        result = self.router.handle("explain ticker TD")
        self.assertEqual(result["intent"], IntentType.STOCK_SNAPSHOT.value)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["payload"]["tickers"], ["TD"])

    def test_routes_explain_the_ticker_without_hitl(self):
        result = self.router.handle("explain the MSFT")
        self.assertEqual(result["intent"], IntentType.STOCK_SNAPSHOT.value)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["payload"]["tickers"], ["MSFT"])

    def test_routes_company_label_explain_request_without_hitl(self):
        result = self.router.handle("ASX: ASE Technology Holding Co., Ltd. explain this")
        self.assertEqual(result["intent"], IntentType.STOCK_SNAPSHOT.value)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["payload"]["tickers"], ["ASX"])

    def test_explanation_query_renders_human_explanation_not_raw_snapshot_dump(self):
        snapshot_text = (
            "MongoDB Stock Snapshot\n\n"
            "Ticker: JNJ\n"
            "- Company: Johnson & Johnson\n"
            "- Universes: U3\n"
            "- Sector: Healthcare\n"
            "- Industry: Drug Manufacturers - General\n"
            "- Country: United States\n"
            "- Website: https://www.jnj.com\n"
            "- Historical price coverage: 2005-01-03 to 2025-12-30\n"
            "- Most recent stored close: 205.82 on 2025-12-30\n"
            "- Historical observations stored: 5282\n"
            "- Key stats:\n"
            "  - market_cap: 579460202496\n"
            "  - trailing_pe: 21.779892\n"
            "  - forward_pe: 19.118612\n"
            "  - profit_margin: 0.28456\n"
            "  - return_on_equity: 0.35029998\n"
            "  - dividend_yield: 2.16\n"
            "  - beta: 0.326\n"
            "- Business summary: Johnson & Johnson operates across Innovative Medicine and MedTech.\n"
        )
        router = IntentRouter(
            classifier=IntentClassifier(verbose=False),
            handlers={
                "list_available_sectors": _stub_result("list_available_sectors"),
                "get_stocks_by_sector": _stub_result("get_stocks_by_sector"),
                "get_stocks_by_universe": _stub_result("get_stocks_by_universe"),
                "get_universe_overview": _stub_result("get_universe_overview"),
                "get_stock_database_snapshot": _snapshot_text_stub(snapshot_text),
                "analyze_institutional_network": _stub_result("analyze_institutional_network"),
                "run_historical_cvar_optimization": _stub_result("run_historical_cvar_optimization"),
            },
        )

        result = router.handle("explain jnj")

        self.assertEqual(result["intent"], IntentType.STOCK_SNAPSHOT.value)
        self.assertEqual(result["status"], "success")
        self.assertIn("Johnson & Johnson (JNJ) is", result["result"])
        self.assertNotIn("MongoDB Stock Snapshot", result["result"])

    def test_router_prefers_raw_func_for_fast_lookup_path(self):
        router = IntentRouter(
            classifier=IntentClassifier(verbose=False),
            handlers={
                "list_available_sectors": _stub_result("list_available_sectors"),
                "get_stocks_by_sector": _stub_result("get_stocks_by_sector"),
                "get_stocks_by_universe": _func_first_stub("get_stocks_by_universe"),
                "get_universe_overview": _stub_result("get_universe_overview"),
                "get_stock_database_snapshot": _stub_result("get_stock_database_snapshot"),
                "analyze_institutional_network": _stub_result("analyze_institutional_network"),
                "run_historical_cvar_optimization": _stub_result("run_historical_cvar_optimization"),
            },
        )
        result = router.handle("stocks in U1")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["path"], "func")
        self.assertEqual(result["result"]["payload"]["universe"], "U1")

    def test_routes_universe_overview_with_correct_payload_key(self):
        result = self.router.handle("summary of U1")
        self.assertEqual(result["intent"], IntentType.UNIVERSE_OVERVIEW.value)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["payload"]["universe"], "U1")

    def test_gates_governance_analysis_before_execution(self):
        result = self.router.handle("Analyze AAPL, MSFT, NVDA for 2008-10-15")
        self.assertEqual(result["intent"], IntentType.ANALYZE_PORTFOLIO.value)
        self.assertEqual(result["status"], "pending_governance_review")
        self.assertTrue(result["requires_hitl"])
        self.assertEqual(result["parameters"]["tickers"], ["AAPL", "MSFT", "NVDA"])

    def test_routes_graph_rag_queries_without_hitl(self):
        with patch("src.intent.intent_router.retrieve_graph_rag_context") as mock_graph_tool:
            mock_graph_tool.func.return_value = "Graph RAG Context\nTickers: AAA, BBB"
            result = self.router.handle("which institutions connect AAA and BBB")

        self.assertEqual(result["intent"], IntentType.INSTITUTIONAL_NETWORK.value)
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["requires_hitl"])
        self.assertIn("Graph RAG Context", result["result"])

    def test_routes_incomplete_in_domain_query_to_conversational_fallback(self):
        result = self.router.handle("analyse")
        self.assertEqual(result["intent"], IntentType.MALFORMED.value)
        self.assertEqual(result["status"], "conversational_fallback")

    def test_routes_sector_explain_query_to_conversational_fallback(self):
        result = self.router.handle("Industrials explain")
        self.assertEqual(result["intent"], IntentType.MALFORMED.value)
        self.assertEqual(result["status"], "conversational_fallback")
        self.assertEqual(result["parameters"]["sector"], "Industrials")

    def test_gates_backtest_requests_as_critical(self):
        result = self.router.handle("Run full governance pipeline across all 11 universes")
        self.assertEqual(result["intent"], IntentType.FULL_PIPELINE_RUN.value)
        self.assertEqual(result["risk_tier"], RiskTier.CRITICAL.value)
        self.assertEqual(result["status"], "pending_governance_review")

    def test_unknown_query_routes_to_conversational_fallback(self):
        result = self.router.handle("What's the weather today?")
        self.assertEqual(result["intent"], IntentType.MALFORMED.value)
        self.assertEqual(result["status"], "conversational_fallback")


if __name__ == "__main__":
    unittest.main()
