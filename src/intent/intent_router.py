import json
import logging
import uuid
from collections.abc import Callable
from typing import Any, Optional

from src.intent.intent_classifier import IntentClassifier, IntentType


logger = logging.getLogger(__name__)


class IntentRouter:
    """
    Deterministically routes supported intents before any LLM tool choice occurs.
    """

    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        handlers: Optional[dict[str, Callable[..., Any]]] = None,
    ):
        self.classifier = classifier or IntentClassifier()
        self.handlers = handlers or self._default_handlers()

    def _default_handlers(self) -> dict[str, Callable[..., Any]]:
        from src.agents.live_data_tools import (
            analyze_institutional_network,
            get_stock_database_snapshot,
            get_stocks_by_sector,
            get_stocks_by_universe,
            get_universe_overview,
            list_available_sectors,
            run_historical_cvar_optimization,
        )

        return {
            "list_available_sectors": list_available_sectors,
            "get_stocks_by_sector": get_stocks_by_sector,
            "get_stocks_by_universe": get_stocks_by_universe,
            "get_universe_overview": get_universe_overview,
            "get_stock_database_snapshot": get_stock_database_snapshot,
            "analyze_institutional_network": analyze_institutional_network,
            "run_historical_cvar_optimization": run_historical_cvar_optimization,
        }

    def handle(self, user_message: str) -> dict:
        intent_match = self.classifier.classify(user_message)
        logger.info(
            "Intent classified: %s (%.2f, %s)",
            intent_match.intent.value,
            intent_match.confidence,
            intent_match.risk_tier.value,
        )

        if intent_match.intent == IntentType.GREETING:
            return self._success(intent_match, self._greeting_help())

        if intent_match.intent == IntentType.LIST_SECTORS:
            return self._success(intent_match, self._invoke("list_available_sectors"))

        if intent_match.intent == IntentType.GET_STOCKS_BY_SECTOR:
            return self._success(
                intent_match,
                self._invoke("get_stocks_by_sector", {"sector": intent_match.parameters.get("sector", "")}),
            )

        if intent_match.intent == IntentType.GET_STOCKS_BY_UNIVERSE:
            return self._success(
                intent_match,
                self._invoke("get_stocks_by_universe", {"universe": intent_match.parameters.get("universe", "")}),
            )

        if intent_match.intent == IntentType.UNIVERSE_OVERVIEW:
            return self._success(
                intent_match,
                self._invoke("get_universe_overview", {"universe": intent_match.parameters.get("universe", "")}),
            )

        if intent_match.intent == IntentType.STOCK_SNAPSHOT:
            tickers = intent_match.parameters.get("tickers", [])
            return self._success(
                intent_match,
                self._invoke("get_stock_database_snapshot", {"tickers": tickers}),
            )

        if intent_match.intent in {
            IntentType.ANALYZE_PORTFOLIO,
            IntentType.INSTITUTIONAL_NETWORK,
            IntentType.HISTORICAL_CVAR,
        }:
            return self._governance_review(intent_match)

        if intent_match.intent in {
            IntentType.FULL_PIPELINE_RUN,
            IntentType.ROLLING_WINDOW_TEST,
        }:
            return self._critical_review(intent_match)

        if intent_match.intent == IntentType.EXPLAIN_PARAMETERS:
            return self._success(intent_match, self._explain_parameters())

        if intent_match.intent == IntentType.METHODOLOGY_QUESTION:
            return self._success(intent_match, self._explain_methodology())

        if intent_match.intent == IntentType.DOCUMENTATION_REQUEST:
            return self._success(intent_match, self._provide_documentation())

        if intent_match.intent == IntentType.INVALID_EXECUTION:
            return self._rejected(intent_match, "Governance system is advisory-only and will not execute trades.")

        if intent_match.intent == IntentType.ADVERSARIAL:
            return self._rejected(intent_match, "Invalid request detected by the security gate.")

        if intent_match.intent == IntentType.MALFORMED:
            # Pass to LLM to ask follow-up questions instead of rejecting
            return {
                "intent": intent_match.intent.value,
                "confidence": intent_match.confidence,
                "risk_tier": intent_match.risk_tier.value,
                "status": "conversational_fallback",
                "parameters": intent_match.parameters,
            }

        if intent_match.intent == IntentType.OUT_OF_SCOPE:
            return self._rejected(
                intent_match,
                "Query is out of scope. Ask about sectors, universes, stocks, governance, or historical risk.",
            )

        return self._rejected(intent_match, "Unknown intent.")

    def _invoke(self, handler_name: str, payload: Optional[dict] = None) -> Any:
        handler = self.handlers[handler_name]
        raw_func = getattr(handler, "func", None)
        if callable(raw_func):
            if payload:
                return raw_func(**payload)
            return raw_func()
        if hasattr(handler, "invoke"):
            return handler.invoke(payload or {})
        if payload:
            return handler(**payload)
        return handler()

    def _success(self, intent_match, result: str) -> dict:
        return {
            "intent": intent_match.intent.value,
            "confidence": intent_match.confidence,
            "risk_tier": intent_match.risk_tier.value,
            "requires_hitl": False,
            "status": "success",
            "result": result,
            "parameters": intent_match.parameters,
        }

    def _governance_review(self, intent_match) -> dict:
        request_id = str(uuid.uuid4())
        summary = self._governance_summary(intent_match)
        return {
            "intent": intent_match.intent.value,
            "confidence": intent_match.confidence,
            "risk_tier": intent_match.risk_tier.value,
            "requires_hitl": True,
            "status": "pending_governance_review",
            "request_id": request_id,
            "parameters": intent_match.parameters,
            "governance_summary": summary,
            "message": f"Governance analysis requires explicit {intent_match.risk_tier.value} approval before execution.",
        }

    def _critical_review(self, intent_match) -> dict:
        request_id = str(uuid.uuid4())
        summary = {
            "intent": intent_match.intent.value,
            "parameters": intent_match.parameters,
            "approval_policy": "multi_approval_required",
        }
        return {
            "intent": intent_match.intent.value,
            "confidence": intent_match.confidence,
            "risk_tier": intent_match.risk_tier.value,
            "requires_hitl": True,
            "status": "pending_governance_review",
            "request_id": request_id,
            "parameters": intent_match.parameters,
            "governance_summary": summary,
            "message": "Batch backtests are gated and require multi-step approval before execution.",
        }

    def _governance_summary(self, intent_match) -> dict:
        parameters = intent_match.parameters
        return {
            "intent": intent_match.intent.value,
            "tickers": parameters.get("tickers", []),
            "universes": parameters.get("universes", []),
            "target_date": parameters.get("target_date"),
            "explanation": intent_match.explanation,
        }

    def _rejected(self, intent_match, reason: str) -> dict:
        return {
            "intent": intent_match.intent.value,
            "confidence": intent_match.confidence,
            "risk_tier": intent_match.risk_tier.value,
            "status": "rejected",
            "reason": reason,
            "requires_hitl": False,
            "parameters": intent_match.parameters,
        }

    def _explain_parameters(self) -> str:
        return (
            "Composite Instability Index (I_t)\n"
            "Measures systemic market instability using the leading eigenvalue of the correlation matrix.\n"
            "Typical reading:\n"
            "- I_t < 0.30: calm regime\n"
            "- 0.30 to 0.50: normal regime\n"
            "- 0.50 to 0.75: elevated instability\n"
            "- I_t >= 0.75: crisis regime\n\n"
            "Graph Penalty (lambda_t)\n"
            "Adaptive penalty that increases exposure control on structurally risky assets as instability rises."
        )

    def _explain_methodology(self) -> str:
        return (
            "G-CVaR Optimization Framework\n"
            "1. Agent 1 computes instability from historical returns.\n"
            "2. Agent 2 builds the institutional-holder network and structural-risk scores.\n"
            "3. Agent 3 optimizes CVaR with a graph-regularized penalty.\n"
            "4. Agent 4 explains high-risk outcomes for human review."
        )

    def _provide_documentation(self) -> str:
        docs = {
            "chat_orchestrator": "src/orchestrator/chatbot_orchestrator.py",
            "deterministic_dag": "src/orchestrator/langgraph_dag.py",
            "historical_tools": "src/agents/live_data_tools.py",
            "time_series_agent": "src/agents/time_series_a1.py",
            "graph_agent": "src/agents/graph_cag_a2.py",
            "optimizer_agent": "src/agents/optimizer_a3.py",
            "explainer_agent": "src/agents/explainer_a4.py",
        }
        return json.dumps(docs, indent=2)

    def _greeting_help(self) -> str:
        return (
            "Hi. You can ask for things like:\n"
            "- sectors\n"
            "- stocks in U1\n"
            "- show me tech stocks\n"
            "- analyze AAPL, MSFT, NVDA for 2008-10-15\n"
            "- how does G-CVaR work?"
        )
