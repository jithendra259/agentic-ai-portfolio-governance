import json
import logging
import re
import uuid
from collections.abc import Callable
from typing import Any, Optional

from src.intent.intent_classifier import IntentClassifier, IntentType
from src.rag.rag_tools import (
    compare_common_institutional_holders,
    retrieve_graph_rag_context,
    search_methodology_knowledge_base,
)


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
            raw_snapshot = self._invoke("get_stock_database_snapshot", {"tickers": tickers})
            return self._success(
                intent_match,
                self._format_stock_snapshot_response(user_message, raw_snapshot),
            )

        if intent_match.intent == IntentType.INSTITUTIONAL_NETWORK:
            return self._success(intent_match, self._search_graph_context(user_message, intent_match))

        if intent_match.intent in {
            IntentType.ANALYZE_PORTFOLIO,
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
            return self._success(intent_match, self._search_methodology(user_message))

        if intent_match.intent == IntentType.DOCUMENTATION_REQUEST:
            return self._success(intent_match, self._provide_documentation())

        if intent_match.intent == IntentType.INVALID_EXECUTION:
            return self._rejected(intent_match, "Governance system is advisory-only and will not execute trades.")

        if intent_match.intent == IntentType.ADVERSARIAL:
            return self._rejected(intent_match, "Invalid request detected by the security gate.")

        if intent_match.intent in {IntentType.MALFORMED, IntentType.OUT_OF_SCOPE}:
            # Pass to the chatbot for best-effort retrieval or follow-up questions.
            return {
                "intent": intent_match.intent.value,
                "confidence": intent_match.confidence,
                "risk_tier": intent_match.risk_tier.value,
                "status": "conversational_fallback",
                "parameters": intent_match.parameters,
            }

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
            "Measures systemic market instability using a weighted composite of three signals:\n"
            "  1. Volatility spike (40% weight) -- annualised mean asset vol, normalised [0,1]\n"
            "  2. Correlation spike (30% weight) -- mean pairwise correlation, normalised [0,1]\n"
            "  3. Max drawdown (30% weight) -- mean asset drawdown, normalised [0,1]\n"
            "Formula: I_t = 0.4 * vol_norm + 0.3 * corr_norm + 0.3 * drawdown_norm\n\n"
            "Regime thresholds (TAU):\n"
            "- I_t < 0.50: Calm regime (no HITL intervention)\n"
            "- 0.50 <= I_t < 0.85: Elevated instability (advisory monitoring)\n"
            "- I_t >= 0.85: Crisis regime (Human-in-the-Loop mandatory review)\n\n"
            "Graph Penalty (lambda_t)\n"
            "Adaptive sigmoid penalty on structurally risky assets:\n"
            "  lambda_t = 1.0 / (1 + exp(-10 * (I_t - 0.85)))\n\n"
            "Weight constraint: max 15% per asset (MAX_WEIGHT=0.15).\n"
            "Turnover HITL trigger: > 40% turnover (TAU_TURNOVER=0.40)."
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
            "graph_agent": "src/agents/graph_rag_a2.py",
            "optimizer_agent": "src/agents/optimizer_a3.py",
            "explainer_agent": "src/agents/explainer_a4.py",
        }
        return json.dumps(docs, indent=2)

    def _search_methodology(self, user_message: str) -> str:
        raw_func = getattr(search_methodology_knowledge_base, "func", None)
        if callable(raw_func):
            return raw_func(question=user_message)
        return search_methodology_knowledge_base.invoke({"question": user_message})

    def _search_graph_context(self, user_message: str, intent_match) -> str:
        parameters = intent_match.parameters
        tickers = parameters.get("tickers", [])
        universes = parameters.get("universes", [])
        universe = parameters.get("universe", "")
        if not universe and universes:
            universe = universes[0]

        if self._wants_common_holder_comparison(user_message, parameters):
            raw_compare_func = getattr(compare_common_institutional_holders, "func", None)
            if callable(raw_compare_func):
                return raw_compare_func(universes=universes)
            return compare_common_institutional_holders.invoke({"universes": universes})

        raw_func = getattr(retrieve_graph_rag_context, "func", None)
        if callable(raw_func):
            return raw_func(tickers=tickers, universe=universe)
        return retrieve_graph_rag_context.invoke({"tickers": tickers, "universe": universe})

    def _wants_common_holder_comparison(self, user_message: str, parameters: dict[str, Any]) -> bool:
        normalized = str(user_message or "").lower()
        mentions_common = any(
            phrase in normalized
            for phrase in (
                "common holder",
                "common holders",
                "shared holder",
                "shared holders",
                "common institution",
                "common institutions",
                "common institute",
                "shared institution",
                "shared institutions",
            )
        )
        universe_count = len(parameters.get("universes", []))
        return mentions_common and universe_count >= 2

    def _greeting_help(self) -> str:
        return (
            "Hi. You can ask for things like:\n"
            "- sectors\n"
            "- stocks in U1\n"
            "- show me tech stocks\n"
            "- analyze AAPL, MSFT, NVDA for 2008-10-15\n"
            "- how does G-CVaR work?"
        )

    def _format_stock_snapshot_response(self, user_message: str, raw_snapshot: Any) -> Any:
        if not isinstance(raw_snapshot, str):
            return raw_snapshot
        if not self._wants_stock_explanation(user_message):
            return raw_snapshot
        if "Ticker:" not in raw_snapshot:
            return raw_snapshot

        stock_sections = self._parse_stock_snapshot_sections(raw_snapshot)
        if not stock_sections:
            return raw_snapshot

        explanations = [self._build_stock_explanation(section) for section in stock_sections]
        return "\n\n".join(part for part in explanations if part)

    def _wants_stock_explanation(self, user_message: str) -> bool:
        normalized = str(user_message or "").lower()
        explanation_markers = (
            "explain",
            "describe",
            "summarize",
            "summary",
            "overview",
            "tell me more",
            "more about",
            "what do we know about",
            "brief",
        )
        return any(marker in normalized for marker in explanation_markers)

    def _parse_stock_snapshot_sections(self, snapshot_text: str) -> list[dict[str, Any]]:
        body = str(snapshot_text or "")
        ticker_index = body.find("Ticker:")
        if ticker_index >= 0:
            body = body[ticker_index:]
        body = body.split("\n\nUnavailable tickers:", 1)[0].strip()
        if not body:
            return []

        blocks = re.split(r"\n\s*\n(?=Ticker:\s*)", body)
        parsed_sections = []

        for block in blocks:
            lines = [line.rstrip() for line in block.splitlines() if line.strip()]
            if not lines or not lines[0].startswith("Ticker:"):
                continue

            record: dict[str, Any] = {
                "key_stats": {},
                "financial": {},
                "graph": {},
                "analysis": {},
            }
            current_section = None

            for line in lines:
                if line.startswith("Ticker:"):
                    record["ticker"] = line.split(":", 1)[1].strip()
                    current_section = None
                elif line.startswith("- Company:"):
                    record["company"] = line.split(":", 1)[1].strip()
                elif line.startswith("- Universes:"):
                    record["universes"] = line.split(":", 1)[1].strip()
                elif line.startswith("- Sector:"):
                    record["sector"] = line.split(":", 1)[1].strip()
                elif line.startswith("- Industry:"):
                    record["industry"] = line.split(":", 1)[1].strip()
                elif line.startswith("- Country:"):
                    record["country"] = line.split(":", 1)[1].strip()
                elif line.startswith("- Website:"):
                    record["website"] = line.split(":", 1)[1].strip()
                elif line.startswith("- Historical price coverage:"):
                    record["historical_price_coverage"] = line.split(":", 1)[1].strip()
                elif line.startswith("- Most recent stored close:"):
                    record["most_recent_stored_close"] = line.split(":", 1)[1].strip()
                elif line.startswith("- Historical observations stored:"):
                    record["historical_observations"] = line.split(":", 1)[1].strip()
                elif line.startswith("- Key stats:"):
                    current_section = "key_stats"
                elif line.startswith("- Financial statement coverage:"):
                    current_section = "financial"
                elif line.startswith("- Graph and ownership data:"):
                    current_section = "graph"
                elif line.startswith("- Analyst and estimates data:"):
                    current_section = "analysis"
                elif line.startswith("- Business summary:"):
                    record["business_summary"] = line.split(":", 1)[1].strip()
                    current_section = None
                elif line.startswith("  - ") and current_section:
                    key, _, value = line[4:].partition(":")
                    record[current_section][key.strip()] = value.strip()

            parsed_sections.append(record)

        return parsed_sections

    def _build_stock_explanation(self, record: dict[str, Any]) -> str:
        ticker = record.get("ticker", "UNKNOWN")
        company = record.get("company", "Unknown Company")
        sector = record.get("sector", "Unknown sector")
        industry = record.get("industry", "Unknown industry")
        country = record.get("country", "Unknown country")
        universes = record.get("universes", "None stored")
        price_coverage = record.get("historical_price_coverage")
        latest_close = record.get("most_recent_stored_close")
        summary = record.get("business_summary", "")
        key_stats = record.get("key_stats", {})

        lines = [
            f"{company} ({ticker}) is a {country}-based company in the {sector} sector, specifically {industry}.",
            f"In this project database it is currently mapped to universe(s): {universes}.",
        ]

        if price_coverage:
            lines.append(f"The stored price history covers {price_coverage}.")
        if latest_close:
            lines.append(f"The latest stored close in MongoDB is {latest_close}.")

        fundamentals_line = self._build_fundamentals_explanation(key_stats)
        if fundamentals_line:
            lines.append(fundamentals_line)

        if summary:
            lines.append(f"Business summary: {summary}")

        return "\n".join(lines)

    def _build_fundamentals_explanation(self, key_stats: dict[str, str]) -> str:
        if not isinstance(key_stats, dict) or not key_stats:
            return ""

        market_cap = self._safe_float(key_stats.get("market_cap"))
        trailing_pe = self._safe_float(key_stats.get("trailing_pe"))
        forward_pe = self._safe_float(key_stats.get("forward_pe"))
        profit_margin = self._safe_float(key_stats.get("profit_margin"))
        return_on_equity = self._safe_float(key_stats.get("return_on_equity"))
        dividend_yield = self._safe_float(key_stats.get("dividend_yield"))
        beta = self._safe_float(key_stats.get("beta"))

        parts = []
        if market_cap is not None:
            parts.append(f"It has a stored market capitalization of about {self._humanize_number(market_cap)}.")
        if trailing_pe is not None:
            valuation_view = self._describe_pe(trailing_pe)
            forward_text = f", with a forward P/E of {forward_pe:.2f}" if forward_pe is not None else ""
            parts.append(
                f"On valuation, the trailing P/E is {trailing_pe:.2f}{forward_text}, which looks {valuation_view}."
            )
        if profit_margin is not None or return_on_equity is not None:
            margin_text = f"profit margin of {profit_margin * 100:.2f}%" if profit_margin is not None else None
            roe_text = f"return on equity of {return_on_equity * 100:.2f}%" if return_on_equity is not None else None
            metrics = " and ".join(part for part in [margin_text, roe_text] if part)
            quality_view = self._describe_quality(profit_margin, return_on_equity)
            if metrics:
                parts.append(f"Profitability looks {quality_view}, based on a {metrics}.")
        if dividend_yield is not None:
            parts.append(
                f"The stored dividend yield is {dividend_yield:.2f}%, which suggests {self._describe_dividend(dividend_yield)}."
            )
        if beta is not None:
            parts.append(f"The beta is {beta:.2f}, implying {self._describe_beta(beta)}.")

        return " ".join(parts)

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value in (None, "", "N/A"):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _humanize_number(value: float) -> str:
        thresholds = [
            (1_000_000_000_000, "trillion"),
            (1_000_000_000, "billion"),
            (1_000_000, "million"),
        ]
        absolute_value = abs(value)
        for threshold, label in thresholds:
            if absolute_value >= threshold:
                return f"{value / threshold:.2f} {label}"
        return f"{value:,.0f}"

    @staticmethod
    def _describe_pe(pe_ratio: float) -> str:
        if pe_ratio < 15:
            return "relatively low"
        if pe_ratio < 30:
            return "moderate"
        return "relatively rich"

    @staticmethod
    def _describe_quality(profit_margin: Optional[float], return_on_equity: Optional[float]) -> str:
        margin = profit_margin if profit_margin is not None else 0.0
        roe = return_on_equity if return_on_equity is not None else 0.0
        if margin >= 0.2 or roe >= 0.2:
            return "strong"
        if margin >= 0.1 or roe >= 0.1:
            return "solid"
        if margin > 0 or roe > 0:
            return "positive"
        return "weak"

    @staticmethod
    def _describe_dividend(dividend_yield: float) -> str:
        if dividend_yield >= 3:
            return "a meaningful income component"
        if dividend_yield >= 1:
            return "a modest income component"
        return "limited income contribution"

    @staticmethod
    def _describe_beta(beta: float) -> str:
        if beta < 0.8:
            return "lower volatility than the broad market"
        if beta <= 1.2:
            return "volatility roughly in line with the broad market"
        return "higher volatility than the broad market"
