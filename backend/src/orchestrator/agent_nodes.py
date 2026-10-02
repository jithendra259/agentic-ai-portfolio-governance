"""Agent Nodes & State Machine Handlers.

Defines the individual execution nodes for all specialized domain agents,
the conversational supervisor, memory compression, tool interception,
and response finalization.
"""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from src.orchestrator.caveman_agent import detect_caveman_request, get_caveman_system_prompt
from src.orchestrator.prompts import (
    DATA_SENTINEL_PROMPT,
    ECONOMETRIC_PROMPT,
    EXPLAINABILITY_PROMPT,
    GOVERNANCE_AUDIT_PROMPT,
    MARKET_DATA_PROMPT,
    NETWORK_GRAPH_RAG_PROMPT,
    PORTFOLIO_OPTIMIZATION_PROMPT,
    RISK_VOLATILITY_PROMPT,
    STRESS_TESTING_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TECHNICAL_ANALYSIS_PROMPT,
    get_agent_prompt,
    get_chatbot_prompt,
    get_prompt,
)
from src.agents.specialized_domain_agents import classify_agent_for_query
from src.intent.intent_classifier import IntentType
from src.orchestrator.formatters import (
    _build_governance_markdown,
    _extract_latest_governance_payload,
    _extract_latest_tool_output,
    _extract_portfolio_from_messages,
    _message_content_to_text,
    _sanitize_user_visible_response,
)
from src.orchestrator.llm_factory import (
    PRIMARY_OLLAMA_MODEL,
    _get_chat_llm,
    _invoke_llm_with_fallback,
    _trim_context,
)
from src.orchestrator.prompt_templates import (
    _MAX_CONTEXT_MESSAGES,
    _MAX_SUMMARY_CHARS,
    _latest_human_text,
    assemble_system_prompt,
)

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Specialized Agent Factory
# =====================================================================

def create_specialized_agent_node(agent_name: str, prompt_text: Optional[str] = None, agent_tools: Optional[list] = None) -> Callable:
    """Factory creating discrete, specialized domain agent nodes with on-demand prompt access."""
    resolved_tools = agent_tools or []

    def agent_node(state: dict[str, Any], config: RunnableConfig = None) -> dict[str, Any]:
        configured_session_state = (config or {}).get("configurable", {}).get("session_state") if config else None
        if isinstance(configured_session_state, dict):
            state = {**state, "session_state": configured_session_state}
        messages = state.get("messages", [])
        working_messages = list(messages)
        remembered_portfolio = _extract_portfolio_from_messages(working_messages)

        # Access the exact specialized agent prompt on demand from the dedicated prompts module
        sys_prompt = prompt_text or get_agent_prompt(agent_name)
        if remembered_portfolio:
            sys_prompt += f"\n\nActive Portfolio Context: {', '.join(remembered_portfolio)}."

        summary = state.get("summary", "").strip()
        if summary:
            if len(summary) > _MAX_SUMMARY_CHARS:
                summary = summary[:_MAX_SUMMARY_CHARS] + "..."
            sys_prompt += f"\n\nConversation Context Summary: {summary}"

        system_messages = [SystemMessage(content=sys_prompt)]
        if not working_messages or not isinstance(working_messages[0], SystemMessage):
            working_messages = system_messages + working_messages
        else:
            working_messages = system_messages + [
                m for m in working_messages if not isinstance(m, SystemMessage)
            ]

        working_messages = _trim_context(working_messages)
        response = _invoke_llm_with_fallback(working_messages, config, bound_tools=resolved_tools)

        if hasattr(response, "content") and response.content:
            clean_content = re.sub(r"```python.*?```", "", response.content, flags=re.DOTALL)
            clean_content = re.sub(r"```.*?```", "", clean_content, flags=re.DOTALL)
            if any(marker in clean_content for marker in ["plt.style.use", "import matplotlib", "sns.heatmap"]):
                lines = clean_content.splitlines()
                clean_content = "\n".join(l for l in lines if not any(m in l for m in ["plt.", "sns.", "import ", "pd.DataFrame"]))
            clean_content = re.sub(r"(?<!\bno\s)\bbuy\b(?!-)", "increase advisory exposure to", clean_content, flags=re.IGNORECASE)
            clean_content = re.sub(r"(?<!\bno\s)\bsell\b(?!-)", "reduce advisory exposure to", clean_content, flags=re.IGNORECASE)
            response.content = _sanitize_user_visible_response(clean_content.strip(), scratchpad=state.get("scratchpad"))

        return {
            "messages": [response],
            "active_agent": agent_name,
            "user_portfolio": remembered_portfolio or state.get("user_portfolio", []),
        }

    agent_node.__name__ = f"{agent_name}_node"
    return agent_node


# =====================================================================
# 2. Tool Interception & Anti-Loop Gating
# =====================================================================

def _latest_ai_tool_calls(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    latest = next((m for m in reversed(messages or []) if isinstance(m, AIMessage)), None)
    calls = getattr(latest, "tool_calls", None) if latest else None
    return calls if isinstance(calls, list) else []


def _tool_signature(tool_call: dict[str, Any]) -> dict[str, Any]:
    name = str(tool_call.get("name") or "").strip()
    args = tool_call.get("args") or {}
    try:
        args_text = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        args_text = str(args)
    return {"tool_name": name, "args_text": args_text.lower()}


def _tool_signature_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    if left.get("tool_name") != right.get("tool_name"):
        return 0.0
    return SequenceMatcher(None, left.get("args_text", ""), right.get("args_text", "")).ratio()


def tool_interceptor_node(state: dict[str, Any]) -> dict[str, Any]:
    """Block repeated semantically similar tool calls before ToolNode execution."""
    proposed_calls = _latest_ai_tool_calls(state.get("messages", []))
    if not proposed_calls:
        return {"route_status": "no_tools"}

    previous = state.get("recent_tool_signatures", [])
    new_signatures = [_tool_signature(call) for call in proposed_calls]
    for proposed in new_signatures:
        for old in previous[-8:]:
            if _tool_signature_similarity(proposed, old) > 0.92:
                return {
                    "route_status": "loop_blocked",
                    "messages": [
                        SystemMessage(
                            content=(
                                "CRITICAL SYSTEM OVERRIDE: You are in a semantic loop. "
                                "You must change strategies, use already available scratchpad/tool results, "
                                "or explain the blocker clearly instead of repeating the same tool call."
                            )
                        )
                    ],
                }

    return {
        "route_status": "tool_allowed",
        "recent_tool_signatures": new_signatures,
    }


def capture_tool_state_node(state: dict[str, Any]) -> dict[str, Any]:
    """Mirror scratchpad-save tool outputs into reducer-backed state."""
    latest_tool_name, latest_tool_output = _extract_latest_tool_output(state.get("messages", []))
    if latest_tool_name != "save_financial_metric":
        return {}

    try:
        payload = json.loads(latest_tool_output)
    except Exception:
        return {}
    if not isinstance(payload, dict) or not payload.get("scratchpad_save"):
        return {}

    metric_name = str(payload.get("metric_name") or "").strip()
    if not metric_name:
        return {}
    return {
        "scratchpad": {
            metric_name: {
                "exact_value": payload.get("exact_value"),
                "context": payload.get("context", ""),
            }
        }
    }


# =====================================================================
# 3. Routing Functions
# =====================================================================

def supervisor_router(state: dict[str, Any]) -> str:
    """Route from memory manager to the appropriate specialized agent or general chatbot supervisor."""
    messages = state.get("messages", [])
    if not messages:
        return "chatbot"

    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if not last_human:
        return "chatbot"

    user_text = _message_content_to_text(last_human)
    target_agent = classify_agent_for_query(user_text, state)
    logger.info("Supervisor orchestrator routed request to: %s", target_agent)
    return target_agent


def _route_after_agent(state: dict[str, Any]) -> str:
    return "tool_interceptor" if _latest_ai_tool_calls(state.get("messages", [])) else "end"

_route_after_chatbot = _route_after_agent


def _route_after_interceptor(state: dict[str, Any]) -> str:
    if state.get("route_status") == "tool_allowed":
        return "tools"
    if state.get("route_status") == "loop_blocked":
        if state.get("scratchpad"):
            return "finalize_governance"
        active = state.get("active_agent")
        if active in {
            "market_data_agent", "econometrics_agent", "risk_volatility_agent",
            "portfolio_optimizer_agent", "network_graph_rag_agent",
            "technical_analysis_agent", "governance_audit_agent",
            "stress_testing_agent", "explainability_agent", "data_sentinel_agent",
        }:
            return active
        return "chatbot"
    return "end"


def _route_after_tool(state: dict[str, Any]) -> str:
    latest_tool_name, latest_tool_output = _extract_latest_tool_output(state.get("messages", []))
    if latest_tool_name in {"run_full_governance_pipeline", "get_stock_database_snapshot"}:
        return "finalize_governance"
    if latest_tool_name == "plot_historical_prices":
        last_human = next((m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)), None)
        user_text = _message_content_to_text(last_human) if last_human else ""
        if not re.search(r"\b(volatility|cagr|drawdown|sharpe|sortino|return\s+distribution|correlation|covariance)\b", user_text, re.IGNORECASE):
            return "finalize_governance"

    active = state.get("active_agent")
    if active in {
        "market_data_agent", "econometrics_agent", "risk_volatility_agent",
        "portfolio_optimizer_agent", "network_graph_rag_agent",
        "technical_analysis_agent", "governance_audit_agent",
        "stress_testing_agent", "explainability_agent", "data_sentinel_agent",
        "chatbot",
    }:
        return active
    return "chatbot"


def _route_after_classification(state: dict[str, Any]) -> str:
    return state.get("route_status", "chatbot")
