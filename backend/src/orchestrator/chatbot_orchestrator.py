"""Conversational Agentic Supervisor & Multi-Agent LangGraph Orchestrator.

Orchestrates specialized domain agents (Market Data, Econometrics, Risk & Volatility,
Portfolio Optimization, Network Graph RAG, Technical Analysis, Governance Audit)
and coordinates them via LangGraph state machine.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated, Any, Optional, Tuple, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

# Checkpointer fallbacks
try:
    from langgraph.checkpoint.mongodb import MongoDBSaver
except Exception:
    MongoDBSaver = None

try:
    from langgraph.checkpoint.postgres import PostgresSaver
except Exception:
    PostgresSaver = None

# Historical and real-time tools
from src.agents.custom_math_plot import generate_custom_math_plot
from src.agents.derived_plot_tools import (
    generate_missing_data_heatmap,
    generate_ohlc_correlation_heatmap,
    run_data_analysis_plot,
)
from src.agents.generate_dynamic_plot import generate_financial_plot
from src.agents.history_tools import get_detailed_past_weights, get_user_analysis_history
from src.agents.live_data_tools import (
    get_market_data_bundle,
    get_stock_database_snapshot,
    get_stocks_by_sector,
    get_stocks_by_universe,
    get_universe_overview,
    get_yfinance_market_data,
    list_available_sectors,
    list_available_universes,
    plot_historical_prices,
    plot_us_economic_indicators,
    run_full_governance_pipeline,
)
from src.agents.math_eda_tools import (
    calculate_full_stock_eda,
    compute_optimal_portfolio_frontier,
    fit_garch_volatility_model,
    forecast_price_arima,
    run_stationarity_and_econometric_tests,
)
from src.agents.advanced_governance_tools import (
    audit_data_integrity,
    generate_executive_governance_memo,
    run_portfolio_stress_test,
)
from src.agents.price_series_tool import get_price_series_for_analysis
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
from src.agents.specialized_domain_agents import (
    SPECIALIZED_AGENTS_METADATA,
    classify_agent_for_query,
)
from src.intent.intent_classifier import IntentClassifier, IntentType
from src.intent.intent_router import IntentRouter
from src.memory.mongodb_memory_layer import MongoMemoryManager
from src.orchestrator.agent_nodes import (
    _latest_ai_tool_calls,
    _route_after_agent,
    _route_after_chatbot,
    _route_after_classification,
    _route_after_interceptor,
    _route_after_tool,
    _tool_signature,
    _tool_signature_similarity,
    capture_tool_state_node,
    create_specialized_agent_node,
    supervisor_router,
    tool_interceptor_node,
)
from src.orchestrator.formatters import (
    _build_governance_markdown,
    _extract_latest_governance_payload,
    _extract_latest_tool_output,
    _extract_portfolio_from_messages,
    _extract_tickers_from_text,
    _humanize_status,
    _message_content_to_text,
    _sanitize_user_visible_response,
)
from src.orchestrator.llm_factory import (
    CONFIGURED_DEFAULT_LLM_MODEL,
    CONFIGURED_FALLBACK_OLLAMA_MODEL,
    CONFIGURED_PRIMARY_OLLAMA_MODEL,
    FALLBACK_OLLAMA_MODEL,
    INSTALLED_OLLAMA_MODELS,
    PRIMARY_OLLAMA_MODEL,
    _available_models_text,
    _build_llm_with_tools,
    _clean_messages_for_ashna,
    _clean_messages_for_model,
    _get_chat_llm,
    _invoke_llm_with_fallback,
    _is_ashna_model,
    _is_ollama_memory_error,
    _is_ollama_model_not_found_error,
    _is_ollama_unavailable_error,
    _is_retryable_ollama_error,
    _list_installed_ollama_models,
    _memory_error_message,
    _model_not_found_message,
    _resolve_ollama_model,
    _should_probe_ollama_on_startup,
    _trim_context,
    get_fallback_llm_with_tools,
    get_llm_with_tools,
    register_global_tools,
)
from src.orchestrator.prompt_templates import (
    CHATBOT_CONVERSATION_GUIDANCE_DIR,
    SYSTEM_PROMPT,
    _MAX_CONTEXT_MESSAGES,
    _MAX_SUMMARY_CHARS,
    _latest_human_text,
    _load_chatbot_conversation_guidance,
    assemble_system_prompt,
)

logger = logging.getLogger(__name__)
GOVERNANCE_CACHE_VERSION = "optimizer-audit-v3-yfinance"


# =====================================================================
# Reducers & State Definition
# =====================================================================

def add_messages(current: list[BaseMessage] | None, update: list[BaseMessage] | None) -> list[BaseMessage]:
    return [*(current or []), *(update or [])]


def merge_scratchpads(current: dict[str, Any] | None, update: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(current or {})
    merged.update(update or {})
    return merged


def append_recent_tool_signatures(current: list[dict[str, Any]] | None, update: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    combined = [*(current or []), *(update or [])]
    return combined[-12:]


def replace_summary(current: str | None, update: str | None) -> str:
    return str(update) if update is not None else (current or "")


class AgentState(TypedDict, total=False):
    original_goal: str
    scratchpad: Annotated[dict[str, Any], merge_scratchpads]
    historical_summary: Annotated[str, replace_summary]
    recent_messages: Annotated[list[BaseMessage], add_messages]
    recent_tool_signatures: Annotated[list[dict[str, Any]], append_recent_tool_signatures]
    messages: Annotated[list[BaseMessage], add_messages]
    user_portfolio: list[str]
    risk_profile: str
    route_status: str
    route_result: dict[str, Any]
    summary: str
    caveman_mode: bool
    caveman_intensity: str
    chat_history_last_25: list[dict[str, Any]]
    session_state: dict[str, Any]
    resolved_context: dict[str, Any]
    pending_action: dict[str, Any] | None
    memory_update: dict[str, Any]
    validation_result: dict[str, Any]
    active_agent: str


# =====================================================================
# Memory Management & Persistence
# =====================================================================

_mongo_memory_instance: Optional[MongoMemoryManager] = None
_checkpointer_instance: Optional[Any] = None
_memory_init_lock = threading.Lock()


def _init_mongo_memory() -> MongoMemoryManager:
    global _mongo_memory_instance, _checkpointer_instance
    if _mongo_memory_instance is None:
        with _memory_init_lock:
            if _mongo_memory_instance is None:
                _mongo_memory_instance = MongoMemoryManager()
                try:
                    _checkpointer_instance = _mongo_memory_instance.get_checkpointer()
                except Exception as exc:
                    logger.warning("Failed to initialize checkpointer: %s. Using MemorySaver fallback.", exc)
                    _checkpointer_instance = MemorySaver()
    return _mongo_memory_instance


def get_memory_manager() -> MongoMemoryManager:
    return _init_mongo_memory()


def get_checkpointer():
    if _checkpointer_instance is None:
        _init_mongo_memory()
    return _checkpointer_instance


def is_memory_initialized() -> bool:
    return _mongo_memory_instance is not None


def get_postgres_status() -> dict[str, Any]:
    if not is_memory_initialized():
        return {"status": "uninitialized", "pg_pool": False, "write_errors": 0}
    return get_memory_manager().get_postgres_health()


class LazyMemoryManager:
    def __getattr__(self, name: str) -> Any:
        return getattr(get_memory_manager(), name)

    def __bool__(self) -> bool:
        return True


memory_manager = LazyMemoryManager()
intent_classifier = IntentClassifier(verbose=True)
intent_router = IntentRouter(classifier=intent_classifier)


# =====================================================================
# Tool Wrappers
# =====================================================================

@tool("save_financial_metric")
def save_financial_metric(metric_name: str, exact_value: float | str, context: str = "") -> str:
    """Persist an exact financial metric in the agent scratchpad."""
    clean_name = str(metric_name or "").strip()
    if not clean_name:
        return "SCRATCHPAD_SAVE_ERROR: metric_name is required."
    return json.dumps({
        "scratchpad_save": True,
        "metric_name": clean_name,
        "exact_value": exact_value,
        "context": str(context or ""),
    })


@tool("search_methodology_knowledge_base")
def search_methodology_knowledge_base(question: str) -> str:
    """Search local methodology/PDF knowledge for framework and EDA questions."""
    from src.rag.rag_tools import search_methodology_knowledge_base as real_tool
    raw_func = getattr(real_tool, "func", None)
    return raw_func(question=question) if callable(raw_func) else real_tool.invoke({"question": question})


@tool("retrieve_graph_rag_context")
def retrieve_graph_rag_context(tickers: list[str] | None = None, universe: str = "") -> str:
    """Retrieve institutional ownership and graph context for stocks or a universe."""
    from src.rag.rag_tools import retrieve_graph_rag_context as real_tool
    payload = {"tickers": tickers or [], "universe": universe}
    raw_func = getattr(real_tool, "func", None)
    return raw_func(**payload) if callable(raw_func) else real_tool.invoke(payload)


@tool("compare_common_institutional_holders")
def compare_common_institutional_holders(universes: list[str] | None = None) -> str:
    """Compare common institutional holders across multiple universes."""
    from src.rag.rag_tools import compare_common_institutional_holders as real_tool
    payload = {"universes": universes or []}
    raw_func = getattr(real_tool, "func", None)
    return raw_func(**payload) if callable(raw_func) else real_tool.invoke(payload)


@tool("run_full_governance_pipeline")
def governance_pipeline_with_cache(
    tickers: list[str],
    target_date: str,
    risk_tolerance: str = "moderate",
    previous_weights: Optional[dict[str, float]] = None,
    config: RunnableConfig = None,
) -> str:
    """Governance wrapper with L2 semantic plan cache."""
    normalized_risk_tolerance = (risk_tolerance or "moderate").strip().lower()
    configurable = (config or {}).get("configurable", {})
    configured_weights = configurable.get("previous_weights")
    configured_weights = dict(configured_weights) if isinstance(configured_weights, dict) else {}
    resolved_previous_weights = previous_weights or configured_weights or None
    cache_risk_tolerance = f"{normalized_risk_tolerance}|{GOVERNANCE_CACHE_VERSION}"

    query_hash = memory_manager.compute_query_hash(
        tickers=tickers, target_date=target_date, risk_tolerance=cache_risk_tolerance
    )
    cached = None if resolved_previous_weights else memory_manager.retrieve_cached_plan(query_hash)
    if cached:
        return cached

    result = run_full_governance_pipeline.invoke({
        "tickers": tickers,
        "target_date": target_date,
        "risk_tolerance": normalized_risk_tolerance,
        "previous_weights": resolved_previous_weights,
    })
    if isinstance(result, dict):
        if not resolved_previous_weights:
            memory_manager.cache_governance_plan(query_hash=query_hash, payload=result, ttl_days=7)
        return result

    serialized = json.dumps(result)
    if not resolved_previous_weights:
        memory_manager.cache_governance_plan(query_hash=query_hash, payload=serialized, ttl_days=7)
    return serialized


# =====================================================================
# Tool Subsets for Specialized Agents
# =====================================================================

tools = [
    list_available_sectors,
    list_available_universes,
    get_stocks_by_sector,
    get_stocks_by_universe,
    get_universe_overview,
    get_stock_database_snapshot,
    get_market_data_bundle,
    get_yfinance_market_data,
    plot_historical_prices,
    plot_us_economic_indicators,
    get_price_series_for_analysis,
    governance_pipeline_with_cache,
    search_methodology_knowledge_base,
    retrieve_graph_rag_context,
    compare_common_institutional_holders,
    get_user_analysis_history,
    get_detailed_past_weights,
    generate_financial_plot,
    generate_custom_math_plot,
    generate_ohlc_correlation_heatmap,
    generate_missing_data_heatmap,
    run_data_analysis_plot,
    save_financial_metric,
    run_stationarity_and_econometric_tests,
    fit_garch_volatility_model,
    calculate_full_stock_eda,
    compute_optimal_portfolio_frontier,
    forecast_price_arima,
    run_portfolio_stress_test,
    generate_executive_governance_memo,
    audit_data_integrity,
]

# Register with LLM factory
register_global_tools(tools)

market_data_agent_tools = [
    list_available_sectors,
    list_available_universes,
    get_stocks_by_sector,
    get_stocks_by_universe,
    get_universe_overview,
    get_stock_database_snapshot,
    get_market_data_bundle,
    get_yfinance_market_data,
    plot_historical_prices,
    plot_us_economic_indicators,
]

econometrics_agent_tools = [
    run_stationarity_and_econometric_tests,
    forecast_price_arima,
    get_price_series_for_analysis,
]

risk_volatility_agent_tools = [
    fit_garch_volatility_model,
    calculate_full_stock_eda,
    generate_ohlc_correlation_heatmap,
    generate_missing_data_heatmap,
    save_financial_metric,
]

portfolio_optimizer_agent_tools = [
    compute_optimal_portfolio_frontier,
    governance_pipeline_with_cache,
    get_detailed_past_weights,
    save_financial_metric,
]

network_graph_rag_agent_tools = [
    retrieve_graph_rag_context,
    compare_common_institutional_holders,
    search_methodology_knowledge_base,
]

technical_analysis_agent_tools = [
    get_price_series_for_analysis,
    generate_financial_plot,
    generate_custom_math_plot,
    run_data_analysis_plot,
]

governance_audit_agent_tools = [
    get_user_analysis_history,
    save_financial_metric,
]

stress_testing_agent_tools = [
    run_portfolio_stress_test,
    fit_garch_volatility_model,
    get_price_series_for_analysis,
    save_financial_metric,
]

explainability_agent_tools = [
    generate_executive_governance_memo,
    search_methodology_knowledge_base,
    get_detailed_past_weights,
    save_financial_metric,
]

data_sentinel_agent_tools = [
    audit_data_integrity,
    generate_missing_data_heatmap,
    get_price_series_for_analysis,
    save_financial_metric,
]


# =====================================================================
# Specialized Domain Agent Nodes
# =====================================================================

market_data_agent_node = create_specialized_agent_node(
    "market_data_agent", agent_tools=market_data_agent_tools
)
econometrics_agent_node = create_specialized_agent_node(
    "econometrics_agent", agent_tools=econometrics_agent_tools
)
risk_volatility_agent_node = create_specialized_agent_node(
    "risk_volatility_agent", agent_tools=risk_volatility_agent_tools
)
portfolio_optimizer_agent_node = create_specialized_agent_node(
    "portfolio_optimizer_agent", agent_tools=portfolio_optimizer_agent_tools
)
network_graph_rag_agent_node = create_specialized_agent_node(
    "network_graph_rag_agent", agent_tools=network_graph_rag_agent_tools
)
technical_analysis_agent_node = create_specialized_agent_node(
    "technical_analysis_agent", agent_tools=technical_analysis_agent_tools
)
governance_audit_agent_node = create_specialized_agent_node(
    "governance_audit_agent", agent_tools=governance_audit_agent_tools
)
stress_testing_agent_node = create_specialized_agent_node(
    "stress_testing_agent", agent_tools=stress_testing_agent_tools
)
explainability_agent_node = create_specialized_agent_node(
    "explainability_agent", agent_tools=explainability_agent_tools
)
data_sentinel_agent_node = create_specialized_agent_node(
    "data_sentinel_agent", agent_tools=data_sentinel_agent_tools
)


# =====================================================================
# Conversational Supervisor Node & Intent Gates
# =====================================================================

def chatbot_node(state: AgentState, config: RunnableConfig):
    """The central conversational supervisor and coordinator."""
    configured_session_state = (config or {}).get("configurable", {}).get("session_state") if config else None
    if isinstance(configured_session_state, dict):
        state = {**state, "session_state": configured_session_state}
    messages = state.get("messages", [])

    working_messages = list(messages)
    remembered_portfolio = _extract_portfolio_from_messages(working_messages)

    system_messages = [SystemMessage(content=assemble_system_prompt(state))]
    if remembered_portfolio:
        system_messages.append(
            SystemMessage(content=f"Active portfolio context: {', '.join(remembered_portfolio)}.")
        )

    if not working_messages or not isinstance(working_messages[0], SystemMessage):
        working_messages = system_messages + working_messages
    else:
        working_messages = system_messages + [
            m for m in working_messages if not isinstance(m, SystemMessage)
        ]

    working_messages = _trim_context(working_messages)
    response = _invoke_llm_with_fallback(working_messages, config)

    if hasattr(response, "content") and response.content:
        clean = re.sub(r"```python.*?```", "", response.content, flags=re.DOTALL)
        clean = re.sub(r"```.*?```", "", clean, flags=re.DOTALL)
        clean = re.sub(r"(?<!\bno\s)\bbuy\b(?!-)", "increase advisory exposure to", clean, flags=re.IGNORECASE)
        clean = re.sub(r"(?<!\bno\s)\bsell\b(?!-)", "reduce advisory exposure to", clean, flags=re.IGNORECASE)
        response.content = _sanitize_user_visible_response(clean.strip(), scratchpad=state.get("scratchpad"))

    return {
        "messages": [response],
        "user_portfolio": remembered_portfolio or state.get("user_portfolio", []),
        "active_agent": "chatbot",
    }


def memory_manager_node(state: AgentState, config: RunnableConfig):
    """Context summarization node for long-term memory."""
    messages = state.get("messages", [])
    original_goal = state.get("original_goal") or _latest_human_text(messages)
    if len(messages) <= _MAX_CONTEXT_MESSAGES:
        return {
            "original_goal": original_goal,
            "summary": state.get("summary", ""),
            "historical_summary": state.get("historical_summary", state.get("summary", "")),
        }

    existing_summary = state.get("historical_summary") or state.get("summary", "")
    to_summarize = messages[:-_MAX_CONTEXT_MESSAGES]
    history_str = "\n".join([f"{m.type}: {_message_content_to_text(m)}" for m in to_summarize])

    prompt = (
        "You are a long-term memory processor for a Portfolio Governance Assistant.\n"
        "Update the existing summary with new historical messages concisely.\n\n"
        f"EXISTING SUMMARY: {existing_summary or 'None'}\n\n"
        f"MESSAGES:\n{history_str}\n\n"
        "Return ONLY the updated summary."
    )
    try:
        summarizer = _get_chat_llm(PRIMARY_OLLAMA_MODEL, temperature=0, num_predict=256)
        response = summarizer.invoke(prompt)
        new_summary = (response.content if hasattr(response, "content") else str(response)).strip()
        return {"original_goal": original_goal, "summary": new_summary, "historical_summary": new_summary}
    except Exception as exc:
        logger.warning("Summarization failed: %s", exc)
        return {"original_goal": original_goal, "summary": existing_summary, "historical_summary": existing_summary}


def summarize_conversation_node(state: AgentState, config: RunnableConfig):
    return memory_manager_node(state, config)


def classify_and_route_node(state: AgentState, config: RunnableConfig = None):
    """Deterministic intent gate before conversational LLM."""
    messages = state.get("messages", [])
    if not messages:
        return {"route_status": "chatbot"}

    latest_msg = messages[-1]
    if not isinstance(latest_msg, HumanMessage):
        return {"route_status": "chatbot"}

    user_input = _message_content_to_text(latest_msg)
    normalized = user_input.lower()

    if ("unemployment" in normalized and "gdp" in normalized) or "recession band" in normalized:
        plot_us_economic_indicators.func(config=config)
        return {
            "messages": [AIMessage(content="Here is the US unemployment rate comparison with GDP per capita, including the shaded recession bands and dual Y-axes.")],
            "route_status": "end",
        }

    match = intent_router.classifier.classify(user_input)
    if match.intent == IntentType.ADVERSARIAL:
        return {"route_status": "blocked", "route_explanation": match.explanation}

    allowed_chatbot = {
        IntentType.STOCK_SNAPSHOT, IntentType.METHODOLOGY_QUESTION,
        IntentType.EXPLAIN_PARAMETERS, IntentType.HISTORICAL_CHART,
        IntentType.MALFORMED, IntentType.GREETING, IntentType.LIST_SECTORS,
        IntentType.UNIVERSE_OVERVIEW, IntentType.DOCUMENTATION_REQUEST,
    }
    if match.intent in allowed_chatbot:
        return {"route_status": "chatbot"}

    route_result = intent_router.handle(user_input)
    if route_result.get("status") == "success":
        return {"messages": [AIMessage(content=str(route_result["result"]))], "route_status": "end"}

    return {"route_status": "chatbot", "route_result": route_result}


def finalize_governance_node(state: AgentState, config: RunnableConfig):
    """Render markdown report or pass synthesized content."""
    messages = state.get("messages", [])
    if state.get("route_status") == "loop_blocked" and state.get("scratchpad"):
        content = _sanitize_user_visible_response(
            '{"scratchpad_save": true}',
            scratchpad=state.get("scratchpad"),
        )
        return {"messages": [AIMessage(content=content)]}

    if not messages:
        return {"messages": [AIMessage(content="Unable to generate a response for this request.")]}

    latest_tool_name, latest_tool_output = _extract_latest_tool_output(messages)
    if latest_tool_name == "get_stock_database_snapshot" and latest_tool_output:
        last_human = next((message for message in reversed(messages) if isinstance(message, HumanMessage)), None)
        user_text = _message_content_to_text(last_human) if last_human is not None else ""
        if intent_router._wants_stock_explanation(user_text):
            stock_sections = intent_router._parse_stock_snapshot_sections(latest_tool_output)
            if stock_sections:
                formatted = "\n\n".join(
                    intent_router._build_stock_explanation(section)
                    for section in stock_sections
                )
                return {"messages": [AIMessage(content=formatted)]}

    if latest_tool_name != "run_full_governance_pipeline":
        content = _sanitize_user_visible_response(latest_tool_output or "Analysis complete.")
        return {"messages": [AIMessage(content=content)]}

    governance_payload, raw_text = _extract_latest_governance_payload(messages)
    content = _build_governance_markdown(governance_payload, raw_text)
    return {"messages": [AIMessage(content=content)]}


# =====================================================================
# 4. LangGraph Multi-Agent State Machine Builder
# =====================================================================

def _build_graph():
    """Build the LangGraph multi-agent state machine."""
    from langgraph.graph import END, StateGraph
    from langgraph.prebuilt import ToolNode

    builder = StateGraph(AgentState)

    # Core nodes
    builder.add_node("classify_and_route", classify_and_route_node)
    builder.add_node("memory_manager", memory_manager_node)
    builder.add_node("chatbot", chatbot_node)

    # 10 Specialized Domain Agent Nodes
    builder.add_node("market_data_agent", market_data_agent_node)
    builder.add_node("econometrics_agent", econometrics_agent_node)
    builder.add_node("risk_volatility_agent", risk_volatility_agent_node)
    builder.add_node("portfolio_optimizer_agent", portfolio_optimizer_agent_node)
    builder.add_node("network_graph_rag_agent", network_graph_rag_agent_node)
    builder.add_node("technical_analysis_agent", technical_analysis_agent_node)
    builder.add_node("governance_audit_agent", governance_audit_agent_node)
    builder.add_node("stress_testing_agent", stress_testing_agent_node)
    builder.add_node("explainability_agent", explainability_agent_node)
    builder.add_node("data_sentinel_agent", data_sentinel_agent_node)

    # Execution pipeline
    builder.add_node("tool_interceptor", tool_interceptor_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("capture_tool_state", capture_tool_state_node)
    builder.add_node("finalize_governance", finalize_governance_node)

    builder.set_entry_point("classify_and_route")
    builder.add_conditional_edges(
        "classify_and_route",
        _route_after_classification,
        {"chatbot": "memory_manager", "end": END},
    )

    # Supervisor routing to target specialized agent or general chatbot
    builder.add_conditional_edges(
        "memory_manager",
        supervisor_router,
        {
            "market_data_agent": "market_data_agent",
            "econometrics_agent": "econometrics_agent",
            "risk_volatility_agent": "risk_volatility_agent",
            "portfolio_optimizer_agent": "portfolio_optimizer_agent",
            "network_graph_rag_agent": "network_graph_rag_agent",
            "technical_analysis_agent": "technical_analysis_agent",
            "governance_audit_agent": "governance_audit_agent",
            "stress_testing_agent": "stress_testing_agent",
            "explainability_agent": "explainability_agent",
            "data_sentinel_agent": "data_sentinel_agent",
            "chatbot": "chatbot",
        },
    )

    # All agents route to tool interceptor or end
    all_agent_nodes = [
        "chatbot", "market_data_agent", "econometrics_agent",
        "risk_volatility_agent", "portfolio_optimizer_agent",
        "network_graph_rag_agent", "technical_analysis_agent",
        "governance_audit_agent", "stress_testing_agent",
        "explainability_agent", "data_sentinel_agent",
    ]
    for agent_name in all_agent_nodes:
        builder.add_conditional_edges(
            agent_name,
            _route_after_agent,
            {"tool_interceptor": "tool_interceptor", "end": END},
        )

    builder.add_conditional_edges(
        "tool_interceptor",
        _route_after_interceptor,
        {
            "tools": "tools",
            "chatbot": "chatbot",
            "market_data_agent": "market_data_agent",
            "econometrics_agent": "econometrics_agent",
            "risk_volatility_agent": "risk_volatility_agent",
            "portfolio_optimizer_agent": "portfolio_optimizer_agent",
            "network_graph_rag_agent": "network_graph_rag_agent",
            "technical_analysis_agent": "technical_analysis_agent",
            "governance_audit_agent": "governance_audit_agent",
            "stress_testing_agent": "stress_testing_agent",
            "explainability_agent": "explainability_agent",
            "data_sentinel_agent": "data_sentinel_agent",
            "finalize_governance": "finalize_governance",
            "end": END,
        },
    )

    builder.add_edge("tools", "capture_tool_state")
    builder.add_conditional_edges(
        "capture_tool_state",
        _route_after_tool,
        {
            "finalize_governance": "finalize_governance",
            "chatbot": "chatbot",
            "market_data_agent": "market_data_agent",
            "econometrics_agent": "econometrics_agent",
            "risk_volatility_agent": "risk_volatility_agent",
            "portfolio_optimizer_agent": "portfolio_optimizer_agent",
            "network_graph_rag_agent": "network_graph_rag_agent",
            "technical_analysis_agent": "technical_analysis_agent",
            "governance_audit_agent": "governance_audit_agent",
            "stress_testing_agent": "stress_testing_agent",
            "explainability_agent": "explainability_agent",
            "data_sentinel_agent": "data_sentinel_agent",
        },
    )
    builder.add_edge("finalize_governance", END)
    return builder


class LazyCompiledAssistant:
    """Compile LangGraph only when chat is actually invoked."""

    def __init__(self, *, streaming: bool = False) -> None:
        self.streaming = streaming
        self._compiled = None
        self._lock = threading.Lock()

    def _get(self):
        if self._compiled is None:
            with self._lock:
                if self._compiled is None:
                    checkpointer = MemorySaver() if self.streaming else get_checkpointer()
                    self._compiled = _build_graph().compile(checkpointer=checkpointer)
        return self._compiled

    def invoke(self, *args, **kwargs):
        return self._get().invoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        return self._get().stream(*args, **kwargs)

    def astream_events(self, *args, **kwargs):
        return self._get().astream_events(*args, **kwargs)


portfolio_assistant = LazyCompiledAssistant()
streaming_portfolio_assistant = LazyCompiledAssistant(streaming=True)
logger.info("Conversational Multi-Agent Supervisor configured with lazy memory.")
