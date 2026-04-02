import json
import logging
import os
import re
from typing import Annotated, Any, Optional, Tuple, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from pymongo import MongoClient

try:
    from langgraph.checkpoint.mongodb import MongoDBSaver
except Exception:  # pragma: no cover - fallback for environments missing mongodb checkpointer package
    MongoDBSaver = None

# Import MongoDB-backed historical tools only.
from src.agents.live_data_tools import (
    get_stock_database_snapshot,
    get_stocks_by_sector,
    get_stocks_by_universe,
    get_universe_overview,
    list_available_sectors,
    list_available_universes,
    plot_historical_prices,
    run_full_governance_pipeline,
)
from src.intent.intent_classifier import IntentClassifier
from src.intent.intent_router import IntentRouter
from src.memory.mongodb_memory_layer import MongoMemoryManager


logger = logging.getLogger(__name__)


def _init_mongo_memory() -> tuple[MongoMemoryManager, object]:
    mongo_uri = (os.getenv("MONGO_URI") or "").strip()

    if not mongo_uri:
        logger.warning("MONGO_URI is not set; falling back to in-memory LangGraph checkpointing.")
        return MongoMemoryManager(mongo_uri=""), MemorySaver()

    try:
        mongo_client = MongoClient(
            mongo_uri,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
            appname="agentic-ai-portfolio-governance-chatbot",
        )
        mongo_client.admin.command("ping")
    except Exception as exc:
        logger.warning("MongoDB connection failed for checkpointer; using MemorySaver. Error: %s", exc)
        return MongoMemoryManager(mongo_uri=""), MemorySaver()

    memory_manager = MongoMemoryManager(client=mongo_client)
    memory_manager.setup_indexes()

    if MongoDBSaver is None:
        logger.warning("MongoDBSaver is unavailable in this environment; using MemorySaver fallback.")
        return memory_manager, MemorySaver()

    try:
        checkpointer = MongoDBSaver(mongo_client)
    except TypeError:
        # Supports alternate constructor signatures across LangGraph versions.
        checkpointer = MongoDBSaver(client=mongo_client)

    return memory_manager, checkpointer


memory_manager, checkpointer = _init_mongo_memory()
intent_classifier = IntentClassifier(verbose=True)
intent_router = IntentRouter(classifier=intent_classifier)


@tool("run_full_governance_pipeline")
def governance_pipeline_with_cache(tickers: list[str], target_date: str) -> str:
    """
    Governance wrapper with L2 semantic cache.
    Reuses plans for seven days via MongoDB TTL index.
    """
    query_hash = memory_manager.compute_query_hash(tickers=tickers, target_date=target_date)
    cached = memory_manager.retrieve_cached_plan(query_hash)
    if cached:
        logger.info("Cache Hit (-46%% cost) | query_hash=%s", query_hash)
        return cached

    result = run_full_governance_pipeline.invoke({"tickers": tickers, "target_date": target_date})
    if isinstance(result, str):
        memory_manager.cache_governance_plan(query_hash=query_hash, payload=result, ttl_days=7)
        return result

    serialized = json.dumps(result)
    memory_manager.cache_governance_plan(query_hash=query_hash, payload=serialized, ttl_days=7)
    return serialized

# Define the State: This is the Chatbot's Memory!
class AgentState(TypedDict, total=False):
    # 'add_messages' ensures new chat messages are appended, not overwritten
    messages: Annotated[list[BaseMessage], add_messages]
    user_portfolio: list[str]
    risk_profile: str
    route_status: str
    route_result: dict[str, Any]

# 1. Initialize the LLM (Your Local Qwen 2.5)
llm = ChatOllama(model="qwen2.5:7b", temperature=0.2)

# 2. Bind the Tools to the LLM
# Historical database lookup + advisory optimization only. No execution tools are exposed.
tools = [
    list_available_sectors,
    list_available_universes,
    get_stocks_by_sector,
    get_stocks_by_universe,
    get_universe_overview,
    get_stock_database_snapshot,
    plot_historical_prices,
    governance_pipeline_with_cache,
]
llm_with_tools = llm.bind_tools(tools)

# 3. Define the System Prompt
SYSTEM_PROMPT = """You are an elite Quantitative Portfolio Governance Agent.
You strictly use historical data (2005-2025) from your local MongoDB.
ABSOLUTE RULE: You are an advisory system. ZERO execution, buying, or selling.
ABSOLUTE RULE: NEVER hallucinate or invent data. If a tool fails, tell the user the tool failed.

REQUEST TYPES:
1. Discovery requests: sectors, universes, universe membership, or stored ticker information.
2. Historical chart requests: price plots or visual comparisons over a date range.
3. Governance requests: structural risk analysis, optimization, or allocation recommendations.

DISCOVERY RULES:
- If the user asks "universes", "available universes", or similar, use list_available_universes.
- If the user asks for available sectors, use list_available_sectors.
- If the user asks for stocks by sector, use get_stocks_by_sector.
- If the user asks for stocks in a universe such as U1 through U11, use get_stocks_by_universe.
- If the user asks what sector a universe belongs to or asks for a universe summary, use get_universe_overview.
- If the user asks for all stored MongoDB data or a full ticker snapshot, use get_stock_database_snapshot.

HISTORICAL CHART RULES:
- If the user asks for a historical price chart, line plot, comparison plot, or visualization over a date range, use plot_historical_prices.
- Do NOT use run_full_governance_pipeline for a pure historical chart request.
- If the user already selected a universe or explicitly listed tickers earlier in the conversation, reuse that same ticker set for follow-up requests such as "plot all the tickers".
- If the user provides a custom list of stock tickers such as AAPL, MSFT, and NVDA, use that exact custom list.
- If the user gives a historical range such as 2005 to 2025, pass it as start_date=2005-01-01 and end_date=2025-12-31.
- If the request already contains enough information, act immediately instead of asking for confirmation.

GOVERNANCE RULES:
- Use run_full_governance_pipeline only for governance, optimization, allocation, CVaR, structural risk, or portfolio assessment requests.
- For governance, ensure you have tickers and one historical target date such as 2008-09-15.
- If either the tickers or the target date is missing for a governance request, politely ask for the missing information.
- The tool already performs the historical price lookup, institutional network analysis, historical G-CVaR optimization, and inline plot generation back-to-back using local MongoDB data only.
- The tool returns lightweight structured JSON with valid tickers, final weights, structural risk scores, and markdown plot links.
- Read the tool output carefully instead of inventing any values.

FOLLOW-UP RULES:
- If the user says "yes", continue only the immediately preceding proposal. Do not switch to a different portfolio, date, or task.
- Never substitute an unrelated example date or example ticker list.

GENERAL RULES:
1. Use only MongoDB-backed historical tools. No live internet data and no yfinance.
2. Never execute trades. This system is read-only and advisory only.
3. If a tool fails, say so clearly and do not invent missing values.
4. Always explain the allocation recommendation mathematically and transparently.
5. Never call get_stocks_by_sector with an empty sector. Use list_available_sectors for sector discovery.
6. If the user asks for comprehensive stored ticker information, prefer get_stock_database_snapshot before summarizing.
7. If the user asks about universe membership or requests a universe roster, use get_stocks_by_universe or get_stock_database_snapshot as appropriate.
8. If the user asks about a universe's sector identity or composition, use get_universe_overview.
9. Prefer returning the direct tool result over paraphrasing when the tool already answers the request cleanly.
"""

# 4. Define the Nodes
def chatbot_node(state: AgentState):
    """The main LLM brain that reads the chat and decides what to do."""
    messages = state["messages"]

    working_messages = list(messages)
    remembered_portfolio = _extract_portfolio_from_messages(working_messages)

    system_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if remembered_portfolio:
        system_messages.append(
            SystemMessage(
                content=(
                    "Conversation context: the most recent explicit portfolio in this thread is "
                    f"{', '.join(remembered_portfolio)}. Reuse it for follow-up requests like "
                    "'plot all the tickers' unless the user changes the portfolio."
                )
            )
        )

    if not working_messages or not isinstance(working_messages[0], SystemMessage):
        working_messages = system_messages + working_messages
    else:
        working_messages = system_messages + [
            message for message in working_messages if not isinstance(message, SystemMessage)
        ]

    response = llm_with_tools.invoke(working_messages)
    return {"messages": [response], "user_portfolio": remembered_portfolio}


def classify_and_route_node(state: AgentState):
    """
    Deterministic intent gate that runs before the conversational LLM.
    """
    messages = state["messages"]
    if not messages:
        return {"route_status": "chatbot"}

    latest_msg = messages[-1]
    if not isinstance(latest_msg, HumanMessage):
        return {"route_status": "chatbot"}

    user_input = _message_content_to_text(latest_msg)
    route_result = intent_router.handle(user_input)
    logger.info("Intent route selected: %s (%s)", route_result["intent"], route_result["risk_tier"])

    status = route_result.get("status")
    if status == "success":
        return {
            "messages": [AIMessage(content=str(route_result["result"]))],
            "route_status": "end",
            "route_result": route_result,
        }

    if status == "pending_governance_review":
        governance_summary = route_result.get("governance_summary", {})
        tickers = ", ".join(governance_summary.get("tickers", [])) or "None"
        universes = ", ".join(governance_summary.get("universes", [])) or "None"
        content = (
            f"Governance Request Blocked Pending Approval\n"
            f"Request ID: {route_result['request_id']}\n"
            f"Risk Tier: {route_result['risk_tier']}\n"
            f"Intent: {route_result['intent']}\n"
            f"Tickers: {tickers}\n"
            f"Universes: {universes}\n"
            f"Target date: {governance_summary.get('target_date') or 'None'}\n\n"
            f"{route_result['message']}"
        )
        return {
            "messages": [AIMessage(content=content)],
            "route_status": "end",
            "route_result": route_result,
        }

    if status == "rejected":
        return {
            "messages": [AIMessage(content=route_result["reason"])],
            "route_status": "end",
            "route_result": route_result,
        }

    return {"route_status": "chatbot", "route_result": route_result}


def _route_after_classification(state: AgentState):
    return state.get("route_status", "chatbot")


def _message_content_to_text(message_or_content) -> str:
    content = getattr(message_or_content, "content", message_or_content)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content) if content is not None else ""


def _extract_tickers_from_text(text: str) -> list[str]:
    tickers = []
    for match in re.finditer(r"(?m)(?:^|\s)-\s*([A-Z]{1,5})(?::|\s|\()", text):
        ticker = match.group(1).upper()
        if ticker not in tickers:
            tickers.append(ticker)

    for match in re.finditer(r"(?m)^Ticker:\s*([A-Z]{1,5})\b", text):
        ticker = match.group(1).upper()
        if ticker not in tickers:
            tickers.append(ticker)

    return tickers


def _extract_portfolio_from_messages(messages: list[BaseMessage]) -> list[str]:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            name = getattr(message, "name", "")
            raw_text = _message_content_to_text(message)

            if name == "run_full_governance_pipeline":
                try:
                    payload = json.loads(raw_text)
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    valid_tickers = payload.get("valid_tickers", [])
                    if isinstance(valid_tickers, list) and valid_tickers:
                        return [str(ticker).upper() for ticker in valid_tickers if str(ticker).strip()]

            if name in {"get_stocks_by_universe", "get_universe_overview", "plot_historical_prices"}:
                extracted = _extract_tickers_from_text(raw_text)
                if extracted:
                    return extracted
    return []


def _extract_latest_governance_payload(messages: list[BaseMessage]) -> Tuple[Optional[dict], str]:
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "run_full_governance_pipeline":
            raw = _message_content_to_text(message)
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed, raw
            except Exception:
                return None, raw
    return None, ""


def _extract_latest_tool_output(messages: list[BaseMessage]) -> tuple[str, str]:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            return getattr(message, "name", ""), _message_content_to_text(message)
    return "", ""


def _humanize_status(status: str) -> str:
    status_map = {
        "success": "Governance pipeline completed successfully using local MongoDB history only.",
        "partial_success_some_requested_tickers_were_dropped_due_to_missing_data": (
            "Governance pipeline completed, but some requested tickers were dropped because local historical data was missing or insufficient."
        ),
        "error_no_tickers_provided": "No tickers were provided for the governance analysis.",
        "error_no_valid_tickers_provided": "No valid tickers were provided for the governance analysis.",
        "error_invalid_target_date": "The target date is invalid. Please use the YYYY-MM-DD format.",
        "error_no_requested_tickers_found_in_local_mongodb": (
            "None of the requested tickers were found in the local MongoDB history."
        ),
        "error_fewer_than_two_valid_tickers_after_history_validation": (
            "Fewer than two requested tickers had enough local historical data to run the optimization."
        ),
        "error_optimization_failed": "The graph-regularized CVaR optimizer could not produce a stable allocation.",
        "error_optimization_failed_some_requested_tickers_were_dropped_due_to_missing_data": (
            "The optimizer could not produce a stable allocation, and some requested tickers were dropped because local historical data was missing or insufficient."
        ),
    }
    if status in status_map:
        return status_map[status]
    return status.replace("_", " ").strip().capitalize() or "Governance pipeline returned an unknown status."


def _build_governance_markdown(payload: Optional[dict], raw_text: str) -> str:
    if not payload:
        return raw_text or "Unable to generate a response for this request."

    lines = [
        "## Historical Governance Report",
        f"- Status: {payload.get('status', 'unknown')}",
        f"- Target date: {payload.get('target_date', 'unknown')}",
        f"- Valid tickers used: {', '.join(payload.get('valid_tickers', [])) or 'None'}",
        "- Data source: local MongoDB historical records only",
        "- Advisory only: no execution, no trading, no broker actions",
    ]
    lines.extend(["", _humanize_status(str(payload.get("status", "")))])

    message = payload.get("message")
    if message:
        lines.extend(["", str(message)])

    dropped = payload.get("dropped_tickers", [])
    if isinstance(dropped, list) and dropped:
        lines.append("- Dropped tickers:")
        for item in dropped:
            lines.append(
                f"  - {item.get('ticker', 'UNKNOWN')}: {item.get('reason', 'unspecified reason')}"
            )

    systemic_risk = payload.get("systemic_risk", {}) if isinstance(payload.get("systemic_risk"), dict) else {}
    method = systemic_risk.get("method")
    if method:
        lines.append(f"- Structural risk method: {method}")

    scores = systemic_risk.get("scores", {}) if isinstance(systemic_risk.get("scores"), dict) else {}
    if scores:
        lines.append("- Structural risk scores:")
        for ticker, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"  - {ticker}: {score:.4f}")

    optimization = payload.get("optimization", {}) if isinstance(payload.get("optimization"), dict) else {}
    weights = optimization.get("weights", {}) if isinstance(optimization.get("weights"), dict) else {}
    if weights:
        lines.append("- Recommended allocation weights:")
        for ticker, weight in weights.items():
            lines.append(f"  - {ticker}: {weight:.2%}")

    expected_return = optimization.get("expected_annualized_return")
    expected_cvar = optimization.get("expected_cvar_95")
    instability_index = optimization.get("instability_index")
    lambda_t = optimization.get("lambda_t")

    if expected_return is not None:
        lines.append(f"- Expected annualized return: {expected_return:.2%}")
    if expected_cvar is not None:
        lines.append(f"- Expected 95% CVaR: {expected_cvar:.2%}")
    if instability_index is not None:
        lines.append(f"- Instability index (I_t): {instability_index:.4f}")
    if lambda_t is not None:
        lines.append(f"- Graph penalty (lambda_t): {lambda_t:.4f}")

    return "\n".join(lines)


def finalize_governance_node(state: AgentState):
    """Render governance JSON or return direct tool output for simpler linear tool flow."""
    messages = state["messages"]
    if not messages:
        return {"messages": [AIMessage(content="Unable to generate a response for this request.")]}

    latest_tool_name, latest_tool_output = _extract_latest_tool_output(messages)
    if latest_tool_name != "run_full_governance_pipeline":
        content = latest_tool_output or "Unable to generate a response for this request."
        return {"messages": [AIMessage(content=content)]}

    governance_payload, raw_text = _extract_latest_governance_payload(messages)
    content_parts = [_build_governance_markdown(governance_payload, raw_text)]
    plot_outputs = governance_payload.get("generated_plots", []) if isinstance(governance_payload, dict) else []

    if plot_outputs:
        content_parts.append("## Generated Visuals")
        content_parts.extend(plot_outputs)

    return {"messages": [AIMessage(content="\n\n".join(part for part in content_parts if part))]}

# 5. Build the LangGraph State Machine
builder = StateGraph(AgentState)

# Add the nodes
builder.add_node("classify_and_route", classify_and_route_node)
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools)) # This node automatically runs the Python tools
builder.add_node("finalize_governance", finalize_governance_node)

# Define the routing logic
builder.set_entry_point("classify_and_route")
builder.add_conditional_edges(
    "classify_and_route",
    _route_after_classification,
    {
        "chatbot": "chatbot",
        "end": END,
    },
)

# If the LLM decides it needs a MongoDB-backed historical tool, route to 'tools'
# Otherwise, route to END to output the chat response to the user
builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

builder.add_edge("tools", "finalize_governance")
builder.add_edge("finalize_governance", END)

# 6. Add Conversational Memory (L1 with MongoDBSaver, fallback to MemorySaver)
portfolio_assistant = builder.compile(checkpointer=checkpointer)

print("Conversational Agentic Supervisor Initialized with Memory!")
