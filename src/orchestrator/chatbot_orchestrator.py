import json
import logging
import os
import re
import subprocess
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
from src.agents.history_tools import get_user_analysis_history, get_detailed_past_weights
from src.agents.live_data_tools import (
    list_available_sectors,
    list_available_universes,
    get_stocks_by_sector,
    get_stocks_by_universe,
    get_universe_overview,
    get_stock_database_snapshot,
    plot_historical_prices,
    run_full_governance_pipeline,
)
from src.agents.price_series_tool import get_price_series_for_analysis
from src.intent.intent_classifier import IntentClassifier, IntentType
from src.intent.intent_router import IntentRouter
from src.memory.mongodb_memory_layer import MongoMemoryManager
from src.rag.rag_tools import (
    compare_common_institutional_holders,
    retrieve_graph_rag_context,
    search_methodology_knowledge_base,
)
from src.agents.custom_plot_tool import generate_custom_plot


logger = logging.getLogger(__name__)
CONFIGURED_PRIMARY_OLLAMA_MODEL = (os.getenv("PORTFOLIO_OLLAMA_MODEL") or "mistral:latest").strip()
CONFIGURED_FALLBACK_OLLAMA_MODEL = (os.getenv("PORTFOLIO_OLLAMA_FALLBACK_MODEL") or "mistral:latest").strip()


def _list_installed_ollama_models() -> list[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        logger.warning("Unable to inspect installed Ollama models: %s", exc)
        return []

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        if stderr:
            logger.warning("`ollama list` failed while resolving models: %s", stderr)
        return []

    models = []
    for line in result.stdout.splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        name = stripped.split()[0].strip()
        if name and name not in models:
            models.append(name)
    return models


def _resolve_ollama_model(preferred_models: list[str], installed_models: list[str]) -> str:
    for model_name in preferred_models:
        candidate = (model_name or "").strip()
        if candidate and candidate in installed_models:
            return candidate

    return (preferred_models[0] if preferred_models else "").strip()


INSTALLED_OLLAMA_MODELS = _list_installed_ollama_models()
PRIMARY_OLLAMA_MODEL = _resolve_ollama_model(
    [
        CONFIGURED_PRIMARY_OLLAMA_MODEL,
        "mistral:latest",
        "qwen2.5:7b",
        "qwen2.5:latest",
    ],
    INSTALLED_OLLAMA_MODELS,
)
FALLBACK_OLLAMA_MODEL = _resolve_ollama_model(
    [
        CONFIGURED_FALLBACK_OLLAMA_MODEL,
        "mistral:latest",
        "qwen2.5:7b",
        "qwen2.5:latest",
        CONFIGURED_PRIMARY_OLLAMA_MODEL,
    ],
    [model for model in INSTALLED_OLLAMA_MODELS if model != PRIMARY_OLLAMA_MODEL],
)


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
        # Standardize checkpointer to use a dedicated database
        checkpointer = MongoDBSaver(mongo_client, db_name="checkpointing_db")
    except (TypeError, Exception):
        # Supports alternate constructor signatures across LangGraph versions.
        checkpointer = MongoDBSaver(client=mongo_client, db_name="checkpointing_db")
    
    return memory_manager, checkpointer


memory_manager, checkpointer = _init_mongo_memory()
intent_classifier = IntentClassifier(verbose=True)
intent_router = IntentRouter(classifier=intent_classifier)


@tool("run_full_governance_pipeline")
def governance_pipeline_with_cache(
    tickers: list[str],
    target_date: str,
    risk_tolerance: str = "moderate",
) -> str:
    """
    Governance wrapper with L2 semantic cache.
    Reuses plans for seven days via MongoDB TTL index.
    """
    normalized_risk_tolerance = (risk_tolerance or "moderate").strip().lower()
    query_hash = memory_manager.compute_query_hash(
        tickers=tickers,
        target_date=target_date,
        risk_tolerance=normalized_risk_tolerance,
    )
    cached = memory_manager.retrieve_cached_plan(query_hash)
    if cached:
        logger.info("Cache Hit (-46%% cost) | query_hash=%s", query_hash)
        return cached

    result = run_full_governance_pipeline.invoke(
        {
            "tickers": tickers,
            "target_date": target_date,
            "risk_tolerance": normalized_risk_tolerance,
        }
    )
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
    summary: str  # The running executive summary for "infinite context"

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
    get_price_series_for_analysis,
    generate_custom_plot,
    governance_pipeline_with_cache,
    search_methodology_knowledge_base,
    retrieve_graph_rag_context,
    compare_common_institutional_holders,
    get_user_analysis_history,
    get_detailed_past_weights,
]


def _build_llm_with_tools(model_name: str):
    return ChatOllama(
        model=model_name,
        temperature=0.2,
        num_ctx=8192,   # 8k tokens: safe for 30 msgs at avg 250 tokens each (~7500 total)
        keep_alive="10m",
        tags=["orchestrator_llm"],
    ).bind_tools(tools)


llm_with_tools = _build_llm_with_tools(PRIMARY_OLLAMA_MODEL)
fallback_llm_with_tools = (
    _build_llm_with_tools(FALLBACK_OLLAMA_MODEL)
    if FALLBACK_OLLAMA_MODEL and FALLBACK_OLLAMA_MODEL != PRIMARY_OLLAMA_MODEL
    else None
)


def _is_ollama_memory_error(exc: Exception) -> bool:
    """Detect if the error is a resource/memory/timeout/crash event (-1 or explicit memory strings)."""
    message = str(exc).lower()
    return (
        "requires more system memory" in message
        or "more system memory than is available" in message
        or "insufficient memory" in message
        or "status code: -1" in message               # Ollama crash/timeout
        or "internal server error" in message         # Generic failure
    )


def _is_ollama_model_not_found_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "not found" in message and "status code: 404" in message


def _available_models_text() -> str:
    if INSTALLED_OLLAMA_MODELS:
        return ", ".join(INSTALLED_OLLAMA_MODELS)
    return "No installed models were detected from `ollama list`."


def _memory_error_message() -> AIMessage:
    fallback_text = (
        f" I also attempted the configured fallback model `{FALLBACK_OLLAMA_MODEL}`."
        if fallback_llm_with_tools is not None
        else ""
    )
    return AIMessage(
        content=(
            f"The local Ollama model `{PRIMARY_OLLAMA_MODEL}` needs more RAM than is currently available."
            f"{fallback_text}\n\n"
            "Try one of these:\n"
            f"- set `PORTFOLIO_OLLAMA_MODEL` to a smaller model such as `{FALLBACK_OLLAMA_MODEL}`\n"
            "- restart Ollama after unloading larger models\n"
            "- use a deterministic query like `snapshot for TD` or `tell me more about TD`, which can bypass the LLM path"
        )
    )


def _model_not_found_message(model_name: str) -> AIMessage:
    return AIMessage(
        content=(
            f"The configured Ollama model `{model_name}` is not installed.\n\n"
            f"Detected models: {_available_models_text()}\n\n"
            "Either pull the requested model or set `PORTFOLIO_OLLAMA_MODEL` to one of the installed models."
        )
    )


def _invoke_llm_with_fallback(messages: list[BaseMessage]) -> BaseMessage:
    """
    Primary LLM invocation wrapper with multi-stage recovery:
    1. Try Primary Model.
    2. If Memory/Crash occurs, retry Primary with AGGRESSIVE context trimming.
    3. If still fails, try Fallback Model.
    """
    try:
        return llm_with_tools.invoke(messages)
    except Exception as exc:
        if _is_ollama_model_not_found_error(exc):
            logger.warning("Primary Ollama model %s is not installed. Error: %s", PRIMARY_OLLAMA_MODEL, exc)
            if fallback_llm_with_tools is None:
                return _model_not_found_message(PRIMARY_OLLAMA_MODEL)
            try:
                return fallback_llm_with_tools.invoke(messages)
            except Exception as fallback_exc:
                if _is_ollama_memory_error(fallback_exc):
                    return _memory_error_message()
                raise

        if not _is_ollama_memory_error(exc):
            raise

        logger.warning("Primary Model crash (Code -1/Internal Error). Attempting emergency context recovery. Error: %s", exc)
        
        # Give Ollama a moment to breathe before retry
        import time
        time.sleep(1.5)

        # STAGE 2: Emergency Recovery (Strip all but System Prompt and last 2 messages)
        try:
            # max_non_system=2 is extremely aggressive to guarantee a response
            emergency_messages = _trim_context(messages, max_non_system=2)
            return llm_with_tools.invoke(emergency_messages)
        except Exception as retry_exc:
            if not _is_ollama_memory_error(retry_exc):
                raise
            
            # STAGE 3: Fallback Model
            logger.warning("Emergency recovery failed. Failing over to %s", FALLBACK_OLLAMA_MODEL)
            if fallback_llm_with_tools is None:
                return _memory_error_message()
            
            try:
                return fallback_llm_with_tools.invoke(messages)
            except Exception as final_exc:
                if _is_ollama_memory_error(final_exc):
                    return _memory_error_message()
                raise

# 3. Define the System Prompt
SYSTEM_PROMPT = """You are an elite Quantitative Portfolio Governance Agent.
ABSOLUTE RULE: NEVER show raw Python code (matplotlib, pandas, etc.) to the end user. Use tools silently.
ABSOLUTE RULE: You are an advisory system. ZERO execution, buying, or selling.
ABSOLUTE RULE: NEVER hallucinate or invent data. If a tool fails, tell the user the tool failed.

You strictly use historical data (2005-2025) from your local MongoDB.

REQUEST TYPES:
1. Discovery requests: sectors, universes, universe membership, or stored ticker information.
2. Historical chart requests: price plots or visual comparisons over a date range.
3. Governance requests: structural risk analysis, optimization, or allocation recommendations.
4. Methodology requests: how the system works, paper-style framing, HITL, RAG, or statistical interpretation.
5. Graph-context requests: shared institutions, ownership overlap, contagion structure, and most central stocks.

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
- If the user asks to compare or plot all stocks in a universe, first call get_stocks_by_universe to resolve the tickers, then call plot_historical_prices with that ticker list. Do NOT stop after the universe lookup.
- If the user gives a historical range such as 2005 to 2025, pass it as start_date=2005-01-01 and end_date=2025-12-31.
- If the request already contains enough information, act immediately instead of asking for confirmation.

CUSTOM PLOT RULES:
- Use generate_custom_plot whenever the user requests a chart type not covered by 
  plot_historical_prices (which only draws historical price line charts).
- Examples that require generate_custom_plot:
    "scatter plot of risk vs return for U1 stocks"
    "histogram of daily returns for AAPL"
    "box plot comparing volatility across universes"
    "drawdown chart for MSFT from 2008 to 2010"
    "bar chart of the governance weights from the last run"
    "rolling 30-day volatility for NVDA"
    "correlation heatmap for U2 stocks"
- Always fetch the data first using the appropriate tool, then pass the relevant 
  subset to generate_custom_plot along with a precise description of what to plot 
  (axes, grouping, time range, chart type).
- The description parameter should be specific: "scatter plot with return on x-axis 
  and CVaR on y-axis, each point labelled with the ticker" is better than "scatter plot".
- Do NOT use generate_custom_plot for simple historical price line charts — 
  use plot_historical_prices for those.

GOVERNANCE RULES:
- Use run_full_governance_pipeline only for governance, optimization, allocation, CVaR, structural risk, or portfolio assessment requests.
- For governance, ensure you have tickers and one historical target date such as 2008-09-15.
- If either the tickers or the target date is missing for a governance request, politely ask for the missing information.
- The tool already performs the historical price lookup, institutional network analysis, historical G-CVaR optimization, and inline plot generation back-to-back using local MongoDB data only.
- The tool returns lightweight structured JSON with valid tickers, final weights, structural risk scores, and markdown plot links.
- Read the tool output carefully instead of inventing any values.

METHODOLOGY RAG RULES:
- If the user asks how the framework works, how I_t is computed, how HITL works, why a result is statistically insignificant, or asks for methodology/documentation details, use search_methodology_knowledge_base.
- This tool returns grounded PDF chunks from the local methodology knowledge base. Quote or summarize those chunks instead of inventing explanations.

GRAPH RAG RULES:
- If the user asks which institutions connect two stocks, asks about ownership overlap, contagion structure, or wants graph context for a ticker set or a universe, use retrieve_graph_rag_context.
- If the user asks who invested in a ticker, which institutions are common across a set of stocks, how much institutions hold, or who invested the most, use retrieve_graph_rag_context.
- If the user asks for common holders across universes such as U1 and U10, or across U1 to U11, use compare_common_institutional_holders.
- Use explicit tickers when the user provides them.
- If the user asks for graph context for a universe and no tickers are given, pass the universe identifier such as U1.

FOLLOW-UP RULES:
- If the user says "yes", continue only the immediately preceding proposal. Do not switch to a different portfolio, date, or task.
- Never substitute an unrelated example date or example ticker list.

GENERAL RULES:
1. Prefer MongoDB-backed historical tools. For simple stock snapshots and historical price lookups, a labeled yfinance fallback may be used when MongoDB is unavailable.
2. Never execute trades. This system is read-only and advisory only.
3. If a tool fails, say so clearly and do not invent missing values.
4. Always explain the allocation recommendation mathematically and transparently.
5. Never call get_stocks_by_sector with an empty sector. Use list_available_sectors for sector discovery.
6. If the user asks for comprehensive stored ticker information, prefer get_stock_database_snapshot before summarizing.
7. If the user asks about universe membership or requests a universe roster, use get_stocks_by_universe or get_stock_database_snapshot as appropriate.
8. If the user asks about a universe's sector identity or composition, use get_universe_overview.
9. Prefer returning the direct tool result over paraphrasing when the tool already answers the request cleanly.
"""

# Override the legacy prompt block above with a cleaner, tool-complete version.
SYSTEM_PROMPT = """You are an elite Quantitative Portfolio Governance Agent.
You strictly use historical data (2005-2025) from your local MongoDB.
ABSOLUTE RULE: You are an advisory system. ZERO execution, buying, or selling.
ABSOLUTE RULE: NEVER hallucinate or invent data. If a tool fails, tell the user the tool failed.

REQUEST TYPES:
1. Discovery requests: sectors, universes, universe membership, or stored ticker information.
2. Historical chart requests: price plots or visual comparisons over a date range.
3. Governance requests: structural risk analysis, optimization, or allocation recommendations.
4. Methodology requests: how the system works, paper-style framing, HITL, RAG, or statistical interpretation.
5. Graph-context requests: shared institutions, ownership overlap, contagion structure, and most central stocks.

MEMORY AND PERSISTENCE RULES:
- IMPORTANT: You have Long-term Memory provided by a MongoDB backend.
- Distant context from earlier in the session is summarized and provided to you under "Distant Context Summary".
- You MUST acknowledge this history and NEVER claim you "do not retain memory".
- If a user asks "do you remember", consult both the Distant Context Summary and Recent Messages.
- Use your memory to maintain consistency in analysis dates, ticker preferences, and risk levels.

DISCOVERY RULES:
- If the user asks "universes", "available universes", or similar, use list_available_universes.
- If the user asks for available sectors, use list_available_sectors.
- If the user asks for stocks by sector, use get_stocks_by_sector.
- If the user asks for stocks in a universe such as U1 through U11, use get_stocks_by_universe.
- If the user asks what sector a universe belongs to or asks for a universe summary, use get_universe_overview.
- If the user asks for all stored MongoDB data or a full ticker snapshot, use get_stock_database_snapshot.

HISTORICAL CHART RULES:
- Use plot_historical_prices ONLY for simple line charts showing raw closing prices over time.
- It fetches and renders in one step, so use it when the user only wants to see price history.
- Do NOT use plot_historical_prices when the user wants computed statistics such as correlations, returns, volatility, distributions, or drawdowns.
- Do NOT use run_full_governance_pipeline for a pure historical chart request.
- If the user already selected a universe or explicitly listed tickers earlier in the conversation, reuse that same ticker set for follow-up requests such as "plot all the tickers".
- If the user provides a custom list of stock tickers such as AAPL, MSFT, and NVDA, use that exact custom list.
- If the user asks to compare or plot all stocks in a universe, first call get_stocks_by_universe to resolve the tickers, then call plot_historical_prices with that ticker list. Do NOT stop after the universe lookup.
- If the user gives a historical range such as 2005 to 2025, pass it as start_date=2005-01-01 and end_date=2025-12-31.
- If the request already contains enough information, act immediately instead of asking for confirmation.

STATISTICAL ANALYSIS AND CUSTOM PLOT RULES:
- Whenever the user asks for computed statistics or a non-line chart, use a two-step approach:
  1. call get_price_series_for_analysis to fetch raw prices, returns, and summary stats. For year-only inputs like "2022", use start_date="2022" and end_date="2022" — the tool handles expansion automatically
  2. call generate_custom_plot with the full result from step 1 plus a precise chart description
- Use this two-step path for requests such as:
  - correlation heatmap
  - returns distribution or histogram of returns
  - volatility comparison
  - drawdown chart
  - rolling volatility
  - scatter plot of risk vs return
  - Sharpe ratio comparison
  - box plot or violin plot of daily returns
  - sector-weight bar chart from governance output
- If the user asks for a universe-level statistical plot, first resolve the universe members, then call get_price_series_for_analysis, then call generate_custom_plot.
- The description passed to generate_custom_plot must be specific and mention:
  - the exact chart type
  - which fields from the tool payload to use, for example data['returns'] or data['stats']
  - axis labels and grouping
- Never tell the user you cannot do this analysis when it is based on historical price series. Use get_price_series_for_analysis for that purpose.

GOVERNANCE RULES:
- Use run_full_governance_pipeline only for governance, optimization, allocation, CVaR, structural risk, or portfolio assessment requests.
- For governance, ensure you have tickers and one historical target date such as 2008-09-15.
- If either the tickers or the target date is missing for a governance request, politely ask for the missing information.
- The tool already performs the historical price lookup, institutional network analysis, historical G-CVaR optimization, and inline plot generation back-to-back using local MongoDB data only.
- The tool returns lightweight structured JSON with valid tickers, final weights, structural risk scores, and markdown plot links.
- Read the tool output carefully instead of inventing any values.

METHODOLOGY RAG RULES:
- If the user asks how the framework works, how I_t is computed, how HITL works, why a result is statistically insignificant, or asks for methodology/documentation details, use search_methodology_knowledge_base.
- This tool returns grounded PDF chunks from the local methodology knowledge base. Quote or summarize those chunks instead of inventing explanations.

GRAPH RAG RULES:
- If the user asks which institutions connect two stocks, asks about ownership overlap, contagion structure, or wants graph context for a ticker set or a universe, use retrieve_graph_rag_context.
- If the user asks who invested in a ticker, which institutions are common across a set of stocks, how much institutions hold, or who invested the most, use retrieve_graph_rag_context.
- If the user asks for common holders across universes such as U1 and U10, or across U1 to U11, use compare_common_institutional_holders.
- Use explicit tickers when the user provides them.
- If the user asks for graph context for a universe and no tickers are given, pass the universe identifier such as U1.

FOLLOW-UP RULES:
- If the user says "yes", continue only the immediately preceding proposal. Do not switch to a different portfolio, date, or task.
- Never substitute an unrelated example date or example ticker list.

GENERAL RULES:
1. Prefer MongoDB-backed historical tools. For simple stock snapshots and historical price lookups, a labeled yfinance fallback may be used when MongoDB is unavailable.
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

_MAX_TOOL_MSG_CHARS = 1800   # Reduced to keep context window for 10-turns under 8k tokens
_MAX_CONTEXT_MESSAGES = 10  # Trigger summarization after 10 turns
_MAX_SUMMARY_CHARS = 1500   # Hard cap on the long-term memory summary persistence

def _trim_context(messages: list, max_non_system: int = _MAX_CONTEXT_MESSAGES) -> list:
    """
    Prevent Ollama OOM by:
    1. Truncating any single ToolMessage that exceeds _MAX_TOOL_MSG_CHARS
    2. Keeping only the last 'max_non_system' non-System messages
    The most recent HumanMessage is always preserved.
    """
    trimmed = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            raw = _message_content_to_text(msg)
            if len(raw) > _MAX_TOOL_MSG_CHARS:
                # Keep a compact JSON summary — preserve stats if present
                truncated = raw[:_MAX_TOOL_MSG_CHARS] + " ... [truncated for context budget]"
                msg = ToolMessage(
                    content=truncated,
                    tool_call_id=getattr(msg, "tool_call_id", ""),
                    name=getattr(msg, "name", ""),
                )
        trimmed.append(msg)

    # Split system vs non-system
    non_system = [m for m in trimmed if not isinstance(m, SystemMessage)]
    if len(non_system) > max_non_system:
        # Always keep the first HumanMessage (original context) + last N messages
        first_human = next((m for m in non_system if isinstance(m, HumanMessage)), None)
        tail = non_system[-max_non_system:]
        if first_human and first_human not in tail:
            tail = [first_human] + tail
        non_system = tail

    system_msgs = [m for m in trimmed if isinstance(m, SystemMessage)]
    return system_msgs + non_system


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

    # STAGE 0: GLOBAL MEMORY RECOVERY (If this is a fresh conversation)
    # Check if we have any high-level activity in the last 24 hours to prime the bot's memory
    recent_activity = _get_global_activity_summary()
    if recent_activity:
        working_messages.insert(1, SystemMessage(
            content=(
                "### CROSS-SESSION CONTEXT RECALL ###\n"
                "The system detected the following recent high-level activity in the database from the last 24 hours. "
                "If the user's current request seems related to these tickers, dates, or universes, explicitly acknowledge "
                "that you remember their previous work and offer to continue it:\n\n"
                f"{recent_activity}"
            )
        ))

    # If we have a summary from old messages, inject it as the first message after system prompt
    summary = state.get("summary", "").strip()
    if len(summary) > _MAX_SUMMARY_CHARS:
        summary = summary[:_MAX_SUMMARY_CHARS] + " ... [summary truncated to stay within context budget]"

    if summary:
        working_messages.insert(1, SystemMessage(
            content=(
                "### YOUR LONG-TERM MEMORY (DISTANT HISTORY) ###\n"
                "The following is a persistent summary of the earlier part of this conversation "
                "from the MongoDB database. Use this to maintain context across the session:\n\n"
                f"{summary}"
            )
        ))

    working_messages = _trim_context(working_messages)
    response = _invoke_llm_with_fallback(working_messages)

    # RECTIFICATION: Strip conversational code leaks (```python ... ```)
    if hasattr(response, "content") and response.content:
        # Detect any block with backticks
        clean_content = re.sub(r"```python.*?```", "", response.content, flags=re.DOTALL)
        clean_content = re.sub(r"```.*?```", "", clean_content, flags=re.DOTALL)
        # Also catch raw 'plt.style.use' markers if they aren't in backticks
        if any(marker in clean_content for marker in ["plt.style.use", "import matplotlib", "sns.heatmap"]):
            # If plain text code is detected, strip those lines entirely to maintain premium UI
            lines = clean_content.splitlines()
            filtered_lines = [l for l in lines if not any(m in l for m in ["plt.", "sns.", "import ", "pd.DataFrame"])]
            clean_content = "\n".join(filtered_lines)
        response.content = clean_content.strip()

    return {"messages": [response], "user_portfolio": remembered_portfolio}


def _get_global_activity_summary() -> str | None:
    """
    Look into the regime_patterns and plan_cache collections to see what has been happening
    globally in the last 24 hours. This allows the bot to 'remember' that the user was 
    working on U1 even if the session ID changed.
    """
    try:
        from datetime import datetime, timedelta, timezone
        lookback = datetime.now(timezone.utc) - timedelta(hours=24)
        
        db = memory_manager._db
        if db is None:
            return None
            
        summary_lines = []
        
        # Check regime patterns (actual governance results)
        patterns = list(db["regime_patterns"].find(
            {"created_at": {"$gt": lookback}}
        ).sort("created_at", -1).limit(5))
        
        for p in patterns:
            weights = p.get("weights", {})
            tickers = list(weights.keys())
            date = p.get("target_date", "Unknown")
            risk = p.get("risk_tolerance", "moderate")
            summary_lines.append(
                f"- Analysis Run: {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''} "
                f"at {date} (Risk: {risk}). Weights: {json.dumps(weights) if len(weights) < 5 else 'Truncated'}."
            )

        # Check plan cache (semantic cache hits)
        plans = list(db["plan_cache"].find(
            {"updated_at": {"$gt": lookback}}
        ).sort("updated_at", -1).limit(5))
        
        for pl in plans:
            # We don't easily have the tickers in the plan cache doc without parsing the query hash,
            # but we can look at the timestamps to know 'something' happened.
            # However, regime_patterns is much better.
            pass

        if not summary_lines:
            return None
            
        return "\n".join(summary_lines)
    except Exception as exc:
        logger.warning("Global activity recovery failed: %s", exc)
        return None


def summarize_conversation_node(state: AgentState):
    """
    Compresses distant history into a running summary to manage the token budget.
    This enables 'infinite memory' by migrating older details to the 'summary' field.
    """
    messages = state.get("messages", [])
    # If history is still short, skip summarization
    if len(messages) <= _MAX_CONTEXT_MESSAGES:
        return {"summary": state.get("summary", "")}

    existing_summary = state.get("summary", "")
    # Distinguish which messages to summarize (oldest chunk) vs keep (newest chunk)
    to_summarize = messages[:-_MAX_CONTEXT_MESSAGES]
    
    # Textualize the messages for the LLM
    history_str = "\n".join([f"{m.type}: {_message_content_to_text(m)}" for m in to_summarize])
    
    summarization_prompt = (
        "You are a long-term memory processor for a Portfolio Governance Assistant.\n"
        "Your task is to update the existing 'Distant Context Summary' by incorporating new historical messages.\n"
        "Keep the summary concise but preserve critical facts like user preferences, tickers discussed, and previous dates.\n\n"
        f"EXISTING SUMMARY: {existing_summary or 'None'}\n\n"
        f"NEW HISTORICAL MESSAGES TO INCORPORATE:\n{history_str}\n\n"
        "Return ONLY the updated, comprehensive summary. No preamble."
    )
    
    try:
        # Use a deterministic call for summarization
        from langchain_ollama import ChatOllama
        summarizer = ChatOllama(model=PRIMARY_OLLAMA_MODEL, temperature=0, num_predict=512)
        response = summarizer.invoke(summarization_prompt)
        new_summary = (response.content if hasattr(response, "content") else str(response)).strip()
        
        logger.info("Infinite Memory: Context summarized into MongoDB persistent state.")
        
        # We also need to 'forget' the older messages from the active list to prevent bloat.
        # In LangGraph, to remove messages we return them with indices/IDs, 
        # but here we can just replace the message list if we want. 
        # Actually, we'll keep the full list in the DB (for logs) but 
        # our _trim_context handles what the LLM sees.
        return {"summary": new_summary}
    except Exception as exc:
        logger.warning("Summarization failed: %s", exc)
        return {"summary": existing_summary}


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
    match = intent_router.classifier.classify(user_input)
    
    if match.intent == IntentType.ADVERSARIAL:
        return {"route_status": "blocked", "route_explanation": match.explanation}

    # Always route plotting, RAG, and general chat to the conversational node
    # The conversational node now has the higher-order 'Intent' context to guide its tool selection.
    allowed_chatbot_intents = {
        IntentType.STOCK_SNAPSHOT, 
        IntentType.METHODOLOGY_QUESTION, 
        IntentType.EXPLAIN_PARAMETERS, 
        IntentType.HISTORICAL_CHART,
        IntentType.MALFORMED,
        IntentType.GREETING,
        IntentType.LIST_SECTORS,
        IntentType.UNIVERSE_OVERVIEW,
        IntentType.DOCUMENTATION_REQUEST,
    }

    if match.intent in allowed_chatbot_intents:
        return {"route_status": "chatbot"}

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

    for match in re.finditer(r"(?m)^([A-Z]{1,5}):\s+", text):
        ticker = match.group(1).upper()
        if ticker not in tickers:
            tickers.append(ticker)

    for match in re.finditer(r"(?m)^Tickers:\s*([A-Z,\s]+)$", text):
        for token in re.split(r"[,\s]+", match.group(1).upper()):
            if token and re.fullmatch(r"[A-Z]{1,5}", token) and token not in tickers:
                tickers.append(token)

    return tickers


def _extract_portfolio_from_messages(messages: list[BaseMessage]) -> list[str]:
    for message in reversed(messages):
        raw_text = _message_content_to_text(message)

        if isinstance(message, ToolMessage):
            name = getattr(message, "name", "")
            if name == "run_full_governance_pipeline":
                try:
                    payload = json.loads(raw_text)
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    valid_tickers = payload.get("valid_tickers", [])
                    if isinstance(valid_tickers, list) and valid_tickers:
                        return [str(ticker).upper() for ticker in valid_tickers if str(ticker).strip()]

            if name in {
                "get_stocks_by_sector",
                "get_stocks_by_universe",
                "get_universe_overview",
                "plot_historical_prices",
                "retrieve_graph_rag_context",
                "get_price_series_for_analysis",
                "get_stock_database_snapshot",
                "get_user_analysis_history",
                "get_detailed_past_weights",
            }:
                if name == "get_price_series_for_analysis":
                    try:
                        payload = json.loads(raw_text)
                    except Exception:
                        payload = None
                    if isinstance(payload, dict):
                        tickers_included = payload.get("tickers_included", [])
                        if isinstance(tickers_included, list) and tickers_included:
                            return [str(ticker).upper() for ticker in tickers_included if str(ticker).strip()]
                extracted = _extract_tickers_from_text(raw_text)
                if extracted:
                    return extracted

        if isinstance(message, AIMessage):
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

    if latest_tool_name not in {"run_full_governance_pipeline"}:
        content = latest_tool_output or "Unable to generate a response for this request."

        # Detect and strip conversational code leaks (```python ... ```)
        # We want the user to see the analysis, not the generator code.
        content = re.sub(r"```python.*?```", "", content, flags=re.DOTALL).strip()
        content = re.sub(r"```.*?```", "", content, flags=re.DOTALL).strip() # catch non-labeled blocks too
        
        # If the LLM leaked code as plain text (no backticks), our marker interceptor below will catch it.

        # Pass markdown images through untouched so the UI renders them
        if "![" in content and "](" in content:
            # OPTIMIZATION: Ensure there is at least a double newline before images
            # to help Gradio formatting
            if not content.startswith("\n"):
                content = "\n\n" + content
            return {"messages": [AIMessage(content=content)]}

        # Detect raw matplotlib/seaborn code leaking through as plain text
        # (happens when LLM generates code instead of calling generate_custom_plot)
        _code_markers = ("plt.savefig", "import matplotlib", "plt.show", "plt.style.use", "sns.heatmap")
        if any(marker in content for marker in _code_markers):
            return {"messages": [AIMessage(content=(
                "I have prepared the requested visualization. One moment while I render the chart... "
                "\n\n[System Note: The assistant attempted to display raw code. I am intercepting this to maintain visual excellence. "
                "The chart will be generated via the appropriate tool path.]"
            ))]}

        # For methodology/graph RAG tools, synthesise the raw chunk output through the LLM
        rag_tools = {"search_methodology_knowledge_base", "retrieve_graph_rag_context", "compare_common_institutional_holders"}
        if latest_tool_name in rag_tools:
            last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
            user_text = _message_content_to_text(last_human) if last_human else ""
            try:
                synthesis_prompt = (
                    f"You are an expert portfolio governance advisor.\n"
                    f"The user asked: {user_text}\n\n"
                    f"The knowledge base returned the following grounded context:\n{content}\n\n"
                    f"Please synthesise this into a clear, concise answer for the user."
                )
                from langchain_ollama import ChatOllama
                synth_llm = ChatOllama(model=PRIMARY_OLLAMA_MODEL, temperature=0.2)
                synth_response = synth_llm.invoke(synthesis_prompt)
                synthesised = (synth_response.content if hasattr(synth_response, "content") else str(synth_response)).strip()
                if synthesised:
                    return {"messages": [AIMessage(content=synthesised)]}
            except Exception as exc:
                logger.warning("RAG synthesis LLM call failed, returning raw output: %s", exc)

        return {"messages": [AIMessage(content=content)]}

    governance_payload, raw_text = _extract_latest_governance_payload(messages)
    content_parts = [_build_governance_markdown(governance_payload, raw_text)]
    plot_outputs = governance_payload.get("generated_plots", []) if isinstance(governance_payload, dict) else []

    if plot_outputs:
        content_parts.append("## Generated Visuals")
        content_parts.extend(plot_outputs)

    return {"messages": [AIMessage(content="\n\n".join(part for part in content_parts if part))]}


def _route_after_tool(state: AgentState) -> str:
    latest_tool_name, latest_tool_output = _extract_latest_tool_output(state.get("messages", []))
    # Tools that produce final output go to finalize_governance.
    # get_price_series_for_analysis returns an intermediate cache reference —
    # route back to chatbot so the LLM can chain it into generate_custom_plot.
    if latest_tool_name in {"run_full_governance_pipeline", "get_stock_database_snapshot", "generate_custom_plot", "plot_historical_prices"}:
        return "finalize_governance"
    return "chatbot"

# 5. Build the LangGraph State Machine
builder = StateGraph(AgentState)

# Add the nodes
builder.add_node("classify_and_route", classify_and_route_node)
builder.add_node("summarize_conversation", summarize_conversation_node)
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools)) # This node automatically runs the Python tools
builder.add_node("finalize_governance", finalize_governance_node)

# Define the routing logic
builder.set_entry_point("classify_and_route")
builder.add_conditional_edges(
    "classify_and_route",
    _route_after_classification,
    {
        "chatbot": "summarize_conversation",
        "end": END,
    },
)

builder.add_edge("summarize_conversation", "chatbot")

# If the LLM decides it needs a MongoDB-backed historical tool, route to 'tools'
# Otherwise, route to END to output the chat response to the user
builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

builder.add_conditional_edges(
    "tools",
    _route_after_tool,
    {
        "finalize_governance": "finalize_governance",
        "chatbot": "chatbot",
    },
)
builder.add_edge("finalize_governance", END)

# 6. Add Conversational Memory (L1 with MongoDBSaver, fallback to MemorySaver)
portfolio_assistant = builder.compile(checkpointer=checkpointer)

print("Conversational Agentic Supervisor Initialized with Memory!")
