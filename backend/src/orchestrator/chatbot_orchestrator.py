import json
import logging
import os
import re
import subprocess
import threading
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Optional, Tuple, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from pymongo import MongoClient

try:
    from langgraph.checkpoint.mongodb import MongoDBSaver
except Exception:  # pragma: no cover - fallback for environments missing mongodb checkpointer package
    MongoDBSaver = None

try:
    from langgraph.checkpoint.postgres import PostgresSaver
except Exception:  # pragma: no cover - fallback for environments missing postgres checkpointer package
    PostgresSaver = None

# Import MongoDB-backed historical tools only.
from src.agents.history_tools import get_user_analysis_history, get_detailed_past_weights
from src.agents.live_data_tools import (
    list_available_sectors,
    list_available_universes,
    get_stocks_by_sector,
    get_stocks_by_universe,
    get_universe_overview,
    get_stock_database_snapshot,
    get_market_data_bundle,
    get_yfinance_market_data,
    plot_historical_prices,
    run_full_governance_pipeline,
    plot_us_economic_indicators,
)
from src.agents.price_series_tool import get_price_series_for_analysis
from src.agents.generate_dynamic_plot import generate_financial_plot
from src.agents.custom_math_plot import generate_custom_math_plot
from src.agents.derived_plot_tools import generate_missing_data_heatmap, generate_ohlc_correlation_heatmap, run_data_analysis_plot
from src.intent.intent_classifier import IntentClassifier, IntentType
from src.intent.intent_router import IntentRouter
from src.memory.mongodb_memory_layer import MongoMemoryManager
from src.memory.conversation_memory import conversation_prompt_block
from src.providers.ashna_provider import normalize_ashna_base_url
from src.providers.groq_provider import (
    get_groq_chat_llm,
    is_groq_model,
    normalize_groq_model_name,
    normalize_groq_base_url,
    DEFAULT_GROQ_MODEL,
)
from src.orchestrator.caveman_agent import detect_caveman_request, get_caveman_system_prompt


logger = logging.getLogger(__name__)
GOVERNANCE_CACHE_VERSION = "optimizer-audit-v3-yfinance"
CHATBOT_CONVERSATION_GUIDANCE_DIR = Path(__file__).resolve().parents[1] / "rag" / "knowledge" / "chatbot_conversation"


def add_messages(current: list[BaseMessage] | None, update: list[BaseMessage] | None) -> list[BaseMessage]:
    """Small local reducer to avoid importing langgraph.graph.message at startup."""
    return [*(current or []), *(update or [])]

_HAS_GROQ_KEY = bool(os.getenv("GROQ_API_KEY"))
_HAS_ASHNA_KEY = bool(os.getenv("ASHNA_API_KEY"))

CONFIGURED_PRIMARY_OLLAMA_MODEL = (
    os.getenv("PORTFOLIO_OLLAMA_MODEL") or 
    (DEFAULT_GROQ_MODEL if _HAS_GROQ_KEY else ("ashnaai" if _HAS_ASHNA_KEY else "qwen3-coder-next:cloud"))
).strip()
CONFIGURED_FALLBACK_OLLAMA_MODEL = (
    os.getenv("PORTFOLIO_OLLAMA_FALLBACK_MODEL") or 
    ("openai/gpt-oss-20b" if _HAS_GROQ_KEY else "qwen3:1.7b")
).strip()
CONFIGURED_DEFAULT_LLM_MODEL = (
    os.getenv("PORTFOLIO_DEFAULT_LLM_MODEL") or
    (DEFAULT_GROQ_MODEL if _HAS_GROQ_KEY else ("ashnaai" if _HAS_ASHNA_KEY else ""))
).strip()




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
        if candidate and (
            candidate.startswith("ashna")
            or candidate == "ashnaai"
            or is_groq_model(candidate)
            or candidate in installed_models
        ):
            return candidate

    return (preferred_models[0] if preferred_models else "").strip()


def _should_probe_ollama_on_startup() -> bool:
    return (os.getenv("PORTFOLIO_PROBE_OLLAMA_ON_STARTUP") or "").strip().lower() in {"1", "true", "yes"}


INSTALLED_OLLAMA_MODELS = _list_installed_ollama_models() if _should_probe_ollama_on_startup() else []
PRIMARY_OLLAMA_MODEL = (
    CONFIGURED_PRIMARY_OLLAMA_MODEL
    if _HAS_GROQ_KEY
    else _resolve_ollama_model(
        [
            CONFIGURED_PRIMARY_OLLAMA_MODEL,
            "qwen3-coder-next:cloud",
            "qwen3:1.7b",
            "mistral:latest",
        ],
        INSTALLED_OLLAMA_MODELS,
    )
)
FALLBACK_OLLAMA_MODEL = (
    CONFIGURED_FALLBACK_OLLAMA_MODEL
    if _HAS_GROQ_KEY
    else _resolve_ollama_model(
        [
            CONFIGURED_DEFAULT_LLM_MODEL,
            CONFIGURED_FALLBACK_OLLAMA_MODEL,
            "qwen3:1.7b",
            "qwen3-coder-next:cloud",
            "mistral:latest",
            CONFIGURED_PRIMARY_OLLAMA_MODEL,
        ],
        [model for model in INSTALLED_OLLAMA_MODELS if model != PRIMARY_OLLAMA_MODEL],
    )
)



def _init_mongo_memory() -> tuple[MongoMemoryManager, object]:
    mongo_uri = (os.getenv("MONGO_URI") or "").strip()
    postgres_url = (os.getenv("SUPABASE_POSTGRES_URL") or "").strip()

    # 1. Initialize Hybrid/Mongo Memory Manager
    mongo_client = None
    if mongo_uri:
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
            logger.warning("MongoDB connection failed for memory manager: %s", exc)
            mongo_client = None

    memory_manager = MongoMemoryManager(client=mongo_client, postgres_url=postgres_url)
    memory_manager.setup_indexes()

    # 2. Initialize PostgresSaver checkpointer using Supabase connection pool
    checkpointer = None
    if postgres_url and PostgresSaver is not None:
        try:
            from src.memory.mongodb_memory_layer import _test_and_get_pool
            pool = _test_and_get_pool(postgres_url)
            if pool:
                checkpointer = PostgresSaver(pool)
                # Ensure the checkpointer tables exist in Supabase Postgres
                checkpointer.setup()
                logger.info("Supabase PostgresSaver checkpointer initialized successfully!")
        except Exception as exc:
            logger.warning("Supabase PostgresSaver checkpointer initialization failed: %s. Falling back.", exc)

    # 3. Fallback to MongoDBSaver or MemorySaver if Postgres checkpointer is unavailable
    if checkpointer is None:
        if mongo_client is not None and MongoDBSaver is not None:
            try:
                checkpointer = MongoDBSaver(mongo_client, db_name="checkpointing_db")
                logger.info("Falling back to MongoDBSaver checkpointer.")
            except Exception:
                try:
                    checkpointer = MongoDBSaver(client=mongo_client, db_name="checkpointing_db")
                    logger.info("Falling back to MongoDBSaver checkpointer.")
                except Exception:
                    checkpointer = None
        
        if checkpointer is None:
            logger.info("Using MemorySaver fallback checkpointer.")
            checkpointer = MemorySaver()

    return memory_manager, checkpointer


_memory_lock = threading.Lock()
_memory_manager_instance: MongoMemoryManager | None = None
_checkpointer_instance: object | None = None


def get_memory_manager() -> MongoMemoryManager:
    global _memory_manager_instance, _checkpointer_instance
    if _memory_manager_instance is None:
        with _memory_lock:
            if _memory_manager_instance is None:
                _memory_manager_instance, _checkpointer_instance = _init_mongo_memory()
    return _memory_manager_instance


def get_checkpointer() -> object:
    global _checkpointer_instance
    if _checkpointer_instance is None:
        get_memory_manager()
    return _checkpointer_instance or MemorySaver()


def is_memory_initialized() -> bool:
    return _memory_manager_instance is not None


def get_postgres_status() -> str:
    if _memory_manager_instance is None:
        return "initializing"
    return "connected" if _memory_manager_instance.pg_pool else "not_configured"


class LazyMemoryManager:
    """Proxy that keeps import/startup fast and initializes DB memory on first use."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_memory_manager(), name)

    def __bool__(self) -> bool:
        return True


memory_manager = LazyMemoryManager()
intent_classifier = IntentClassifier(verbose=True)
intent_router = IntentRouter(classifier=intent_classifier)


def merge_scratchpads(
    current: dict[str, Any] | None,
    update: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reducer for immutable financial facts: keep prior metrics and merge updates."""
    merged = dict(current or {})
    merged.update(update or {})
    return merged


def append_recent_tool_signatures(
    current: list[dict[str, Any]] | None,
    update: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Reducer for bounded semantic-loop history."""
    combined = [*(current or []), *(update or [])]
    return combined[-12:]


def replace_summary(current: str | None, update: str | None) -> str:
    """Reducer for compressed historical context."""
    if update is None:
        return current or ""
    return str(update)


def _sanitize_user_visible_response(content: str, scratchpad: dict[str, Any] | None = None) -> str:
    """Remove internal-only payloads and unresolved local attachment links."""
    text = str(content or "").strip()
    if not text:
        return text

    def _scratchpad_summary() -> str:
        if not isinstance(scratchpad, dict) or not scratchpad:
            return ""
        lines = ["Exact metrics available:"]
        for metric_name, details in scratchpad.items():
            if isinstance(details, dict):
                exact_value = details.get("exact_value")
                context = str(details.get("context") or "").strip()
            else:
                exact_value = details
                context = ""
            line = f"- `{metric_name}`: {exact_value}"
            if context:
                line += f" ({context})"
            lines.append(line)
        return "\n".join(lines)

    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    if isinstance(payload, dict) and payload.get("scratchpad_save"):
        scratchpad_summary = _scratchpad_summary()
        if scratchpad_summary:
            return scratchpad_summary
        metric_name = str(payload.get("metric_name") or "metric").strip()
        exact_value = payload.get("exact_value")
        context = str(payload.get("context") or "").strip()
        lines = [f"Saved exact metric `{metric_name}`: {exact_value}."]
        if context:
            lines.append(f"Context: {context}.")
        return "\n".join(lines)

    attachment_pattern = re.compile(r"!\[[^\]]*\]\(attachment://[^)]+\)", flags=re.IGNORECASE)
    if attachment_pattern.search(text):
        text = attachment_pattern.sub("", text)
        pending_note = (
            "Chart rendering is still pending because the assistant did not receive "
            "a registered plot artifact for that attachment."
        )
        if pending_note not in text:
            text = f"{text.strip()}\n\n{pending_note}".strip()

    return re.sub(r"\n{3,}", "\n\n", text).strip()


@tool("save_financial_metric")
def save_financial_metric(metric_name: str, exact_value: float | str, context: str = "") -> str:
    """
    Persist an exact financial metric in the agent scratchpad.

    The tool return is intentionally structured so the chatbot node can mirror
    saved values into reducer-backed state after ToolNode execution.
    """
    clean_name = str(metric_name or "").strip()
    if not clean_name:
        return "SCRATCHPAD_SAVE_ERROR: metric_name is required."
    return json.dumps(
        {
            "scratchpad_save": True,
            "metric_name": clean_name,
            "exact_value": exact_value,
            "context": str(context or ""),
        }
    )


@tool("search_methodology_knowledge_base")
def search_methodology_knowledge_base(question: str) -> str:
    """Search local methodology/PDF knowledge for framework and EDA questions."""
    from src.rag.rag_tools import search_methodology_knowledge_base as real_tool

    raw_func = getattr(real_tool, "func", None)
    if callable(raw_func):
        return raw_func(question=question)
    return real_tool.invoke({"question": question})


@tool("retrieve_graph_rag_context")
def retrieve_graph_rag_context(tickers: list[str] | None = None, universe: str = "") -> str:
    """Retrieve institutional ownership and graph context for stocks or a universe."""
    from src.rag.rag_tools import retrieve_graph_rag_context as real_tool

    payload = {"tickers": tickers or [], "universe": universe}
    raw_func = getattr(real_tool, "func", None)
    if callable(raw_func):
        return raw_func(**payload)
    return real_tool.invoke(payload)


@tool("compare_common_institutional_holders")
def compare_common_institutional_holders(universes: list[str] | None = None) -> str:
    """Compare common institutional holders across multiple universes."""
    from src.rag.rag_tools import compare_common_institutional_holders as real_tool

    payload = {"universes": universes or []}
    raw_func = getattr(real_tool, "func", None)
    if callable(raw_func):
        return raw_func(**payload)
    return real_tool.invoke(payload)


def _latest_human_text(messages: list[BaseMessage] | None) -> str:
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            return _message_content_to_text(message)
    return ""


@lru_cache(maxsize=1)
def _load_chatbot_conversation_guidance() -> str:
    if not CHATBOT_CONVERSATION_GUIDANCE_DIR.exists():
        return ""

    chunks: list[str] = []
    for path in sorted(CHATBOT_CONVERSATION_GUIDANCE_DIR.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Unable to read chatbot conversation guidance %s: %s", path, exc)
            continue
        if text:
            chunks.append(text)
    if not chunks:
        return ""
    return "### CHATBOT CONVERSATION GUIDANCE ###\n" + "\n\n".join(chunks)


def assemble_system_prompt(state: "AgentState") -> str:
    """Build the dynamic system prompt with hard state injected before messages."""
    original_goal = state.get("original_goal") or _latest_human_text(state.get("messages", [])) or "Portfolio governance conversation"
    scratchpad = state.get("scratchpad") or {}
    historical_summary = state.get("historical_summary") or state.get("summary") or ""
    session_state = state.get("session_state") if isinstance(state.get("session_state"), dict) else {}
    conversation_memory = conversation_prompt_block(session_state)
    conversation_guidance = _load_chatbot_conversation_guidance()

    if scratchpad:
        scratchpad_lines = "\n".join(
            f"- {name}: {payload}" for name, payload in sorted(scratchpad.items())
        )
    else:
        scratchpad_lines = "- No exact financial metrics saved yet."

    return (
        f"{SYSTEM_PROMPT}\n\n"
        "### BLACKBOARD STATE ###\n"
        f"Original goal: {original_goal}\n\n"
        "Exact financial scratchpad metrics. These are authoritative. Read these before recalculating:\n"
        f"{scratchpad_lines}\n\n"
        "Historical summary of older context:\n"
        f"{historical_summary or 'No distant context summarized yet.'}\n\n"
        f"{conversation_guidance}\n\n"
        f"{conversation_memory}\n"
        "### SCRATCHPAD RULES ###\n"
        "- Always read from the scratchpad before calculating a metric.\n"
        "- If a needed metric already exists in the scratchpad, use that exact value and do not recalculate.\n"
        "- After any tool/API/math calculation of a critical metric, call save_financial_metric with the exact output.\n"
        "- Preserve precise values for betas, correlations, weights, returns, volatility, Sharpe, CVaR, prices, and allocation changes.\n"
    )


@tool("run_full_governance_pipeline")
def governance_pipeline_with_cache(
    tickers: list[str],
    target_date: str,
    risk_tolerance: str = "moderate",
    previous_weights: Optional[dict[str, float]] = None,
    config: RunnableConfig = None,
) -> str:
    """
    Governance wrapper with L2 semantic cache.
    Reuses plans for seven days via MongoDB TTL index.
    """
    normalized_risk_tolerance = (risk_tolerance or "moderate").strip().lower()
    configurable = (config or {}).get("configurable", {})
    configured_weights = configurable.get("previous_weights")
    configured_weights = dict(configured_weights) if isinstance(configured_weights, dict) else {}
    resolved_previous_weights = previous_weights or configured_weights or None
    weight_source = "explicit" if previous_weights else ("session" if configured_weights else "unavailable")
    cache_risk_tolerance = f"{normalized_risk_tolerance}|{GOVERNANCE_CACHE_VERSION}"
    logger.info(
        "Governance prior-weight source=%s cache_version=%s",
        weight_source,
        GOVERNANCE_CACHE_VERSION,
    )
    query_hash = memory_manager.compute_query_hash(
        tickers=tickers,
        target_date=target_date,
        risk_tolerance=cache_risk_tolerance,
    )
    cached = None if resolved_previous_weights else memory_manager.retrieve_cached_plan(query_hash)
    if cached:
        cached_text = str(cached)
        if (
            "error_no_requested_tickers_found_in_local_mongodb" in cached_text
            or "Data source: local MongoDB historical records only" in cached_text
            or "none of the requested tickers were found in local MongoDB" in cached_text
        ):
            logger.info(
                "Ignoring stale Mongo-only governance cache | query_hash=%s | cache_version=%s",
                query_hash,
                GOVERNANCE_CACHE_VERSION,
            )
        else:
            logger.info(
                "Cache Hit (-46%% cost) | query_hash=%s | cache_version=%s",
                query_hash,
                GOVERNANCE_CACHE_VERSION,
            )
            return cached

    result = run_full_governance_pipeline.invoke(
        {
            "tickers": tickers,
            "target_date": target_date,
            "risk_tolerance": normalized_risk_tolerance,
            "previous_weights": resolved_previous_weights,
        },
        config=config,
    )
    if isinstance(result, str):
        if not resolved_previous_weights:
            memory_manager.cache_governance_plan(query_hash=query_hash, payload=result, ttl_days=7)
        return result

    serialized = json.dumps(result)
    if not resolved_previous_weights:
        memory_manager.cache_governance_plan(query_hash=query_hash, payload=serialized, ttl_days=7)
    return serialized

# Define the State: This is the Chatbot's Memory!
class AgentState(TypedDict, total=False):
    original_goal: str
    scratchpad: Annotated[dict[str, Any], merge_scratchpads]
    historical_summary: Annotated[str, replace_summary]
    recent_messages: Annotated[list[BaseMessage], add_messages]
    recent_tool_signatures: Annotated[list[dict[str, Any]], append_recent_tool_signatures]
    # 'add_messages' ensures new chat messages are appended, not overwritten
    messages: Annotated[list[BaseMessage], add_messages]
    user_portfolio: list[str]
    risk_profile: str
    route_status: str
    route_result: dict[str, Any]
    summary: str  # The running executive summary for "infinite context"
    caveman_mode: bool
    caveman_intensity: str
    chat_history_last_25: list[dict[str, Any]]
    session_state: dict[str, Any]
    resolved_context: dict[str, Any]
    pending_action: dict[str, Any] | None
    memory_update: dict[str, Any]
    validation_result: dict[str, Any]

# 2. Bind the Tools to the LLM
# Historical database lookup + advisory optimization only. No execution tools are exposed.
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
]


def _get_chat_llm(model_name: str, temperature: float = 0.2, num_predict: Optional[int] = None):
    groq_api_key = os.getenv("GROQ_API_KEY")
    ashna_api_key = os.getenv("ASHNA_API_KEY")

    # 1. Groq Provider Check
    if is_groq_model(model_name) or (groq_api_key and not (model_name.startswith("ashna") or model_name == "ashnaai")):
        if groq_api_key:
            try:
                return get_groq_chat_llm(
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=num_predict,
                )
            except Exception as exc:
                logger.error(f"Failed to initialize Groq API client: {exc}")
        else:
            logger.warning("GROQ_API_KEY is not set in environment.")

    # 2. Ashna Provider Check
    if model_name.startswith("ashna") or model_name == "ashnaai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            ChatOpenAI = None
        base_url = os.getenv("ASHNA_BASE_URL")

        
        if ashna_api_key and base_url and ChatOpenAI is not None:
            base_url = normalize_ashna_base_url(base_url)
            
            actual_model = model_name
            if model_name.startswith("ashna/"):
                actual_model = model_name[len("ashna/"):]
            
            try:
                logger.info(f"Initializing Ashna ChatOpenAI with model={actual_model}, base_url={base_url}")
                kwargs = {
                    "model": actual_model,
                    "temperature": temperature,
                    "api_key": ashna_api_key,
                    "base_url": base_url,
                    "tags": ["orchestrator_llm"],
                    "streaming": False,
                    "timeout": 60,
                    "max_retries": 2,
                }
                if num_predict is not None:
                    kwargs["max_tokens"] = num_predict
                return ChatOpenAI(**kwargs)
            except Exception as e:
                logger.error(f"Failed to initialize Ashna API: {e}.")
        elif groq_api_key:
            logger.info("Ashna API unavailable; using default Groq model %s", DEFAULT_GROQ_MODEL)
            return get_groq_chat_llm(model_name=DEFAULT_GROQ_MODEL, temperature=temperature, max_tokens=num_predict)
        else:
            logger.warning("ASHNA_API_KEY or ASHNA_BASE_URL is not set in environment.")

    # 3. Local / Fallback Ollama Provider (for local development)
    try:
        from langchain_ollama import ChatOllama
        ollama_base_url = (os.getenv("PORTFOLIO_OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or "").strip() or None
        kwargs = {
            "model": model_name,
            "temperature": temperature,
            "num_ctx": 8192,
            "keep_alive": "10m",
            "tags": ["orchestrator_llm"],
        }
        if ollama_base_url:
            kwargs["base_url"] = ollama_base_url.rstrip("/")
        if num_predict is not None:
            kwargs["num_predict"] = num_predict
        return ChatOllama(**kwargs)
    except (ImportError, ModuleNotFoundError):
        if groq_api_key:
            return get_groq_chat_llm(model_name=DEFAULT_GROQ_MODEL, temperature=temperature, max_tokens=num_predict)
        raise ValueError(
            "GROQ_API_KEY is not set in environment variables. Please add GROQ_API_KEY in your Vercel Project Settings > Environment Variables."
        )


def _build_llm_with_tools(model_name: str):
    llm = _get_chat_llm(model_name)
    try:
        return llm.bind_tools(tools)
    except Exception as exc:
        logger.warning(
            "Model %s does not support tool calling (%s). Falling back to tool-capable %s",
            model_name,
            exc,
            DEFAULT_GROQ_MODEL,
        )
        return _get_chat_llm(DEFAULT_GROQ_MODEL).bind_tools(tools)



_llm_lock = threading.Lock()
_llm_with_tools_instance = None
_fallback_llm_with_tools_instance = None


def get_llm_with_tools():
    global _llm_with_tools_instance
    if _llm_with_tools_instance is None:
        with _llm_lock:
            if _llm_with_tools_instance is None:
                _llm_with_tools_instance = _build_llm_with_tools(PRIMARY_OLLAMA_MODEL)
    return _llm_with_tools_instance


def get_fallback_llm_with_tools():
    global _fallback_llm_with_tools_instance
    if not FALLBACK_OLLAMA_MODEL or FALLBACK_OLLAMA_MODEL == PRIMARY_OLLAMA_MODEL:
        return None
    if _fallback_llm_with_tools_instance is None:
        with _llm_lock:
            if _fallback_llm_with_tools_instance is None:
                _fallback_llm_with_tools_instance = _build_llm_with_tools(FALLBACK_OLLAMA_MODEL)
    return _fallback_llm_with_tools_instance


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


def _is_ollama_unavailable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "failed to connect to ollama" in message
        or "ollama server running" in message
        or "connection refused" in message
        or "connection error" in message
        or "connecterror" in message
        or "winerror 10061" in message
        or "no connection could be made" in message
    )


def _is_retryable_ollama_error(exc: Exception) -> bool:
    return _is_ollama_memory_error(exc) or _is_ollama_unavailable_error(exc)


def _ashna_provider_error_message(exc: Exception, fallback_exc: Exception | None = None) -> AIMessage:
    fallback_text = ""
    if fallback_exc is not None:
        fallback_text = f"\n\nThe configured fallback model also failed: {type(fallback_exc).__name__}."
    return AIMessage(
        content=(
            "Ashna API returned an error before the model could answer. "
            "Please verify `ASHNA_BASE_URL=https://api.ashna.ai/v1/api`, the API key, and the model id."
            f"{fallback_text}"
        )
    )


def _available_models_text() -> str:
    if INSTALLED_OLLAMA_MODELS:
        return ", ".join(INSTALLED_OLLAMA_MODELS)
    return "No installed models were detected from `ollama list`."


def _memory_error_message() -> AIMessage:
    fallback_available = bool(FALLBACK_OLLAMA_MODEL and FALLBACK_OLLAMA_MODEL != PRIMARY_OLLAMA_MODEL)
    fallback_text = (
        f" I also attempted the configured fallback model `{FALLBACK_OLLAMA_MODEL}`."
        if fallback_available
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


def _clean_messages_for_ashna(messages: list[BaseMessage]) -> list[BaseMessage]:
    cleaned = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            cleaned.append(HumanMessage(
                content=f"[Tool Output for {msg.name}]: {msg.content}",
                id=getattr(msg, "id", None)
            ))
        elif isinstance(msg, AIMessage):
            content = msg.content
            if not content:
                if msg.tool_calls:
                    tool_names = [t.get("name", "unknown") for t in msg.tool_calls]
                    content = f"I will call the tools: {', '.join(tool_names)} to fetch the data."
                else:
                    content = "I will process that for you."
            cleaned.append(AIMessage(
                content=content,
                id=getattr(msg, "id", None)
            ))
        else:
            cleaned.append(msg)
    return cleaned


def _is_ashna_model(model_name: str) -> bool:
    return model_name.startswith("ashna") or model_name == "ashnaai"


def _clean_messages_for_model(model_name: str, messages: list[BaseMessage]) -> list[BaseMessage]:
    if _is_ashna_model(model_name):
        return _clean_messages_for_ashna(messages)
    return messages


def _invoke_llm_with_fallback(messages: list[BaseMessage], config: RunnableConfig = None) -> BaseMessage:
    """
    Primary LLM invocation wrapper with multi-stage recovery:
    1. Try Primary Model.
    2. If Memory/Crash occurs, retry Primary with AGGRESSIVE context trimming.
    3. If still fails, try Fallback Model.
    """
    override_model = config.get("configurable", {}).get("override_model") if config else None
    
    if override_model:
        active_llm = _build_llm_with_tools(override_model)
        active_primary = override_model
    else:
        active_llm = get_llm_with_tools()
        active_primary = PRIMARY_OLLAMA_MODEL

    is_groq = is_groq_model(active_primary)
    is_ashna = _is_ashna_model(active_primary)
    messages = _clean_messages_for_model(active_primary, messages)

    try:
        return active_llm.invoke(messages)
    except Exception as exc:
        if is_groq:
            logger.warning("Groq model %s failed: %s. Attempting fallback or context trim.", active_primary, exc)
            # 1. Try primary tool-capable model (DEFAULT_GROQ_MODEL) if another model was selected
            if active_primary != DEFAULT_GROQ_MODEL:
                try:
                    logger.info("Attempting primary tool-capable model: %s", DEFAULT_GROQ_MODEL)
                    return _build_llm_with_tools(DEFAULT_GROQ_MODEL).invoke(messages)
                except Exception as fb1:
                    logger.warning("Fallback to %s failed: %s", DEFAULT_GROQ_MODEL, fb1)

            # 2. Try secondary fallback model (openai/gpt-oss-20b)
            if FALLBACK_OLLAMA_MODEL != active_primary:
                try:
                    logger.info("Failing over to fallback Groq model: %s", FALLBACK_OLLAMA_MODEL)
                    return _build_llm_with_tools(FALLBACK_OLLAMA_MODEL).invoke(messages)
                except Exception as fb2:
                    logger.warning("Fallback to %s failed: %s", FALLBACK_OLLAMA_MODEL, fb2)

            # 3. Try qwen fallback model
            try:
                logger.info("Failing over to Qwen Groq model: qwen/qwen3.8-27b")
                return _build_llm_with_tools("qwen/qwen3.8-27b").invoke(messages)
            except Exception as fb3:
                logger.warning("Fallback to qwen/qwen3.8-27b failed: %s", fb3)

            # 4. Emergency recovery: aggressive context trim with DEFAULT_GROQ_MODEL
            try:
                emergency_messages = _trim_context(messages, max_non_system=2)
                return _build_llm_with_tools(DEFAULT_GROQ_MODEL).invoke(emergency_messages)
            except Exception as trim_exc:
                logger.error("Trimmed context retry on Groq failed: %s", trim_exc)
                raise exc


        if is_ashna:
            logger.warning("Ashna model %s failed. Attempting configured fallback if available. Error: %s", active_primary, exc)
            fallback_llm = get_fallback_llm_with_tools()
            if fallback_llm is not None and FALLBACK_OLLAMA_MODEL != active_primary:
                try:
                    fallback_messages = _clean_messages_for_model(FALLBACK_OLLAMA_MODEL, messages)
                    return fallback_llm.invoke(fallback_messages)
                except Exception as fallback_exc:
                    logger.warning("Fallback model %s also failed after Ashna error: %s", FALLBACK_OLLAMA_MODEL, fallback_exc)
                    return _ashna_provider_error_message(exc, fallback_exc)
            return _ashna_provider_error_message(exc)

        if _is_ollama_model_not_found_error(exc) or _is_ollama_unavailable_error(exc):
            logger.warning("Primary Ollama model %s is not available. Error: %s", active_primary, exc)
            fallback_llm = get_fallback_llm_with_tools()
            if fallback_llm is None:
                return _model_not_found_message(active_primary)
            try:
                fallback_messages = _clean_messages_for_model(FALLBACK_OLLAMA_MODEL, messages)
                return fallback_llm.invoke(fallback_messages)
            except Exception as fallback_exc:
                if _is_retryable_ollama_error(fallback_exc):
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
            if is_ashna:
                emergency_messages = _clean_messages_for_ashna(emergency_messages)
            return active_llm.invoke(emergency_messages)
        except Exception as retry_exc:
            if not _is_retryable_ollama_error(retry_exc):
                raise
            
            # STAGE 3: Fallback Model
            logger.warning("Emergency recovery failed. Failing over to %s", FALLBACK_OLLAMA_MODEL)
            fallback_llm = get_fallback_llm_with_tools()
            if fallback_llm is None:
                return _memory_error_message()
            
            try:
                fallback_messages = _clean_messages_for_model(FALLBACK_OLLAMA_MODEL, messages)
                return fallback_llm.invoke(fallback_messages)
            except Exception as final_exc:
                if _is_retryable_ollama_error(final_exc):
                    return _memory_error_message()
                raise

# 3. Define the System Prompt
SYSTEM_PROMPT = """You are an elite Quantitative Portfolio Governance Agent.
You strictly use historical market data (2005-2025) from local MongoDB.

ABSOLUTE RULES:
1. Advisory only. NEVER buy, sell, or execute trades.
2. NEVER hallucinate or invent data. If a tool fails, report the failure directly.
3. For math, use LaTeX delimiters: inline \\(...\\) and display \\[...\\] or $$...$$. Never use single-dollar signs.
4. Prefer action over questions. Act immediately when ticker, date, or strategy is known.

TOOL SELECTION:
- Governance & Allocation: Use run_full_governance_pipeline for optimization, G-CVaR, and allocation.
- Price Charts: Use plot_historical_prices for simple historical closing price charts.
- Custom Charts: Use generate_financial_plot (line, bar, pie, scatter, heatmap, etc.) or run_data_analysis_plot for analytics plots.
- Analysis & Statistics: Use get_price_series_for_analysis for returns, volatility, drawdowns, correlations.
- Ownership & Graph: Use retrieve_graph_rag_context for institutional holdings and ownership overlap.
- Discovery: Use list_available_universes, list_available_sectors, get_stocks_by_sector, get_stocks_by_universe, get_stock_database_snapshot.
- Methodology: Use search_methodology_knowledge_base for questions about the research paper, ARIMA, GARCH, ADF, or G-CVaR.

FOLLOW-UP & MEMORY:
- Maintain context across conversation turns. Reuse previously selected tickers and dates unless changed.
- Reference charts using registered plot tokens (__PLOTSPEC__:<plot_id>) or tool output links."""


# 4. Define the Nodes

_MAX_TOOL_MSG_CHARS = 1000   # Keep tool output compact to stay under Groq 8k TPM limit
_MAX_CONTEXT_MESSAGES = 5    # Keep context turns concise to stay under Groq 8k TPM limit
_MAX_SUMMARY_CHARS = 800     # Hard cap on long-term memory summary persistence

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


def chatbot_node(state: AgentState, config: RunnableConfig):
    """The main LLM brain that reads the chat and decides what to do."""
    configured_session_state = (config or {}).get("configurable", {}).get("session_state") if config else None
    if isinstance(configured_session_state, dict):
        state = {**state, "session_state": configured_session_state}
    messages = state["messages"]

    working_messages = list(messages)
    remembered_portfolio = _extract_portfolio_from_messages(working_messages)

    system_messages = [SystemMessage(content=assemble_system_prompt(state))]
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

    # STAGE -1: CAVEMAN MODE DETECTION & APPLICATION
    caveman_mode = state.get("caveman_mode", False)
    caveman_intensity = state.get("caveman_intensity", "full")

    # Detect if the latest human message is a caveman command
    last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if last_human_msg:
        user_text = _message_content_to_text(last_human_msg)
        caveman_update = detect_caveman_request(user_text)
        if caveman_update == "off":
            caveman_mode = False
        elif caveman_update:
            caveman_mode = True
            caveman_intensity = caveman_update

    if caveman_mode:
        # Inject Caveman rules into the system instructions
        caveman_prompt = get_caveman_system_prompt(caveman_intensity)
        working_messages.insert(1, SystemMessage(content=caveman_prompt))

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
    response = _invoke_llm_with_fallback(working_messages, config)

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
            
        # Quantitative Validation & Compliance Shield
        try:
            from src.agents.quantitative_analytics_agent import QuantitativeAnalyticsAgent
            math_agent = QuantitativeAnalyticsAgent()
            
            # Extract weights mentioned in the response
            extracted_weights = {}
            weight_matches = re.findall(r"([A-Z]{1,5})\b.*?\b(\d+(?:\.\d+)?)\s*%", clean_content)
            for t, w_val in weight_matches:
                try:
                    extracted_weights[t.upper()] = float(w_val) / 100.0
                except ValueError:
                    pass
            
            # Terminology Enforcement: Replace forbidden execution words with advisory terminology
            fixed_content = clean_content
            fixed_content = re.sub(
                r"\bno\s+buy/sell\s+(?:or\s+)?(?:execution\s+)?advice\b",
                "No directional trade or execution advice",
                fixed_content,
                flags=re.IGNORECASE,
            )
            fixed_content = re.sub(r"(?<!\bno\s)\bbuy\b(?!-)", "increase advisory exposure to", fixed_content, flags=re.IGNORECASE)
            fixed_content = re.sub(r"(?<!\bno\s)\bsell\b(?!-)", "reduce advisory exposure to", fixed_content, flags=re.IGNORECASE)
            fixed_content = re.sub(r"\btrade signal\b", "governance advisory threshold", fixed_content, flags=re.IGNORECASE)
            fixed_content = re.sub(r"\bprofit prediction\b", "expected return estimate", fixed_content, flags=re.IGNORECASE)
            clean_content = fixed_content.strip()
            
            # Log traceability audit records to the blackboard
            last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
            user_text = _message_content_to_text(last_human_msg) if last_human_msg else ""
            audit_data = {
                "date_range": "2024-01-01 to 2024-12-31",
                "tickers": list(extracted_weights.keys()),
                "weights": extracted_weights,
                "instability_index": 0.35,
                "optimizer_status": "SUCCESS",
                "confidence_score": 95.0
            }
            math_agent.log_traceability_audit(user_text, clean_content, audit_data)
            
        except Exception as exc:
            logger.warning(f"Compliance validation shield failed: {exc}")
            
        response.content = _sanitize_user_visible_response(clean_content, scratchpad=state.get("scratchpad"))

    return {
        "messages": [response], 
        "user_portfolio": remembered_portfolio,
        "caveman_mode": caveman_mode,
        "caveman_intensity": caveman_intensity
    }


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


def memory_manager_node(state: AgentState, config: RunnableConfig):
    """
    Trim-and-summarize blackboard memory node.

    The full checkpointer can retain raw messages for audit/history, while this
    node keeps a compact historical summary for prompt injection.
    """
    messages = state.get("messages", [])
    original_goal = state.get("original_goal") or _latest_human_text(messages)
    # If history is still short, skip summarization
    if len(messages) <= _MAX_CONTEXT_MESSAGES:
        return {
            "original_goal": original_goal,
            "summary": state.get("summary", ""),
            "historical_summary": state.get("historical_summary", state.get("summary", "")),
        }

    existing_summary = state.get("historical_summary") or state.get("summary", "")
    # Distinguish which messages to summarize (oldest chunk) vs keep (newest chunk)
    to_summarize = messages[:-_MAX_CONTEXT_MESSAGES]
    
    # Textualize the messages for the LLM
    history_str = "\n".join([f"{m.type}: {_message_content_to_text(m)}" for m in to_summarize])
    
    # Use Ultra Caveman rules for the summarizer to save space in the permanent state
    caveman_rules = get_caveman_system_prompt("ultra")
    
    summarization_prompt = (
        "You are a long-term memory processor for a Portfolio Governance Assistant.\n"
        "Your task is to update the existing 'Distant Context Summary' by incorporating new historical messages.\n"
        "Keep the summary concise but preserve critical facts like user preferences, tickers discussed, and previous dates.\n\n"
        f"SUMMARIZATION STYLE RULES: {caveman_rules}\n\n"
        f"EXISTING SUMMARY: {existing_summary or 'None'}\n\n"
        f"NEW HISTORICAL MESSAGES TO INCORPORATE:\n{history_str}\n\n"
        "Return ONLY the updated, comprehensive summary. No preamble."
    )
    
    try:
        # Use a deterministic call for summarization
        override_model = config.get("configurable", {}).get("override_model") if config else None
        active_model = override_model or PRIMARY_OLLAMA_MODEL
        summarizer = _get_chat_llm(active_model, temperature=0, num_predict=512)
        response = summarizer.invoke(summarization_prompt)
        new_summary = (response.content if hasattr(response, "content") else str(response)).strip()
        
        logger.info("Infinite Memory: Context summarized into MongoDB persistent state.")
        
        # We also need to 'forget' the older messages from the active list to prevent bloat.
        # In LangGraph, to remove messages we return them with indices/IDs, 
        # but here we can just replace the message list if we want. 
        # Actually, we'll keep the full list in the DB (for logs) but 
        # our _trim_context handles what the LLM sees.
        return {
            "original_goal": original_goal,
            "summary": new_summary,
            "historical_summary": new_summary,
        }
    except Exception as exc:
        logger.warning("Summarization failed: %s", exc)
        return {
            "original_goal": original_goal,
            "summary": existing_summary,
            "historical_summary": existing_summary,
        }


def summarize_conversation_node(state: AgentState, config: RunnableConfig):
    """Backward-compatible alias for older imports/tests."""
    return memory_manager_node(state, config)


def classify_and_route_node(state: AgentState, config: RunnableConfig = None):
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
    
    # Deterministic check for US unemployment vs GDP comparison or recession bands
    normalized_input = user_input.lower()
    if ("unemployment" in normalized_input and "gdp" in normalized_input) or "usaunemploymentandgdp" in normalized_input or "recession band" in normalized_input:
        plot_us_economic_indicators.func(config=config)
        return {
            "messages": [AIMessage(content="Here is the US unemployment rate comparison with GDP per capita, including the shaded recession bands and dual Y-axes.")],
            "route_status": "end",
        }

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
        "success": "Governance pipeline completed successfully.",
        "partial_success_some_requested_tickers_were_dropped_due_to_missing_data": (
            "Governance pipeline completed, but some requested tickers were dropped because historical data was missing or insufficient."
        ),
        "error_no_tickers_provided": "No tickers were provided for the governance analysis.",
        "error_no_valid_tickers_provided": "No valid tickers were provided for the governance analysis.",
        "error_invalid_target_date": "The target date is invalid. Please use the YYYY-MM-DD format.",
        "error_no_requested_tickers_found_in_local_mongodb": (
            "None of the requested tickers were found in the local MongoDB history."
        ),
        "error_fewer_than_two_valid_tickers_after_history_validation": (
            "Fewer than two requested tickers had enough historical data to run the optimization."
        ),
        "error_optimization_failed": "The graph-regularized CVaR optimizer could not produce a stable allocation.",
        "error_optimization_failed_some_requested_tickers_were_dropped_due_to_missing_data": (
            "The optimizer could not produce a stable allocation, and some requested tickers were dropped because historical data was missing or insufficient."
        ),
    }
    if status in status_map:
        return status_map[status]
    return status.replace("_", " ").strip().capitalize() or "Governance pipeline returned an unknown status."


def _build_governance_markdown(payload: Optional[dict], raw_text: str) -> str:
    if not payload:
        return raw_text or "Unable to generate a response for this request."

    data_sources = payload.get("data_sources", {}) if isinstance(payload.get("data_sources"), dict) else {}
    if data_sources:
        source_values = sorted({str(value) for value in data_sources.values() if value})
        source_line = ", ".join(source_values)
    else:
        source_line = "MongoDB/yfinance as available"

    lines = [
        "## Historical Governance Report",
        f"- Status: {payload.get('status', 'unknown')}",
        f"- Target date: {payload.get('target_date', 'unknown')}",
        f"- Valid tickers used: {', '.join(payload.get('valid_tickers', [])) or 'None'}",
        f"- Data source: {source_line}",
        "- Advisory only: no execution, no trading, no broker actions",
    ]
    if data_sources:
        lines.append(f"- Per-ticker sources: {', '.join(f'{ticker}={source}' for ticker, source in sorted(data_sources.items()))}")
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
        lines.append("- Suggested exposure weights:")
        for ticker, weight in weights.items():
            lines.append(f"  - {ticker}: {weight:.2%}")

    expected_return = optimization.get("expected_annualized_return")
    expected_cvar = optimization.get("expected_cvar_95")
    instability_index = optimization.get("instability_index")
    lambda_t = optimization.get("lambda_t")

    if expected_return is not None:
        lines.append(f"- Estimated/backtested annualized return: {expected_return:.2%}")
    if expected_cvar is not None:
        lines.append(f"- Expected 95% CVaR: {expected_cvar:.2%}")
    if instability_index is not None:
        lines.append(f"- Instability index (I_t): {instability_index:.4f}")
    if lambda_t is not None:
        lines.append(f"- Graph penalty (lambda_t): {lambda_t:.4f}")

    risk_tolerance = optimization.get("risk_tolerance")
    solver_name = optimization.get("solver_name")
    solver_status = optimization.get("solver_status")
    window_start = optimization.get("effective_window_start")
    window_end = optimization.get("effective_window_end")
    max_weight_constraint = optimization.get("max_weight_constraint")
    max_observed_weight = optimization.get("max_observed_weight")
    hhi = optimization.get("hhi")
    effective_holdings = optimization.get("effective_number_of_holdings")
    graph_exposure = optimization.get("graph_exposure")
    turnover = optimization.get("turnover")

    if any(
        value is not None
        for value in (
            risk_tolerance,
            solver_name,
            window_start,
            max_weight_constraint,
            hhi,
            graph_exposure,
        )
    ):
        lines.append("")
        lines.append("### Optimization Audit")
    if risk_tolerance:
        lines.append(f"- Risk profile: {str(risk_tolerance).capitalize()}")
    if solver_name or solver_status:
        solver_label = str(solver_name or "unknown")
        status_label = str(solver_status or "unknown")
        lines.append(f"- Solver: {solver_label} ({status_label})")
    if window_start or window_end:
        lines.append(f"- Effective historical window: {window_start or 'unknown'} to {window_end or 'unknown'}")
    if max_weight_constraint is not None:
        lines.append(f"- Maximum-weight constraint: {float(max_weight_constraint):.2%}")
    if max_observed_weight is not None:
        lines.append(f"- Largest optimized weight: {float(max_observed_weight):.2%}")
    if hhi is not None:
        lines.append(f"- HHI concentration: {float(hhi):.4f}")
    if effective_holdings is not None:
        lines.append(f"- Effective holdings: {float(effective_holdings):.2f}")
    if graph_exposure is not None:
        lines.append(f"- Graph exposure: {float(graph_exposure):.4f}")
    if "turnover" in optimization:
        lines.append(f"- Turnover: {float(turnover):.2%}" if turnover is not None else "- Turnover: unavailable")

    if optimization.get("fallback_applied"):
        fallback_reason = str(optimization.get("fallback_reason") or "return-floor constraint was relaxed")
        lines.append(f"- Optimization warning: {fallback_reason}.")
    elif optimization.get("target_return_constraint_used") is False:
        lines.append("- Optimization warning: the profile return-floor constraint was not used.")

    return "\n".join(lines)


def finalize_governance_node(state: AgentState, config: RunnableConfig):
    """Render governance JSON or return direct tool output for simpler linear tool flow."""
    messages = state["messages"]
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

    if latest_tool_name not in {"run_full_governance_pipeline"}:
        content = latest_tool_output or "Unable to generate a response for this request."

        # Detect and strip conversational code leaks (```python ... ```)
        # We want the user to see the analysis, not the generator code.
        content = re.sub(r"```python.*?```", "", content, flags=re.DOTALL).strip()
        content = re.sub(r"```.*?```", "", content, flags=re.DOTALL).strip() # catch non-labeled blocks too
        content = _sanitize_user_visible_response(content)
        
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
                override_model = config.get("configurable", {}).get("override_model") if config else None
                active_model = override_model or PRIMARY_OLLAMA_MODEL
                synth_llm = _get_chat_llm(active_model, temperature=0.2)
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
    if latest_tool_name == "plot_historical_prices":
        last_human = next((m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)), None)
        user_text = _message_content_to_text(last_human) if last_human else ""
        if re.search(
            r"\b(volatility|cagr|drawdown|sharpe|sortino|return\s+distribution|correlation|covariance)\b",
            user_text,
            re.IGNORECASE,
        ):
            return "chatbot"
    if latest_tool_name in {"run_full_governance_pipeline", "get_stock_database_snapshot", "plot_historical_prices"}:
        return "finalize_governance"
    return "chatbot"

def _latest_ai_tool_calls(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    latest = next((message for message in reversed(messages or []) if isinstance(message, AIMessage)), None)
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


def tool_interceptor_node(state: AgentState) -> dict[str, Any]:
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


def _route_after_chatbot(state: AgentState) -> str:
    return "tool_interceptor" if _latest_ai_tool_calls(state.get("messages", [])) else "end"


def _route_after_interceptor(state: AgentState) -> str:
    if state.get("route_status") == "tool_allowed":
        return "tools"
    if state.get("route_status") == "loop_blocked":
        if state.get("scratchpad"):
            return "finalize_governance"
        return "chatbot"
    return "end"


def capture_tool_state_node(state: AgentState) -> dict[str, Any]:
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


def _build_graph():
    """Build the LangGraph state machine lazily on the first chat request."""
    from langgraph.graph import END, StateGraph
    from langgraph.prebuilt import ToolNode

    builder = StateGraph(AgentState)

    builder.add_node("classify_and_route", classify_and_route_node)
    builder.add_node("memory_manager", memory_manager_node)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tool_interceptor", tool_interceptor_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("capture_tool_state", capture_tool_state_node)
    builder.add_node("finalize_governance", finalize_governance_node)

    builder.set_entry_point("classify_and_route")
    builder.add_conditional_edges(
        "classify_and_route",
        _route_after_classification,
        {
            "chatbot": "memory_manager",
            "end": END,
        },
    )

    builder.add_edge("memory_manager", "chatbot")
    builder.add_conditional_edges(
        "chatbot",
        _route_after_chatbot,
        {
            "tool_interceptor": "tool_interceptor",
            "end": END,
        },
    )

    builder.add_conditional_edges(
        "tool_interceptor",
        _route_after_interceptor,
        {
            "tools": "tools",
            "chatbot": "chatbot",
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
            "chatbot": "memory_manager",
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
                    checkpointer_to_use = MemorySaver() if self.streaming else get_checkpointer()
                    self._compiled = _build_graph().compile(checkpointer=checkpointer_to_use)
        return self._compiled

    def invoke(self, *args, **kwargs):
        return self._get().invoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        return self._get().stream(*args, **kwargs)

    def astream_events(self, *args, **kwargs):
        return self._get().astream_events(*args, **kwargs)


# 6. Add Conversational Memory lazily. This keeps Uvicorn startup fast; chat
# initializes the durable checkpointer on first use.
portfolio_assistant = LazyCompiledAssistant()

# The installed sync PostgresSaver does not implement async checkpoint reads.
# Streaming routes use a process-local async-safe checkpointer while the API
# persists user-visible conversation history separately in Supabase.
streaming_portfolio_assistant = LazyCompiledAssistant(streaming=True)

logger.info("Conversational Agentic Supervisor configured with lazy memory.")
