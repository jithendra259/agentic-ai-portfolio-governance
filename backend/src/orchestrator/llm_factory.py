"""LLM Factory & Model Invocation Layer.

Provides fast, lightweight, and resilient model resolution, context trimming,
tool binding, and multi-tier failover across Groq, Ashna AI, OmniRoute, and local Ollama.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from src.providers.ashna_provider import normalize_ashna_base_url
from src.providers.groq_provider import (
    DEFAULT_GROQ_MODEL,
    get_groq_chat_llm,
    is_groq_model,
    normalize_groq_base_url,
    normalize_groq_model_name,
)
from src.providers.omniroute_provider import get_omniroute_chat_llm, is_omniroute_model

logger = logging.getLogger(__name__)

# Constants
_MAX_TOOL_MSG_CHARS = 1000
_MAX_CONTEXT_MESSAGES = 5
_MAX_SUMMARY_CHARS = 800

_HAS_GROQ_KEY = bool(os.getenv("GROQ_API_KEY"))
_HAS_ASHNA_KEY = bool(os.getenv("ASHNA_API_KEY"))

CONFIGURED_PRIMARY_OLLAMA_MODEL = (
    os.getenv("PORTFOLIO_OLLAMA_MODEL")
    or (DEFAULT_GROQ_MODEL if _HAS_GROQ_KEY else ("ashnaai" if _HAS_ASHNA_KEY else "qwen3-coder-next:cloud"))
).strip()
CONFIGURED_FALLBACK_OLLAMA_MODEL = (
    os.getenv("PORTFOLIO_OLLAMA_FALLBACK_MODEL")
    or ("openai/gpt-oss-20b" if _HAS_GROQ_KEY else "qwen3:1.7b")
).strip()
CONFIGURED_DEFAULT_LLM_MODEL = (
    os.getenv("PORTFOLIO_DEFAULT_LLM_MODEL")
    or (DEFAULT_GROQ_MODEL if _HAS_GROQ_KEY else ("ashnaai" if _HAS_ASHNA_KEY else ""))
).strip()


def _list_installed_ollama_models() -> list[str]:
    """Probe local Ollama HTTP API (fastest, PATH-independent) then CLI."""
    try:
        base_url = (os.getenv("PORTFOLIO_OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
        req = urllib.request.Request(f"{base_url}/api/tags", headers={"User-Agent": "PortfolioGovernance/1.0"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [item.get("name") for item in data.get("models", []) if item.get("name")]
            if models:
                return models
    except Exception as exc:
        logger.debug("Ollama HTTP API check failed: %s", exc)

    ollama_bin = shutil.which("ollama")
    if not ollama_bin and os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            candidate = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
            if candidate.exists():
                ollama_bin = str(candidate)

    if not ollama_bin:
        return []

    try:
        result = subprocess.run([ollama_bin, "list"], capture_output=True, text=True, timeout=3, check=False)
        if result.returncode == 0:
            models = []
            for line in result.stdout.splitlines()[1:]:
                stripped = line.strip()
                if stripped:
                    name = stripped.split()[0].strip()
                    if name and name not in models:
                        models.append(name)
            return models
    except Exception as exc:
        logger.debug("Unable to inspect installed Ollama models via CLI: %s", exc)
        return []

    return []


def _resolve_ollama_model(preferred_models: list[str], installed_models: list[str]) -> str:
    for model_name in preferred_models:
        candidate = (model_name or "").strip()
        if candidate and (candidate in installed_models or candidate.startswith("openai/") or candidate.startswith("ashna")):
            return candidate
    return preferred_models[0] if preferred_models else "qwen3:1.7b"


def _should_probe_ollama_on_startup() -> bool:
    explicit = os.getenv("PORTFOLIO_PROBE_OLLAMA_ON_STARTUP")
    if explicit is not None:
        return explicit.strip().lower() in {"1", "true", "yes", "on"}
    return not _HAS_GROQ_KEY and not _HAS_ASHNA_KEY


INSTALLED_OLLAMA_MODELS = (
    _list_installed_ollama_models() if _should_probe_ollama_on_startup() else []
)

PRIMARY_OLLAMA_MODEL = CONFIGURED_PRIMARY_OLLAMA_MODEL
FALLBACK_OLLAMA_MODEL = CONFIGURED_FALLBACK_OLLAMA_MODEL


# =====================================================================
# Error Detection & Recovery
# =====================================================================

def _is_ollama_memory_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "requires more system memory" in message
        or "more system memory than is available" in message
        or "insufficient memory" in message
        or "status code: -1" in message
        or "internal server error" in message
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
    fallback_text = f"\n\nThe configured fallback model also failed: {type(fallback_exc).__name__}." if fallback_exc is not None else ""
    return AIMessage(
        content=(
            "Ashna API returned an error before the model could answer. "
            "Please verify `ASHNA_BASE_URL=https://api.ashna.ai/v1/api`, the API key, and the model id."
            f"{fallback_text}"
        )
    )


def _available_models_text() -> str:
    return ", ".join(INSTALLED_OLLAMA_MODELS) if INSTALLED_OLLAMA_MODELS else "No installed models detected from `ollama list`."


def _memory_error_message() -> AIMessage:
    fallback_text = f" I also attempted fallback `{FALLBACK_OLLAMA_MODEL}`." if FALLBACK_OLLAMA_MODEL != PRIMARY_OLLAMA_MODEL else ""
    return AIMessage(
        content=(
            f"The local Ollama model `{PRIMARY_OLLAMA_MODEL}` needs more RAM than is available."
            f"{fallback_text}\n\nTry switching to `{FALLBACK_OLLAMA_MODEL}` or setting `GROQ_API_KEY`."
        )
    )


def _model_not_found_message(model_name: str) -> AIMessage:
    return AIMessage(
        content=(
            f"The configured Ollama model `{model_name}` is not installed.\n\n"
            f"Detected models: {_available_models_text()}\n\n"
            "Either pull the model or set `PORTFOLIO_OLLAMA_MODEL`."
        )
    )


def _is_ashna_model(model_name: str) -> bool:
    return model_name.startswith("ashna") or model_name == "ashnaai"


def _clean_messages_for_ashna(messages: list[BaseMessage]) -> list[BaseMessage]:
    cleaned = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            cleaned.append(HumanMessage(content=f"[Tool Output for {msg.name}]: {msg.content}", id=getattr(msg, "id", None)))
        elif isinstance(msg, AIMessage):
            content = msg.content
            if not content:
                if msg.tool_calls:
                    tool_names = [t.get("name", "unknown") for t in msg.tool_calls]
                    content = f"I will call tools: {', '.join(tool_names)} to fetch the data."
                else:
                    content = "I will process that for you."
            cleaned.append(AIMessage(content=content, id=getattr(msg, "id", None)))
        else:
            cleaned.append(msg)
    return cleaned


def _clean_messages_for_model(model_name: str, messages: list[BaseMessage]) -> list[BaseMessage]:
    return _clean_messages_for_ashna(messages) if _is_ashna_model(model_name) else messages


# =====================================================================
# Context Trimming for Low TPM & Zero OOM
# =====================================================================

def _trim_context(messages: list, max_non_system: int = _MAX_CONTEXT_MESSAGES) -> list:
    """Trim tool outputs and old turns to stay safely under 8k TPM limits."""
    trimmed = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            raw = str(getattr(msg, "content", ""))
            if len(raw) > _MAX_TOOL_MSG_CHARS:
                compact = raw[:_MAX_TOOL_MSG_CHARS] + " ... [data trimmed]"
                trimmed.append(ToolMessage(content=compact, tool_call_id=getattr(msg, "tool_call_id", ""), name=getattr(msg, "name", "")))
            else:
                trimmed.append(msg)
        else:
            trimmed.append(msg)

    system_msgs = [m for m in trimmed if isinstance(m, SystemMessage)]
    non_system = [m for m in trimmed if not isinstance(m, SystemMessage)]
    if len(non_system) > max_non_system:
        recent = non_system[-max_non_system:]
        latest_human = next((m for m in reversed(non_system) if isinstance(m, HumanMessage)), None)
        if latest_human and latest_human not in recent:
            recent[0] = latest_human
        non_system = recent

    return system_msgs + non_system


# =====================================================================
# Chat Model Construction & Binding
# =====================================================================

def _get_chat_llm(model_name: str, temperature: float = 0.2, num_predict: Optional[int] = None):
    groq_api_key = os.getenv("GROQ_API_KEY")
    ashna_api_key = os.getenv("ASHNA_API_KEY")
    omniroute_api_key = os.getenv("OMNIROUTE_API_KEY")

    # 1. OmniRoute
    if is_omniroute_model(model_name) or (omniroute_api_key and model_name.startswith("omniroute/")):
        try:
            return get_omniroute_chat_llm(model_name=model_name, temperature=temperature, max_tokens=num_predict)
        except Exception as exc:
            logger.error("Failed to initialize OmniRoute client: %s", exc)

    # 2. Local Ollama
    is_ollama_requested = (
        model_name.startswith("ollama/")
        or model_name in INSTALLED_OLLAMA_MODELS
        or model_name.startswith("nemotron")
        or model_name.startswith("gpt-oss:")
    )
    if is_ollama_requested:
        try:
            from langchain_ollama import ChatOllama
            ollama_base_url = (os.getenv("PORTFOLIO_OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").strip()
            actual_model = model_name[len("ollama/"):] if model_name.startswith("ollama/") else model_name
            kwargs = {
                "model": actual_model,
                "temperature": temperature,
                "base_url": ollama_base_url.rstrip("/"),
                "num_ctx": 8192,
                "keep_alive": "10m",
                "tags": ["orchestrator_llm", "local_ollama"],
            }
            if num_predict is not None:
                kwargs["num_predict"] = num_predict
            return ChatOllama(**kwargs)
        except Exception as exc:
            logger.warning("Local Ollama initialization failed: %s. Falling back to Groq.", exc)

    # 3. Groq
    if groq_api_key:
        groq_model = DEFAULT_GROQ_MODEL if (is_ollama_requested or not model_name) else model_name
        return get_groq_chat_llm(model_name=groq_model, temperature=temperature, max_tokens=num_predict)

    # 4. Ashna AI
    if ashna_api_key:
        from langchain_openai import ChatOpenAI
        raw_base = os.getenv("ASHNA_BASE_URL", "https://api.ashna.ai")
        base_url = normalize_ashna_base_url(raw_base)
        ashna_model = os.getenv("ASHNA_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=ashna_model,
            openai_api_key=ashna_api_key,
            openai_api_base=base_url,
            temperature=temperature,
            max_tokens=num_predict,
            tags=["orchestrator_llm"],
        )

    # 5. Local Fallback Ollama
    try:
        from langchain_ollama import ChatOllama
        ollama_base_url = (os.getenv("PORTFOLIO_OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").strip()
        actual_model = model_name[len("ollama/"):] if model_name.startswith("ollama/") else model_name
        kwargs = {
            "model": actual_model,
            "temperature": temperature,
            "base_url": ollama_base_url.rstrip("/"),
            "num_ctx": 8192,
            "keep_alive": "10m",
            "tags": ["orchestrator_llm"],
        }
        if num_predict is not None:
            kwargs["num_predict"] = num_predict
        return ChatOllama(**kwargs)
    except Exception:
        if groq_api_key:
            return get_groq_chat_llm(model_name=DEFAULT_GROQ_MODEL, temperature=temperature, max_tokens=num_predict)
        raise ValueError("No LLM provider available. Set GROQ_API_KEY, ASHNA_API_KEY, or run Ollama locally.")


_llm_lock = threading.Lock()
_cached_tools: list = []


def register_global_tools(tools_list: list):
    global _cached_tools
    _cached_tools = tools_list


def _build_llm_with_tools(model_name: str, bound_tools: list = None):
    target_tools = _cached_tools if bound_tools is None else bound_tools
    llm = _get_chat_llm(model_name)
    try:
        return llm.bind_tools(target_tools)
    except Exception as exc:
        logger.warning(
            "Model %s does not support tool calling (%s). Falling back to %s",
            model_name, exc, DEFAULT_GROQ_MODEL
        )
        return _get_chat_llm(DEFAULT_GROQ_MODEL).bind_tools(target_tools)


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


def _invoke_llm_with_fallback(
    messages: list[BaseMessage],
    config: RunnableConfig = None,
    bound_tools: list = None,
) -> BaseMessage:
    """Primary LLM invocation with multi-tier failover and context trimming."""
    override_model = config.get("configurable", {}).get("override_model") if config else None

    if override_model:
        active_llm = _build_llm_with_tools(override_model, bound_tools=bound_tools)
        active_primary = override_model
    elif bound_tools is not None:
        active_llm = _build_llm_with_tools(PRIMARY_OLLAMA_MODEL, bound_tools=bound_tools)
        active_primary = PRIMARY_OLLAMA_MODEL
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
            logger.warning("Groq model %s failed: %s. Attempting fallback.", active_primary, exc)
            try:
                emergency = _trim_context(messages, max_non_system=2)
                return _build_llm_with_tools(DEFAULT_GROQ_MODEL, bound_tools=bound_tools).invoke(emergency)
            except Exception as trim_exc:
                logger.warning("Trimmed context retry on Groq failed: %s", trim_exc)

            if os.getenv("ASHNA_API_KEY") and os.getenv("ASHNA_BASE_URL"):
                try:
                    logger.info("Failing over from Groq to Ashna AI...")
                    return _build_llm_with_tools("ashnaai", bound_tools=bound_tools).invoke(emergency if 'emergency' in locals() else messages)
                except Exception as ashna_exc:
                    logger.warning("Failover to Ashna AI failed: %s", ashna_exc)

            fallback_llm = _build_llm_with_tools(FALLBACK_OLLAMA_MODEL, bound_tools=bound_tools) if (FALLBACK_OLLAMA_MODEL and bound_tools is not None) else get_fallback_llm_with_tools()
            if fallback_llm is not None:
                try:
                    logger.info("Failing over from Groq to local Ollama...")
                    return fallback_llm.invoke(emergency if 'emergency' in locals() else messages)
                except Exception as ollama_exc:
                    logger.warning("Failover to Ollama failed: %s", ollama_exc)

            raise exc

        if is_ashna:
            logger.warning("Ashna model %s failed: %s", active_primary, exc)
            fallback_llm = _build_llm_with_tools(FALLBACK_OLLAMA_MODEL, bound_tools=bound_tools) if (FALLBACK_OLLAMA_MODEL and bound_tools is not None) else get_fallback_llm_with_tools()
            if fallback_llm is not None and FALLBACK_OLLAMA_MODEL != active_primary:
                try:
                    fallback_msgs = _clean_messages_for_model(FALLBACK_OLLAMA_MODEL, messages)
                    return fallback_llm.invoke(fallback_msgs)
                except Exception as fb_exc:
                    return _ashna_provider_error_message(exc, fb_exc)
            return _ashna_provider_error_message(exc)

        if _is_ollama_model_not_found_error(exc) or _is_ollama_unavailable_error(exc):
            logger.warning("Primary Ollama %s unavailable: %s", active_primary, exc)
            fallback_llm = _build_llm_with_tools(FALLBACK_OLLAMA_MODEL, bound_tools=bound_tools) if (FALLBACK_OLLAMA_MODEL and bound_tools is not None) else get_fallback_llm_with_tools()
            if fallback_llm is None:
                return _model_not_found_message(active_primary)
            try:
                fallback_msgs = _clean_messages_for_model(FALLBACK_OLLAMA_MODEL, messages)
                return fallback_llm.invoke(fallback_msgs)
            except Exception as fb_exc:
                if _is_retryable_ollama_error(fb_exc):
                    return _memory_error_message()
                raise

        if not _is_ollama_memory_error(exc):
            raise

        logger.warning("Primary Model crash. Retrying with emergency trimmed history...")
        time.sleep(1.0)
        try:
            trimmed = _trim_context(messages, max_non_system=1)
            return _build_llm_with_tools(active_primary, bound_tools=bound_tools).invoke(trimmed)
        except Exception:
            fallback_llm = _build_llm_with_tools(FALLBACK_OLLAMA_MODEL, bound_tools=bound_tools) if (FALLBACK_OLLAMA_MODEL and bound_tools is not None) else get_fallback_llm_with_tools()
            if fallback_llm is not None and FALLBACK_OLLAMA_MODEL != active_primary:
                try:
                    fallback_msgs = _clean_messages_for_model(FALLBACK_OLLAMA_MODEL, messages)
                    return fallback_llm.invoke(fallback_msgs)
                except Exception as fb_exc:
                    if _is_retryable_ollama_error(fb_exc):
                        return _memory_error_message()
                    raise
            return _memory_error_message()
