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

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from src.providers.omniroute_provider import (
    DEFAULT_OMNIROUTE_MODEL,
    get_omniroute_chat_llm,
    is_omniroute_model,
)

logger = logging.getLogger(__name__)

# Constants
_MAX_TOOL_MSG_CHARS = 4000
_MAX_CONTEXT_MESSAGES = 15
_MAX_SUMMARY_CHARS = 1200

CONFIGURED_PRIMARY_OLLAMA_MODEL = (
    os.getenv("PORTFOLIO_OLLAMA_MODEL") or "gpt-oss:120b-cloud"
).strip()
CONFIGURED_FALLBACK_OLLAMA_MODEL = (
    os.getenv("PORTFOLIO_OLLAMA_FALLBACK_MODEL") or "gpt-oss:20b-cloud"
).strip()
CONFIGURED_DEFAULT_LLM_MODEL = (
    os.getenv("PORTFOLIO_DEFAULT_LLM_MODEL") or CONFIGURED_PRIMARY_OLLAMA_MODEL
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
        if candidate and candidate in installed_models:
            return candidate
    return installed_models[0] if installed_models else (preferred_models[0] if preferred_models else "nemotron-3.5-lightning:latest")


def _should_probe_ollama_on_startup() -> bool:
    explicit = os.getenv("PORTFOLIO_PROBE_OLLAMA_ON_STARTUP")
    if explicit is not None:
        return explicit.strip().lower() in {"1", "true", "yes", "on"}
    return True


INSTALLED_OLLAMA_MODELS = (
    _list_installed_ollama_models() if _should_probe_ollama_on_startup() else []
)

PRIMARY_OLLAMA_MODEL = _resolve_ollama_model(
    [CONFIGURED_PRIMARY_OLLAMA_MODEL, "nemotron-3.5-lightning:latest"],
    INSTALLED_OLLAMA_MODELS,
)
FALLBACK_OLLAMA_MODEL = _resolve_ollama_model(
    [CONFIGURED_FALLBACK_OLLAMA_MODEL, PRIMARY_OLLAMA_MODEL],
    INSTALLED_OLLAMA_MODELS,
)


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


def _available_models_text() -> str:
    return ", ".join(INSTALLED_OLLAMA_MODELS) if INSTALLED_OLLAMA_MODELS else "No installed models detected from `ollama list`."


def _memory_error_message() -> AIMessage:
    fallback_text = f" I also attempted fallback `{FALLBACK_OLLAMA_MODEL}`." if FALLBACK_OLLAMA_MODEL != PRIMARY_OLLAMA_MODEL else ""
    return AIMessage(
        content=(
            f"The local Ollama model `{PRIMARY_OLLAMA_MODEL}` needs more system RAM/VRAM than is available."
            f"{fallback_text}\n\nTry running a smaller Ollama model or ensuring Ollama has sufficient resources."
        )
    )


def _model_not_found_message(model_name: str) -> AIMessage:
    return AIMessage(
        content=(
            f"The configured Ollama model `{model_name}` is not installed.\n\n"
            f"Detected local models: {_available_models_text()}\n\n"
            "Either pull the model via `ollama run <name>` or set `PORTFOLIO_OLLAMA_MODEL`."
        )
    )


def _clean_messages_for_model(model_name: str, messages: list[BaseMessage]) -> list[BaseMessage]:
    if not is_omniroute_model(model_name):
        return messages

    cleaned: list[BaseMessage] = []
    sys_texts: list[str] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            sys_texts.append(str(m.content))
        elif isinstance(m, ToolMessage):
            name = getattr(m, "name", None) or "tool"
            cleaned.append(HumanMessage(content=f"[Tool Output ({name})]:\n{m.content}"))
        else:
            cleaned.append(m)

    if sys_texts:
        combined_sys = "\n\n".join(sys_texts)
        found_human = False
        for i, m in enumerate(cleaned):
            if isinstance(m, HumanMessage):
                cleaned[i] = HumanMessage(content=f"System Instructions:\n{combined_sys}\n\nUser Request:\n{m.content}")
                found_human = True
                break
        if not found_human:
            cleaned.insert(0, HumanMessage(content=f"System Instructions:\n{combined_sys}"))

    return cleaned


def _is_ashna_model(model_name: str) -> bool:
    return False


def _clean_messages_for_ashna(messages: list[BaseMessage]) -> list[BaseMessage]:
    return messages


def _ashna_provider_error_message(exc: Exception, fallback_exc: Exception | None = None) -> AIMessage:
    return AIMessage(content="External providers disabled. Only local Ollama and OmniRoute models are enabled.")


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
    omniroute_api_key = os.getenv("OMNIROUTE_API_KEY")

    # 1. OmniRoute (Local proxy)
    if is_omniroute_model(model_name) or (omniroute_api_key and str(model_name).startswith("omniroute/")):
        try:
            return get_omniroute_chat_llm(model_name=model_name, temperature=temperature, max_tokens=num_predict)
        except Exception as exc:
            logger.error("Failed to initialize OmniRoute client: %s", exc)

    # 2. Local Ollama (Primary / Fallback)
    try:
        from langchain_ollama import ChatOllama
        ollama_base_url = (os.getenv("PORTFOLIO_OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").strip()
        actual_model = model_name[len("ollama/"):] if str(model_name).startswith("ollama/") else (model_name or PRIMARY_OLLAMA_MODEL)
        if not actual_model or actual_model in ("default", "primary"):
            actual_model = PRIMARY_OLLAMA_MODEL
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
        logger.error("Local Ollama initialization failed: %s", exc)
        raise ValueError(f"No local LLM available. Ensure Ollama is running at {ollama_base_url} or OmniRoute is active.")


_llm_lock = threading.Lock()
_cached_tools: list = []


def register_global_tools(tools_list: list):
    global _cached_tools
    _cached_tools = tools_list


def _build_llm_with_tools(model_name: str, bound_tools: list = None):
    target_tools = _cached_tools if bound_tools is None else bound_tools
    llm = _get_chat_llm(model_name)
    if is_omniroute_model(model_name):
        # DuckDuckGo and web routes behind OmniRoute do not accept native tool/function payloads
        # and reject requests with HTTP 400 ERR_BAD_REQUEST if tools are attached.
        # Return base model directly so conversational inference succeeds cleanly.
        return llm
    try:
        return llm.bind_tools(target_tools)
    except Exception as exc:
        logger.warning(
            "Model %s does not support tool calling (%s). Binding to primary Ollama %s",
            model_name, exc, PRIMARY_OLLAMA_MODEL
        )
        if model_name != PRIMARY_OLLAMA_MODEL:
            try:
                return _get_chat_llm(PRIMARY_OLLAMA_MODEL).bind_tools(target_tools)
            except Exception:
                pass
        return llm


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
    """Local LLM invocation across Ollama and OmniRoute with failover."""
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

    is_omni = is_omniroute_model(active_primary)
    cleaned_messages = _clean_messages_for_model(active_primary, messages)

    try:
        return active_llm.invoke(cleaned_messages)
    except Exception as exc:
        err_str = str(exc).lower()
        if "bad_request" in err_str or "err_bad_request" in err_str or "400" in err_str:
            logger.warning("Invocation error on %s (%s). Retrying raw invocation without tools...", active_primary, exc)
            try:
                raw_llm = _get_chat_llm(active_primary)
                return raw_llm.invoke(cleaned_messages)
            except Exception as retry_exc:
                logger.warning("Raw invocation retry on %s also failed: %s", active_primary, retry_exc)

        if is_omni:
            # Automatic sibling failover across verified active OmniRoute models if one route fails or hits rate limits
            candidate_siblings = [
                "omniroute/ddgw/claude-haiku-4-5",
                "omniroute/ddgw/mistral-small-2603",
                "omniroute/ddgw/gpt-5.4-nano",
                "omniroute/ddgw/gpt-5.4-mini",
            ]
            for sibling in candidate_siblings:
                if sibling != active_primary and not sibling.endswith(active_primary):
                    try:
                        logger.info("Attempting OmniRoute sibling failover to %s...", sibling)
                        sib_llm = _get_chat_llm(sibling)
                        return sib_llm.invoke(_clean_messages_for_model(sibling, messages))
                    except Exception as sib_exc:
                        logger.debug("OmniRoute sibling %s failed: %s", sibling, sib_exc)

            logger.warning("OmniRoute model %s failed: %s. Attempting failover to local Ollama (%s).", active_primary, exc, PRIMARY_OLLAMA_MODEL)
            try:
                fallback_llm = _build_llm_with_tools(PRIMARY_OLLAMA_MODEL, bound_tools=bound_tools)
                return fallback_llm.invoke(_clean_messages_for_model(PRIMARY_OLLAMA_MODEL, messages))
            except Exception as ollama_exc:
                logger.warning("Failover from OmniRoute to Ollama failed: %s", ollama_exc)

            return AIMessage(content=f"OmniRoute request encountered an error: {exc}. All sibling routes and local Ollama fallback were unavailable.")

        if _is_ollama_model_not_found_error(exc) or _is_ollama_unavailable_error(exc):
            logger.warning("Primary Ollama %s unavailable: %s", active_primary, exc)
            if os.getenv("OMNIROUTE_API_KEY"):
                try:
                    logger.info("Failing over from Ollama to OmniRoute...")
                    omni_llm = _build_llm_with_tools("omniroute/auto", bound_tools=bound_tools)
                    return omni_llm.invoke(messages)
                except Exception as omni_exc:
                    logger.warning("Failover from Ollama to OmniRoute failed: %s", omni_exc)
            return _model_not_found_message(active_primary)

        if not _is_ollama_memory_error(exc):
            try:
                trimmed = _trim_context(messages, max_non_system=1)
                return _build_llm_with_tools(active_primary, bound_tools=bound_tools).invoke(trimmed)
            except Exception:
                pass
            raise

        logger.warning("Ollama memory error. Retrying with emergency trimmed history...")
        time.sleep(1.0)
        try:
            trimmed = _trim_context(messages, max_non_system=1)
            return _build_llm_with_tools(active_primary, bound_tools=bound_tools).invoke(trimmed)
        except Exception:
            fallback_llm = _build_llm_with_tools(FALLBACK_OLLAMA_MODEL, bound_tools=bound_tools) if (FALLBACK_OLLAMA_MODEL and bound_tools is not None) else get_fallback_llm_with_tools()
            if fallback_llm is not None and FALLBACK_OLLAMA_MODEL != active_primary:
                try:
                    return fallback_llm.invoke(_clean_messages_for_model(FALLBACK_OLLAMA_MODEL, messages))
                except Exception as fb_exc:
                    if _is_retryable_ollama_error(fb_exc):
                        return _memory_error_message()
                    raise
            return _memory_error_message()
