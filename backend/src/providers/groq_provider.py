"""Groq API provider helpers and client factory."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
KNOWN_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "allam-2-7b",
]


def normalize_groq_base_url(base_url: str | None) -> str:
    """Return Groq's OpenAI-compatible base URL root."""
    if not base_url or not base_url.strip():
        return GROQ_OPENAI_BASE_URL

    normalized = base_url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/chat"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].rstrip("/")

    if normalized.endswith("/openai/v1"):
        return normalized
    if normalized.endswith("/openai"):
        return f"{normalized}/v1"
    if normalized.endswith("/v1"):
        return normalized

    return f"{normalized}/openai/v1"


def normalize_groq_model_name(model_name: str | None) -> str:
    """Normalize model identifier for Groq API."""
    if not model_name or not model_name.strip():
        return DEFAULT_GROQ_MODEL

    cleaned = model_name.strip()
    if cleaned.startswith("groq/"):
        cleaned = cleaned[len("groq/"):]

    if cleaned in ("groq", "default", "groq-default"):
        return os.getenv("PORTFOLIO_GROQ_MODEL") or os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL

    return cleaned


def is_groq_model(model_name: str | None) -> bool:
    """Check if model name corresponds to a Groq model."""
    if not model_name:
        return False
    name = model_name.strip().lower()
    if name.startswith("groq") or name.startswith("groq/"):
        return True
    if name.startswith("openai/gpt-oss") or name.startswith("qwen/qwen3.8") or name.startswith("allam-"):
        return True
    if name in [m.lower() for m in KNOWN_GROQ_MODELS]:
        return True
    return False


def get_groq_chat_llm(
    model_name: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    streaming: bool = False,
    **kwargs: Any,
) -> Any:
    """Build a LangChain Chat client connected to Groq.
    
    Prefers `langchain_groq.ChatGroq` if available, otherwise seamlessly uses
    `langchain_openai.ChatOpenAI` pointed to Groq's OpenAI-compatible endpoint.
    """
    resolved_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not resolved_api_key:
        raise ValueError("GROQ_API_KEY is not set in environment or arguments.")

    actual_model = normalize_groq_model_name(model_name)
    resolved_base_url = normalize_groq_base_url(base_url or os.getenv("GROQ_BASE_URL"))

    # Try langchain_groq first if installed
    try:
        from langchain_groq import ChatGroq

        params = {
            "model_name": actual_model,
            "temperature": temperature,
            "groq_api_key": resolved_api_key,
            "streaming": streaming,
            "max_retries": 2,
            "tags": ["groq_llm"],
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        params.update(kwargs)
        logger.info(f"Initialized ChatGroq with model={actual_model}")
        return ChatGroq(**params)
    except ImportError:
        pass

    # Fallback to langchain_openai with Groq's OpenAI-compatible endpoint
    from langchain_openai import ChatOpenAI

    params = {
        "model": actual_model,
        "temperature": temperature,
        "api_key": resolved_api_key,
        "base_url": resolved_base_url,
        "streaming": streaming,
        "timeout": 60,
        "max_retries": 2,
        "tags": ["groq_llm", "orchestrator_llm"],
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    params.update(kwargs)
    logger.info(f"Initialized Groq via ChatOpenAI with model={actual_model}, base_url={resolved_base_url}")
    return ChatOpenAI(**params)
