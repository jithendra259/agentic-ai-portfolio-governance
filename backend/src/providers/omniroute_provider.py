"""OmniRoute OpenAI-compatible client helpers and provider factory."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)

DEFAULT_OMNIROUTE_BASE_URL = "http://127.0.0.1:20128/v1"
DEFAULT_OMNIROUTE_MODEL = "ddgw/gpt-5.4-mini"


def normalize_omniroute_base_url(base_url: str | None) -> str:
    """Return OmniRoute's OpenAI-compatible base URL root."""
    if not base_url or not base_url.strip():
        return (os.getenv("OMNIROUTE_BASE_URL") or DEFAULT_OMNIROUTE_BASE_URL).strip().rstrip("/")

    normalized = base_url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/chat"):
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)].rstrip("/")

    if normalized.endswith("/v1"):
        return normalized

    return f"{normalized}/v1"


def get_omniroute_available_models(
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 2.0,
) -> list[str]:
    """Query OmniRoute /v1/models to fetch available active models."""
    resolved_base = normalize_omniroute_base_url(base_url)
    resolved_key = (api_key or os.getenv("OMNIROUTE_API_KEY") or "").strip()
    url = f"{resolved_base}/models"

    headers = {"User-Agent": "PortfolioGovernance/1.0"}
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"

    # Verified active streaming models in OmniRoute
    priority_models = [
        "omniroute/ddgw/gpt-5.4-mini",
        "omniroute/ddgw/claude-haiku-4-5",
        "omniroute/ddgw/mistral-small-2603",
        "omniroute/ddgw/gpt-5.4-nano",
    ]

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            model_items = data.get("data", [])
            models = list(priority_models)
            for item in model_items:
                model_id = item.get("id")
                if model_id and not item.get("id", "").startswith("aihorde"):
                    name = f"omniroute/{model_id}"
                    if name not in models:
                        models.append(name)
            return models
    except Exception as exc:
        logger.debug("OmniRoute models fetch failed or unreachable: %s", exc)
        return list(priority_models)


def is_omniroute_model(model_name: str | None) -> bool:
    """Check if model name corresponds to an OmniRoute route."""
    if not model_name:
        return False
    name = model_name.strip().lower()
    return (
        name.startswith("omniroute/")
        or name.startswith("omni/")
        or name.startswith("ddgw/")
        or name.startswith("aug/")
        or name.startswith("zc/")
        or name.startswith("cxa/")
        or name.startswith("tllm/")
        or name.startswith("pol/")
        or name.startswith("pollinations/")
        or name.startswith("horde/")
        or name == "omniroute"
    )


def get_omniroute_chat_llm(
    model_name: str = DEFAULT_OMNIROUTE_MODEL,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    streaming: bool = False,
    **kwargs: Any,
) -> Any:
    """Build a LangChain ChatOpenAI client connected to OmniRoute."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise RuntimeError("langchain-openai is required to use OmniRoute.")

    resolved_api_key = (api_key or os.getenv("OMNIROUTE_API_KEY") or "").strip()
    if not resolved_api_key:
        raise ValueError("OMNIROUTE_API_KEY is not set.")

    resolved_base_url = normalize_omniroute_base_url(base_url)

    actual_model = model_name.strip()
    if actual_model.startswith("omniroute/"):
        actual_model = actual_model[len("omniroute/"):]
    elif actual_model.startswith("omni/"):
        actual_model = actual_model[len("omni/"):]
    elif actual_model == "omniroute":
        actual_model = os.getenv("OMNIROUTE_MODEL") or DEFAULT_OMNIROUTE_MODEL

    params = {
        "model": actual_model,
        "temperature": temperature,
        "api_key": resolved_api_key,
        "base_url": resolved_base_url,
        "streaming": streaming,
        "timeout": 60,
        "max_retries": 2,
        "tags": ["omniroute_llm", "orchestrator_llm"],
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    params.update(kwargs)

    logger.info(f"Initialized OmniRoute ChatOpenAI with model={actual_model}, base_url={resolved_base_url}")
    return ChatOpenAI(**params)
