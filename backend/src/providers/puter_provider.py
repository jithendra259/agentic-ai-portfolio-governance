"""Backend adapter for Puter (Puter REST-compatible endpoints).

Usage:
- Set `PUTER_BASE_URL` in backend/.env (e.g. https://api.puter.com)
- Optionally set `PUTER_API_KEY` for authenticated requests

This module provides a small `call_puter_chat` helper that posts a chat-style
request to the Puter REST endpoint and returns the assistant text.

Note: Puter is primarily designed for client-side usage; server-side APIs
and endpoints may differ. This adapter attempts to call a standard
OpenAI-compatible `/v1/chat/completions` endpoint on the configured base URL.
If your provider uses a different endpoint, adjust `endpoint` accordingly.
"""
from typing import Optional, Dict, Any
import os
import requests

PUTER_BASE_URL = os.getenv("PUTER_BASE_URL")
PUTER_API_KEY = os.getenv("PUTER_API_KEY")


class PuterError(Exception):
    pass


def _normalize_base(base: str) -> str:
    b = base.rstrip("/")
    # remove legacy /api if present
    if b.endswith("/api"):
        b = b[:-4]
    # ensure no trailing /v1 (we will append v1 path)
    b = b.rstrip("/v1")
    return b


def call_puter_chat(prompt: str, model: str = "qwen/qwen3.7-plus", timeout: int = 30, **kwargs) -> Dict[str, Any]:
    """Call Puter REST chat endpoint and return parsed response.

    Returns full JSON response on success.
    Raises PuterError on failure or if not configured.
    """
    if not PUTER_BASE_URL:
        raise PuterError("PUTER_BASE_URL is not configured in environment")

    base = _normalize_base(PUTER_BASE_URL)
    endpoint = f"{base}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if PUTER_API_KEY:
        headers["Authorization"] = f"Bearer {PUTER_API_KEY}"

    payload = {
        "model": model,
        # Adapt to the most common schema: messages array
        "messages": [{"role": "user", "content": prompt}],
    }
    # merge any allowed kwargs into payload (like max_tokens, temperature)
    payload.update(kwargs)

    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    except Exception as e:
        raise PuterError(f"HTTP request failed: {e}")

    if not resp.ok:
        # include short body snippet for diagnostics but avoid leaking secrets
        body_snippet = resp.text[:2000]
        raise PuterError(f"Puter API returned status {resp.status_code}: {body_snippet}")

    try:
        data = resp.json()
    except Exception as e:
        raise PuterError(f"Failed to parse JSON response: {e}")

    # Try to extract assistant message (OpenAI-like structure)
    try:
        # Chat completions may return choices -> message -> content
        choices = data.get("choices")
        if choices and isinstance(choices, list):
            first = choices[0]
            # support both `message.content` and `text`
            message = first.get("message") or {}
            text = message.get("content") or first.get("text")
            if text is not None:
                return {"text": text, "raw": data}
    except Exception:
        pass

    # Fallback: return full JSON
    return {"text": None, "raw": data}


class PuterLLM:
    """A tiny LLM-like wrapper that adapts `call_puter_chat` to the orchestrator.

    It implements `bind_tools()` and `invoke(messages)` so it can be used where
    a LangChain/ollama-like runnable is expected. `invoke` accepts a list of
    `langchain_core.messages` style messages and returns an `AIMessage`-like
    object (dict with `content`) to keep the orchestrator integration simple.
    """

    def __init__(self, model: str = "qwen/qwen3.7-plus", temperature: float = 0.2, **kwargs):
        self.model = model
        self.temperature = temperature
        # allow passthrough of other kwargs for future extension
        self._kwargs = kwargs or {}

    def bind_tools(self, tools):
        # No-op: tools are not used by Puter wrapper, but keep API compatible
        return self

    def _messages_to_prompt(self, messages: list) -> str:
        parts = []
        for m in messages:
            role = getattr(m, "type", None) or getattr(m, "role", None) or m.__class__.__name__
            content = getattr(m, "content", None) or str(m)
            parts.append(f"[{role}] {content}")
        return "\n".join(parts)

    def invoke(self, messages: list, timeout: int = 30):
        # Convert messages to a single prompt and call Puter
        prompt = self._messages_to_prompt(messages)
        try:
            result = call_puter_chat(prompt, model=self.model, timeout=timeout, temperature=self.temperature)
        except PuterError as e:
            raise

        text = result.get("text")
        # Minimal AIMessage-like dict expected by orchestrator
        return {"content": text or "", "raw": result.get("raw")}
