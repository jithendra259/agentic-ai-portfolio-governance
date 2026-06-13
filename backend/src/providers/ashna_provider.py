"""Ashna OpenAI-compatible client helpers."""

from __future__ import annotations


ASHNA_OPENAI_BASE_URL = "https://api.ashna.ai/v1/api"


def normalize_ashna_base_url(base_url: str | None) -> str:
    """Return Ashna's OpenAI-compatible root for SDK clients.

    Ashna's docs expose chat completions under ``/v1/api/chat/completions``.
    LangChain/OpenAI clients append ``/chat/completions`` themselves, so their
    ``base_url`` must end at ``/v1/api``.
    """
    if not base_url or not base_url.strip():
        return ""

    normalized = base_url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/chat"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].rstrip("/")

    if normalized.endswith("/v1/api"):
        return normalized

    if normalized.endswith("/v1"):
        return f"{normalized}/api"

    if normalized.endswith("/api"):
        root = normalized[: -len("/api")].rstrip("/")
        return f"{root}/v1/api"

    return f"{normalized}/v1/api"
