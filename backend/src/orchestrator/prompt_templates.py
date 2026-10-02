"""Prompt Engineering & Assembly Layer.

Defines the core institutional system prompt, guidelines loading,
and blackboard state assembly for the orchestrator.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from src.memory.conversation_memory import conversation_prompt_block
from src.orchestrator.formatters import _message_content_to_text

logger = logging.getLogger(__name__)

_MAX_TOOL_MSG_CHARS = 1000
_MAX_CONTEXT_MESSAGES = 5
_MAX_SUMMARY_CHARS = 800

CHATBOT_CONVERSATION_GUIDANCE_DIR = Path(__file__).resolve().parents[1] / "rag" / "knowledge" / "chatbot_conversation"

from src.orchestrator.prompts import SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT, get_chatbot_prompt


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
    priority_files = ["hallucination_guardrails.md", "portfolio_followups.md"]
    for filename in priority_files:
        path = CHATBOT_CONVERSATION_GUIDANCE_DIR / filename
        if path.exists():
            try:
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    chunks.append(text)
            except OSError as exc:
                logger.warning("Unable to read guidance %s: %s", path, exc)
    if not chunks:
        return ""
    return "### CHATBOT CONVERSATION GUIDANCE ###\n" + "\n\n".join(chunks)


def assemble_system_prompt(state: dict[str, Any]) -> str:
    """Build the dynamic system prompt with blackboard state injected."""
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
