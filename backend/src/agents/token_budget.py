from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


DEFAULT_MAX_TOKEN_BUDGET = 100_000
DEFAULT_MAX_CONTEXT_TOKENS = 20_000
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 4_000
DEFAULT_INPUT_COST_PER_TOKEN = 0.0000025
DEFAULT_OUTPUT_COST_PER_TOKEN = 0.00001


@dataclass(frozen=True)
class BudgetCheck:
    allowed: bool
    total_tokens_used: int
    remaining_token_budget: int
    estimated_cost_usd: float
    estimated_context_tokens: int
    max_token_budget: int
    reason: str = ""


def estimate_tokens(value: Any) -> int:
    """Cheap provider-independent token estimate for budget circuit breakers."""
    if value is None:
        return 0
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, default=str, sort_keys=True)
        except TypeError:
            text = str(value)
    return max(1, (len(text) + 3) // 4)


def extract_usage_details(response: Any) -> dict[str, int]:
    """Read LangChain/OpenAI-style usage metadata when the provider exposes it."""
    usage = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", {}).get("token_usage")
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    total_tokens = usage.get("total_tokens")
    if total_tokens is not None:
        total = int(total_tokens or 0)
    else:
        total = int(input_tokens) + int(output_tokens)
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": total,
    }


def extract_usage_tokens(response: Any) -> int:
    return extract_usage_details(response)["total_tokens"]


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * DEFAULT_INPUT_COST_PER_TOKEN) + (output_tokens * DEFAULT_OUTPUT_COST_PER_TOKEN)


def update_token_ledger(
    state: dict[str, Any],
    *,
    prompt: Any = None,
    response: Any = None,
    estimated_tokens: int | None = None,
) -> dict[str, int | float]:
    used = int(state.get("total_tokens_used", 0) or 0)
    max_budget = int(state.get("max_token_budget", DEFAULT_MAX_TOKEN_BUDGET) or DEFAULT_MAX_TOKEN_BUDGET)
    usage = extract_usage_details(response) if response is not None else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    response_tokens = usage["total_tokens"]
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    if response_tokens <= 0:
        response_tokens = estimated_tokens if estimated_tokens is not None else estimate_tokens(prompt) + estimate_tokens(response)
        input_tokens = estimate_tokens(prompt)
        output_tokens = max(0, int(response_tokens or 0) - input_tokens)
    new_total = used + max(0, int(response_tokens or 0))
    new_cost = float(state.get("estimated_cost_usd", 0.0) or 0.0) + estimate_cost_usd(input_tokens, output_tokens)
    return {
        "total_tokens_used": new_total,
        "remaining_token_budget": max(0, max_budget - new_total),
        "estimated_cost_usd": round(new_cost, 8),
    }


def check_token_budget(
    state: dict[str, Any],
    context: Any,
    *,
    reserved_tokens: int = 0,
) -> BudgetCheck:
    total_tokens_used = int(state.get("total_tokens_used", 0) or 0)
    max_token_budget = int(state.get("max_token_budget", DEFAULT_MAX_TOKEN_BUDGET) or DEFAULT_MAX_TOKEN_BUDGET)
    estimated_cost_usd = float(state.get("estimated_cost_usd", 0.0) or 0.0)
    estimated_context_tokens = estimate_tokens(context) + max(0, int(reserved_tokens or 0))
    projected_total = total_tokens_used + estimated_context_tokens
    remaining = max(0, max_token_budget - total_tokens_used)
    if projected_total > max_token_budget:
        return BudgetCheck(
            allowed=False,
            total_tokens_used=total_tokens_used,
            remaining_token_budget=remaining,
            estimated_cost_usd=estimated_cost_usd,
            estimated_context_tokens=estimated_context_tokens,
            max_token_budget=max_token_budget,
            reason=(
                f"Token budget would be exceeded: used={total_tokens_used}, "
                f"context={estimated_context_tokens}, max={max_token_budget}"
            ),
        )
    return BudgetCheck(
        allowed=True,
        total_tokens_used=total_tokens_used,
        remaining_token_budget=remaining,
        estimated_cost_usd=estimated_cost_usd,
        estimated_context_tokens=estimated_context_tokens,
        max_token_budget=max_token_budget,
    )


def cap_tool_output(raw_output: Any, max_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS) -> Any:
    """Keep large tool outputs from poisoning the next LLM context."""
    if raw_output is None:
        return raw_output
    if isinstance(raw_output, str):
        if len(raw_output) <= max_chars:
            return raw_output
        return (
            raw_output[:max_chars]
            + "\n...[OUTPUT TRUNCATED DUE TO LENGTH. Filter or aggregate this data instead of reading it raw.]..."
        )
    try:
        encoded = json.dumps(raw_output, default=str, sort_keys=True)
    except TypeError:
        encoded = str(raw_output)
    if len(encoded) <= max_chars:
        return raw_output
    return {
        "status": "truncated",
        "message": "Tool output exceeded the context budget and was truncated.",
        "preview": encoded[:max_chars],
        "truncated_chars": len(encoded) - max_chars,
    }
