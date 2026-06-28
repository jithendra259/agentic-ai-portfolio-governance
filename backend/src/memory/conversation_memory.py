from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from src.memory.session_state import utc_now_iso


MAX_MEMORY_MESSAGES = 20
MAX_SUMMARY_CHARS = 900
_TICKER_RE = re.compile(r"(?<![A-Z0-9.])([A-Z0-9]{1,10}(?:[.-][A-Z0-9]{1,8})?)(?![A-Z0-9.])")
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_STOP_TICKERS = {
    "AI", "API", "AND", "ARE", "AS", "BY", "CV", "CVAR", "DO", "FOR", "G", "GCVAR",
    "G-CV", "HI", "IF", "IN", "IS", "IT", "LLM", "ME", "NO", "NOW", "OF", "OR", "PLOT",
    "THE", "THIS", "TO", "UI", "USE", "VAR", "YES",
}

_STRATEGY_ALIASES = [
    ("Adaptive G-CVaR", re.compile(r"\badaptive\s+g[-\s]?cvar\b", re.I)),
    ("Standard CVaR", re.compile(r"\bstandard\s+cvar\b", re.I)),
    ("G-CVaR", re.compile(r"\bg[-\s]?cvar\b", re.I)),
    ("CVaR", re.compile(r"\bcvar\b", re.I)),
]

_METRIC_ALIASES = [
    ("revenue", re.compile(r"\b(revenue|sales|total revenue)\b", re.I)),
    ("net income", re.compile(r"\b(net income|net profit|profit after tax|pat|earnings)\b", re.I)),
    ("market cap", re.compile(r"\b(market cap|market capitalization)\b", re.I)),
    ("PE ratio", re.compile(r"\b(pe|p/e|trailing pe|forward pe)\b", re.I)),
    ("operating cash flow", re.compile(r"\b(operating cash flow|cash flow from operations|cfo)\b", re.I)),
    ("free cash flow", re.compile(r"\b(free cash flow|fcf)\b", re.I)),
    ("total assets", re.compile(r"\b(total assets|assets)\b", re.I)),
    ("total liabilities", re.compile(r"\b(total liabilities|liabilities)\b", re.I)),
    ("debt", re.compile(r"\b(total debt|debt)\b", re.I)),
    ("dividends", re.compile(r"\b(dividend|dividends)\b", re.I)),
    ("price history", re.compile(r"\b(price history|recent price|closing price|close price)\b", re.I)),
]


def default_conversation_state() -> dict[str, Any]:
    return {
        "messages": [],
        "summary": "",
        "current_topic": None,
        "selected_tickers": [],
            "current_strategy": None,
            "last_run_id": None,
            "comparison_target": None,
            "current_metrics": [],
            "user_intent": None,
        "resolved_question": None,
        "safety_context": "advisory_only_no_trading_execution",
        "updated_at": utc_now_iso(),
    }


def normalize_conversation_state(value: dict[str, Any] | None) -> dict[str, Any]:
    state = default_conversation_state()
    if isinstance(value, dict):
        for key, existing in value.items():
            if key in state:
                state[key] = deepcopy(existing)
    state["messages"] = [dict(row) for row in (state.get("messages") or [])[-MAX_MEMORY_MESSAGES:] if isinstance(row, dict)]
    state["selected_tickers"] = _unique_tickers(state.get("selected_tickers") or [])
    return state


def update_for_user_message(session_state: dict[str, Any], message: str) -> dict[str, Any]:
    state = deepcopy(session_state or {})
    conversation = normalize_conversation_state(state.get("conversation_state"))
    tickers = _extract_tickers(message)
    explicit_strategy = _extract_strategy(message)
    strategy = explicit_strategy or conversation.get("current_strategy")
    metrics = _extract_metrics(message) or conversation.get("current_metrics") or []
    if _has_reference(message) and conversation.get("current_strategy") and explicit_strategy:
        if explicit_strategy != conversation.get("current_strategy"):
            conversation["comparison_target"] = explicit_strategy
            strategy = conversation.get("current_strategy")
    intent = _detect_intent(message)
    topic = _detect_topic(message, strategy, tickers) or conversation.get("current_topic")

    if not tickers:
        tickers = _unique_tickers(state.get("active_tickers") or conversation.get("selected_tickers") or [])
    if tickers:
        conversation["selected_tickers"] = tickers
        state["active_tickers"] = tickers
        state["active_ticker_count"] = len(tickers)
    if strategy:
        conversation["current_strategy"] = strategy
    if metrics:
        conversation["current_metrics"] = metrics
    if topic:
        conversation["current_topic"] = topic
    conversation["user_intent"] = intent
    conversation["resolved_question"] = _resolve_question(message, conversation)
    _append_memory_note(conversation, "user", _summarize_user(message, intent, tickers, strategy, metrics))
    conversation["summary"] = _build_summary(conversation)
    conversation["updated_at"] = utc_now_iso()
    state["conversation_state"] = conversation
    return state


def update_for_assistant_response(
    session_state: dict[str, Any],
    response_text: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = deepcopy(session_state or {})
    conversation = normalize_conversation_state(state.get("conversation_state"))
    metadata = metadata or {}

    if state.get("active_tickers"):
        conversation["selected_tickers"] = _unique_tickers(state.get("active_tickers"))
    strategy = _extract_strategy(response_text) or conversation.get("current_strategy")
    metrics = _extract_metrics(response_text) or conversation.get("current_metrics") or []
    if strategy:
        conversation["current_strategy"] = strategy
    if metrics:
        conversation["current_metrics"] = metrics

    run_id = (
        state.get("active_analysis_id")
        or (state.get("latest_governance_run") or {}).get("run_id")
        or metadata.get("run_id")
    )
    if run_id:
        conversation["last_run_id"] = str(run_id)

    if not conversation.get("current_topic") and strategy:
        conversation["current_topic"] = f"{strategy} discussion"
    _append_memory_note(conversation, "assistant", _summarize_assistant(response_text))
    conversation["summary"] = _build_summary(conversation)
    conversation["updated_at"] = utc_now_iso()
    state["conversation_state"] = conversation
    return state


def conversation_prompt_block(session_state: dict[str, Any]) -> str:
    conversation = normalize_conversation_state((session_state or {}).get("conversation_state"))
    if not any(
        conversation.get(key)
        for key in ("summary", "current_topic", "selected_tickers", "current_strategy", "comparison_target", "current_metrics", "last_run_id", "resolved_question")
    ):
        return ""

    messages = conversation.get("messages") or []
    recent_lines = "\n".join(
        f"- {row.get('role', 'note')}: {row.get('note', '')}"
        for row in messages[-10:]
        if row.get("note")
    )
    return (
        "### CONVERSATION MEMORY ###\n"
        "Use this compact thread memory before answering follow-up questions.\n"
        f"Summary: {conversation.get('summary') or 'No running summary yet.'}\n"
        f"Current topic: {conversation.get('current_topic') or 'unknown'}\n"
        f"Selected tickers: {', '.join(conversation.get('selected_tickers') or []) or 'none'}\n"
        f"Current strategy: {conversation.get('current_strategy') or 'unknown'}\n"
        f"Comparison target: {conversation.get('comparison_target') or 'none'}\n"
        f"Current metrics: {', '.join(conversation.get('current_metrics') or []) or 'none'}\n"
        f"Last analysis run: {conversation.get('last_run_id') or 'none'}\n"
        f"User intent: {conversation.get('user_intent') or 'unknown'}\n"
        f"Resolved question: {conversation.get('resolved_question') or 'not needed'}\n"
        f"Safety context: {conversation.get('safety_context') or 'advisory_only_no_trading_execution'}\n"
        "Recent memory notes:\n"
        f"{recent_lines or '- none'}\n"
        "Memory rule: resolve words like it, this, that, same, previous, above, values, numbers, and comparison using this block unless the user changes context.\n"
        "If Resolved question is not 'not needed', answer the resolved question rather than asking for tickers or metrics again.\n"
    )


def _append_memory_note(conversation: dict[str, Any], role: str, note: str) -> None:
    if not note:
        return
    rows = list(conversation.get("messages") or [])
    rows.append({"role": role, "note": note[:220], "created_at": utc_now_iso()})
    conversation["messages"] = rows[-MAX_MEMORY_MESSAGES:]


def _extract_tickers(text: str) -> list[str]:
    tickers = []
    for raw in _TICKER_RE.findall(str(text or "")):
        if raw != raw.upper():
            continue
        normalized = raw.upper()
        if normalized.isdigit() or normalized in _STOP_TICKERS:
            continue
        if len(normalized) == 1 and "." not in normalized and "-" not in normalized:
            continue
        if normalized not in tickers:
            tickers.append(normalized)
    return tickers


def _unique_tickers(values: Any) -> list[str]:
    output = []
    if not isinstance(values, (list, tuple, set)):
        return output
    for value in values:
        ticker = str(value or "").strip().upper()
        if ticker and ticker not in output:
            output.append(ticker)
    return output


def _extract_strategy(text: str) -> str | None:
    for label, pattern in _STRATEGY_ALIASES:
        if pattern.search(str(text or "")):
            return label
    return None


def _extract_metrics(text: str) -> list[str]:
    metrics = []
    for label, pattern in _METRIC_ALIASES:
        if pattern.search(str(text or "")) and label not in metrics:
            metrics.append(label)
    return metrics


def _detect_intent(text: str) -> str:
    normalized = " ".join(str(text or "").lower().split())
    if any(token in normalized for token in ("compare", "difference", "versus", " vs ")):
        return "compare"
    if any(token in normalized for token in ("explain", "what is", "describe")):
        return "explain"
    if any(token in normalized for token in ("plot", "chart", "visualize", "show graph")):
        return "plot"
    if any(token in normalized for token in ("portfolio analysis", "governance", "optimization", "optimisation")):
        return "portfolio_analysis"
    return "general_followup" if _has_reference(text) else "general"


def _detect_topic(text: str, strategy: str | None, tickers: list[str]) -> str | None:
    intent = _detect_intent(text)
    metrics = _extract_metrics(text)
    if metrics and intent == "compare":
        return f"{' and '.join(metrics)} comparison"
    if metrics:
        return f"{' and '.join(metrics)} analysis"
    if strategy and intent == "compare":
        return f"{strategy} comparison"
    if strategy:
        return f"{strategy} discussion"
    if tickers and intent == "portfolio_analysis":
        return "Portfolio governance analysis"
    if tickers and intent == "plot":
        return "Price chart analysis"
    return None


def _resolve_question(text: str, conversation: dict[str, Any]) -> str | None:
    if not (_has_reference(text) or _is_vague_values_followup(text)):
        return None
    strategy = conversation.get("current_strategy")
    topic = conversation.get("current_topic")
    tickers = conversation.get("selected_tickers") or []
    metrics = conversation.get("current_metrics") or []
    parts = [str(text or "").strip()]
    if strategy:
        parts.append(f"Reference resolution: it/this/that = {strategy}.")
    elif topic:
        parts.append(f"Reference resolution: it/this/that = {topic}.")
    if tickers:
        parts.append(f"Use selected tickers: {', '.join(tickers)}.")
    if metrics:
        parts.append(f"Use active metrics: {', '.join(metrics)}.")
    if topic:
        parts.append(f"Use current topic: {topic}.")
    return " ".join(parts)


def _has_reference(text: str) -> bool:
    return bool(re.search(r"\b(it|this|that|these|those|same|previous|above|them)\b", str(text or ""), re.I))


def _is_vague_values_followup(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    return bool(
        re.search(r"\b(value|values|numbers|exact|actual|comparison|compare|table)\b", normalized)
        and not _extract_tickers(text)
        and not _extract_metrics(text)
    )


def _summarize_user(text: str, intent: str, tickers: list[str], strategy: str | None, metrics: list[str] | None = None) -> str:
    parts = [f"User intent={intent}"]
    if strategy:
        parts.append(f"strategy={strategy}")
    if metrics:
        parts.append(f"metrics={', '.join(metrics)}")
    if tickers:
        parts.append(f"tickers={', '.join(tickers)}")
    dates = _DATE_RE.findall(str(text or ""))
    if dates:
        parts.append(f"dates={', '.join(dates)}")
    return "; ".join(parts)


def _summarize_assistant(text: str) -> str:
    compact = " ".join(str(text or "").split())
    if not compact:
        return ""
    if "Historical Governance Report" in compact:
        return "Bot returned a historical governance report."
    if "Chart ready" in compact or "generated the" in compact.lower():
        return "Bot generated or referenced an analysis chart."
    return f"Bot answered: {compact[:160]}"


def _build_summary(conversation: dict[str, Any]) -> str:
    topic = conversation.get("current_topic")
    strategy = conversation.get("current_strategy")
    tickers = conversation.get("selected_tickers") or []
    intent = conversation.get("user_intent")
    pieces = []
    if topic:
        pieces.append(f"Current topic is {topic}.")
    if strategy:
        pieces.append(f"Current strategy is {strategy}.")
    if conversation.get("comparison_target"):
        pieces.append(f"Comparison target is {conversation['comparison_target']}.")
    if tickers:
        pieces.append(f"Selected tickers are {', '.join(tickers)}.")
    metrics = conversation.get("current_metrics") or []
    if metrics:
        pieces.append(f"Current metrics are {', '.join(metrics)}.")
    if intent:
        pieces.append(f"Latest user intent is {intent}.")
    if conversation.get("last_run_id"):
        pieces.append(f"Last analysis run id is {conversation['last_run_id']}.")
    summary = " ".join(pieces)
    return summary[:MAX_SUMMARY_CHARS]
