from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from src.memory.session_state import KNOWN_UNIVERSE_TICKERS, utc_now_iso


MAX_CONTINUITY_ANCHORS = 12
_DATE_PATTERN = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_TICKER_STOPWORDS = {
    "AND",
    "ARE",
    "BAR",
    "CHART",
    "COMPARE",
    "DATE",
    "FOR",
    "FROM",
    "LINE",
    "NOW",
    "PLOT",
    "SHOW",
    "THE",
    "THEM",
    "TICKER",
    "TICKERS",
    "TO",
    "USE",
    "WITH",
}


def recover_state_from_chat_history(
    session_state: dict[str, Any],
    chat_history_last_25: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Recover compact planner context from persisted chat rows.

    Raw chat history is durable in Supabase, while the planner state is an
    in-process cache. This bounded priority memory rebuilds high-value anchors
    after a restart without calling another model.
    """

    state = deepcopy(session_state or {})
    messages = [row for row in (chat_history_last_25 or []) if isinstance(row, dict)]
    if not messages:
        return state

    memory = build_continuity_memory(state.get("continuity_memory"), messages)
    top = memory.get("top", {})
    if top:
        state["continuity_memory"] = memory

    if not state.get("active_universe") and top.get("universe"):
        state["active_universe"] = top["universe"]

    if not state.get("active_tickers") and top.get("tickers"):
        state["active_tickers"] = top["tickers"]
        state["active_ticker_count"] = len(top["tickers"])

    current_range = state.get("active_date_range") if isinstance(state.get("active_date_range"), dict) else {}
    if top.get("date_range") and not (current_range.get("start") or current_range.get("end")):
        state["active_date_range"] = top["date_range"]

    if not state.get("last_response_summary"):
        last_assistant = next((row for row in reversed(messages) if row.get("role") == "assistant"), None)
        if last_assistant and last_assistant.get("content"):
            state["last_response_summary"] = str(last_assistant["content"])[:500]

    return state


def build_continuity_memory(
    existing_memory: dict[str, Any] | None,
    chat_history_last_25: list[dict[str, Any]],
) -> dict[str, Any]:
    anchors = list((existing_memory or {}).get("anchors") or [])
    total = max(1, len(chat_history_last_25 or []))
    for index, row in enumerate(chat_history_last_25 or []):
        recency = (index + 1) / total
        anchors.extend(_anchors_from_row(row, recency))

    ranked = _rank_anchors(anchors)
    top = _top_context(ranked)
    return {
        "strategy": "bounded_priority_recency_memory",
        "updated_at": utc_now_iso(),
        "anchors": ranked[:MAX_CONTINUITY_ANCHORS],
        "top": top,
    }


def _anchors_from_row(row: dict[str, Any], recency: float) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    contract = metadata.get("response_contract") if isinstance(metadata.get("response_contract"), dict) else {}
    context = contract.get("resolved_context") if isinstance(contract.get("resolved_context"), dict) else {}
    created_at = row.get("created_at")

    contract_universe = _clean_universe(context.get("universe"))
    contract_tickers = _clean_tickers(context.get("tickers"), filter_stopwords=False)
    contract_date_range = _clean_date_range(context.get("date_range"))
    if contract_universe:
        anchors.append(_anchor("universe", contract_universe, 0.95, recency, "response_contract", created_at))
    if contract_tickers:
        anchors.append(_anchor("tickers", contract_tickers, 0.95, recency, "response_contract", created_at))
    if contract_date_range:
        anchors.append(_anchor("date_range", contract_date_range, 0.9, recency, "response_contract", created_at))

    text = str(row.get("content") or "")
    text_universe = _extract_universe(text)
    text_tickers = _extract_tickers(text)
    text_date_range = _extract_date_range(text)
    if text_universe:
        anchors.append(_anchor("universe", text_universe, 0.72, recency, "message_text", created_at))
        if text_universe in KNOWN_UNIVERSE_TICKERS:
            anchors.append(_anchor("tickers", list(KNOWN_UNIVERSE_TICKERS[text_universe]), 0.7, recency, "universe_registry", created_at))
    if text_tickers:
        anchors.append(_anchor("tickers", text_tickers, 0.64, recency, "message_text", created_at))
    if text_date_range:
        anchors.append(_anchor("date_range", text_date_range, 0.62, recency, "message_text", created_at))

    return anchors


def _anchor(kind: str, value: Any, confidence: float, recency: float, source: str, created_at: Any) -> dict[str, Any]:
    return {
        "type": kind,
        "value": value,
        "confidence": round(float(confidence), 4),
        "recency": round(float(recency), 4),
        "score": round(float(confidence) * 10 + float(recency), 4),
        "source": source,
        "last_seen_at": str(created_at or ""),
    }


def _rank_anchors(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for anchor in anchors:
        kind = str(anchor.get("type") or "")
        value_key = _value_key(anchor.get("value"))
        if not kind or not value_key:
            continue
        key = (kind, value_key)
        previous = best_by_key.get(key)
        if previous is None or float(anchor.get("score") or 0) >= float(previous.get("score") or 0):
            best_by_key[key] = anchor
    return sorted(best_by_key.values(), key=lambda item: float(item.get("score") or 0), reverse=True)


def _top_context(anchors: list[dict[str, Any]]) -> dict[str, Any]:
    top: dict[str, Any] = {}
    for anchor in anchors:
        kind = anchor.get("type")
        if kind in top:
            continue
        if kind in {"universe", "tickers", "date_range"}:
            top[kind] = anchor.get("value")
    return top


def _value_key(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return "|".join(f"{key}:{value.get(key)}" for key in sorted(value))
    return str(value or "")


def _clean_universe(value: Any) -> str | None:
    normalized = str(value or "").strip().upper()
    return normalized if re.fullmatch(r"U\d{1,2}", normalized) else None


def _clean_tickers(value: Any, *, filter_stopwords: bool = True) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    cleaned = []
    for item in value:
        ticker = str(item or "").strip().upper()
        if (
            re.fullmatch(r"[A-Z][A-Z0-9.]{0,5}", ticker)
            and (not filter_stopwords or ticker not in _TICKER_STOPWORDS)
            and ticker not in cleaned
        ):
            cleaned.append(ticker)
    return cleaned


def _clean_date_range(value: Any) -> dict[str, str | None] | None:
    if not isinstance(value, dict):
        return None
    start = _clean_date(value.get("start"))
    end = _clean_date(value.get("end"))
    return {"start": start, "end": end} if start or end else None


def _clean_date(value: Any) -> str | None:
    text = str(value or "").strip()
    return text if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", text) else None


def _extract_universe(text: str) -> str | None:
    match = re.search(r"\bU\d{1,2}\b", str(text or "").upper())
    return match.group(0) if match else None


def _extract_tickers(text: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][A-Z0-9.]{0,5}\b", str(text or "").upper())
    return _clean_tickers(candidates)


def _extract_date_range(text: str) -> dict[str, str | None] | None:
    dates = _DATE_PATTERN.findall(str(text or ""))
    if not dates:
        return None
    return {"start": dates[0], "end": dates[1] if len(dates) > 1 else None}
