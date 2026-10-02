"""Formatting & Response Sanitization Layer.

Extracts tickers, normalizes tool outputs, humanizes solver and pipeline statuses,
and formats markdown reports for advisory governance.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

logger = logging.getLogger(__name__)


def _sanitize_user_visible_response(content: str, scratchpad: dict[str, Any] | None = None) -> str:
    """Remove internal-only payloads, strip raw code, and format scratchpad metrics."""
    text = str(content or "").strip()
    if not text:
        return text

    def _scratchpad_summary() -> str:
        if not isinstance(scratchpad, dict) or not scratchpad:
            return ""
        lines = ["Exact metrics available:"]
        for metric_name, details in scratchpad.items():
            if isinstance(details, dict):
                exact_value = details.get("exact_value")
                context = str(details.get("context") or "").strip()
            else:
                exact_value = details
                context = ""
            line = f"- `{metric_name}`: {exact_value}"
            if context:
                line += f" ({context})"
            lines.append(line)
        return "\n".join(lines)

    try:
        payload = json.loads(text)
    except Exception:
        payload = None

    if isinstance(payload, dict) and payload.get("scratchpad_save"):
        summary = _scratchpad_summary()
        if summary:
            return summary
        metric_name = str(payload.get("metric_name") or "metric").strip()
        exact_value = payload.get("exact_value")
        context = str(payload.get("context") or "").strip()
        lines = [f"Saved exact metric `{metric_name}`: {exact_value}."]
        if context:
            lines.append(f"Context: {context}.")
        return "\n".join(lines)

    # Strip unresolved attachment://
    attachment_pattern = re.compile(r"!\[[^\]]*\]\(attachment://[^)]+\)", flags=re.IGNORECASE)
    if attachment_pattern.search(text):
        text = attachment_pattern.sub("", text)
        pending_note = (
            "Chart rendering is still pending because the assistant did not receive "
            "a registered plot artifact for that attachment."
        )
        if pending_note not in text:
            text = f"{text.strip()}\n\n{pending_note}".strip()

    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _message_content_to_text(message_or_content) -> str:
    content = getattr(message_or_content, "content", message_or_content)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content) if content is not None else ""


def _extract_tickers_from_text(text: str) -> list[str]:
    tickers = []
    patterns = [
        r"(?m)(?:^|\s)-\s*([A-Z]{1,5})(?::|\s|\()",
        r"(?m)^Ticker:\s*([A-Z]{1,5})\b",
        r"(?m)^([A-Z]{1,5}):\s+",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            t = match.group(1).upper()
            if t not in tickers:
                tickers.append(t)

    for match in re.finditer(r"(?m)^Tickers:\s*([A-Z,\s]+)$", text):
        for token in re.split(r"[,\s]+", match.group(1).upper()):
            if token and re.fullmatch(r"[A-Z]{1,5}", token) and token not in tickers:
                tickers.append(token)

    return tickers


def _extract_portfolio_from_messages(messages: list[BaseMessage]) -> list[str]:
    for message in reversed(messages):
        raw_text = _message_content_to_text(message)

        if isinstance(message, ToolMessage):
            name = getattr(message, "name", "")
            if name == "run_full_governance_pipeline":
                try:
                    payload = json.loads(raw_text)
                    if isinstance(payload, dict):
                        valid = payload.get("valid_tickers", [])
                        if isinstance(valid, list) and valid:
                            return [str(t).upper() for t in valid if str(t).strip()]
                except Exception:
                    pass

            if name == "get_price_series_for_analysis":
                try:
                    payload = json.loads(raw_text)
                    if isinstance(payload, dict):
                        included = payload.get("tickers_included", [])
                        if isinstance(included, list) and included:
                            return [str(t).upper() for t in included if str(t).strip()]
                except Exception:
                    pass

            extracted = _extract_tickers_from_text(raw_text)
            if extracted:
                return extracted

        if isinstance(message, AIMessage):
            extracted = _extract_tickers_from_text(raw_text)
            if extracted:
                return extracted
    return []


def _extract_latest_governance_payload(messages: list[BaseMessage]) -> Tuple[Optional[dict], str]:
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "run_full_governance_pipeline":
            raw = _message_content_to_text(message)
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed, raw
            except Exception:
                return None, raw
    return None, ""


def _extract_latest_tool_output(messages: list[BaseMessage]) -> tuple[str, str]:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            return getattr(message, "name", ""), _message_content_to_text(message)
    return "", ""


def _humanize_status(status: str) -> str:
    status_map = {
        "success": "Governance pipeline completed successfully.",
        "partial_success_some_requested_tickers_were_dropped_due_to_missing_data": (
            "Governance pipeline completed, but some requested tickers were dropped due to insufficient data."
        ),
        "error_no_tickers_provided": "No tickers were provided for governance analysis.",
        "error_no_valid_tickers_provided": "No valid tickers were provided for governance analysis.",
        "error_invalid_target_date": "Target date invalid. Please use YYYY-MM-DD.",
        "error_no_requested_tickers_found_in_local_mongodb": "Requested tickers not found in historical records.",
        "error_fewer_than_two_valid_tickers_after_history_validation": "Fewer than two valid tickers available for portfolio allocation.",
        "error_optimization_failed": "The G-CVaR optimizer could not produce a stable allocation.",
    }
    return status_map.get(status, status.replace("_", " ").strip().capitalize() or "Governance pipeline returned status.")


def _build_governance_markdown(payload: Optional[dict], raw_text: str) -> str:
    if not payload:
        return raw_text or "Unable to generate a response."

    data_sources = payload.get("data_sources", {}) if isinstance(payload.get("data_sources"), dict) else {}
    source_line = ", ".join(sorted({str(v) for v in data_sources.values() if v})) if data_sources else "MongoDB/yfinance"

    lines = [
        "## Historical Governance Report",
        f"- Status: {payload.get('status', 'unknown')}",
        f"- Target date: {payload.get('target_date', 'unknown')}",
        f"- Valid tickers used: {', '.join(payload.get('valid_tickers', [])) or 'None'}",
        f"- Data source: {source_line}",
        "- Advisory only: no execution, no trading, no broker actions",
        "",
        _humanize_status(str(payload.get("status", ""))),
    ]

    optimization = payload.get("optimization", {}) if isinstance(payload.get("optimization"), dict) else {}
    weights = optimization.get("weights", {}) if isinstance(optimization.get("weights"), dict) else {}
    if weights:
        lines.append("- Suggested exposure weights:")
        for ticker, weight in weights.items():
            lines.append(f"  - {ticker}: {weight:.2%}")

    expected_return = optimization.get("expected_annualized_return")
    expected_cvar = optimization.get("expected_cvar_95")
    instability_index = optimization.get("instability_index")
    lambda_t = optimization.get("lambda_t")

    if expected_return is not None:
        lines.append(f"- Estimated annualized return: {expected_return:.2%}")
    if expected_cvar is not None:
        lines.append(f"- Expected 95% CVaR: {expected_cvar:.2%}")
    if instability_index is not None:
        lines.append(f"- Instability index (I_t): {instability_index:.4f}")
    if lambda_t is not None:
        lines.append(f"- Graph penalty (lambda_t): {lambda_t:.4f}")

    return "\n".join(lines)
