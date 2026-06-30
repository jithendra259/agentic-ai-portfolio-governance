from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from typing import Any


logger = logging.getLogger(__name__)

_SHORT_FOLLOWUPS = {
    "plot it",
    "plot the scatter plot",
    "show it",
    "do it",
    "use that",
    "same",
    "plot that",
}
_SYSTEMIC_RISK_WEIGHT_REQUEST = re.compile(
    r"(?:systemic|structural)\s+risk(?:\s+score)?\s+(?:vs\.?|versus|against)\s+portfolio\s+weight",
    re.IGNORECASE,
)
_GENERIC_SCATTER_REQUEST = re.compile(r"\b(?:plot|show|draw|generate)?\s*(?:the\s+)?scatter(?:\s+plot)?\b", re.IGNORECASE)


def _normalize_message(message: str) -> str:
    return re.sub(r"[^a-z0-9/+.]+", " ", str(message or "").lower()).strip()


def is_short_followup(message: str) -> bool:
    return _normalize_message(message) in _SHORT_FOLLOWUPS


def _decode_payload(result: Any) -> dict[str, Any] | None:
    content = getattr(result, "content", result)
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    try:
        decoded = json.loads(content)
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def store_latest_governance_run(session_state: dict[str, Any], result: Any) -> dict[str, Any]:
    state = deepcopy(session_state or {})
    payload = _decode_payload(result)
    if not payload or not str(payload.get("status") or "").startswith(("success", "partial_success")):
        return state

    optimization = payload.get("optimization") if isinstance(payload.get("optimization"), dict) else {}
    systemic = payload.get("systemic_risk") if isinstance(payload.get("systemic_risk"), dict) else {}
    tickers = [str(ticker).upper() for ticker in payload.get("valid_tickers", []) if str(ticker).strip()]
    instability = optimization.get("instability_index")
    try:
        regime = "crisis" if float(instability) > 0.5 else "calm"
    except (TypeError, ValueError):
        regime = None

    latest_run = {
        "universe": state.get("active_universe"),
        "tickers": tickers,
        "target_date": payload.get("target_date"),
        "data_source": payload.get("data_source"),
        "data_sources": dict(payload.get("data_sources") or {}) if isinstance(payload.get("data_sources"), dict) else {},
        "risk_profile": optimization.get("risk_tolerance"),
        "weights": dict(optimization.get("weights") or {}),
        "systemic_risk_scores": dict(systemic.get("scores") or {}),
        "systemic_risk_method": systemic.get("method"),
        "annualized_return": optimization.get("expected_annualized_return"),
        "expected_cvar": optimization.get("expected_cvar_95"),
        "instability_index": instability,
        "regime": regime,
        "graph_penalty": optimization.get("lambda_t"),
        "hhi": optimization.get("hhi"),
        "effective_holdings": optimization.get("effective_number_of_holdings"),
        "max_weight_constraint": optimization.get("max_weight_constraint"),
        "max_observed_weight": optimization.get("max_observed_weight"),
        "effective_window_start": optimization.get("effective_window_start"),
        "effective_window_end": optimization.get("effective_window_end"),
    }
    state["active_tickers"] = tickers
    state["active_ticker_count"] = len(tickers)
    state["latest_governance_run"] = latest_run
    logger.info(
        "Stored latest_governance_run for session %s with %d tickers and target_date=%s",
        state.get("session_id"),
        len(tickers),
        latest_run.get("target_date"),
    )
    return state


def _is_systemic_risk_weight_request(message: str) -> bool:
    return bool(_SYSTEMIC_RISK_WEIGHT_REQUEST.search(str(message or "")))


def build_systemic_risk_weight_plot(latest_governance_run: dict[str, Any]) -> dict[str, Any]:
    scores = latest_governance_run.get("systemic_risk_scores") or {}
    weights = latest_governance_run.get("weights") or {}
    points = []
    for ticker in scores:
        if ticker not in weights or scores.get(ticker) is None or weights.get(ticker) is None:
            continue
        try:
            points.append(
                {
                    "id": str(ticker),
                    "label": str(ticker),
                    "x": float(scores[ticker]),
                    "y": float(weights[ticker]) * 100.0,
                }
            )
        except (TypeError, ValueError):
            continue

    return {
        "plot_type": "scatter",
        "chart_type": "scatter",
        "title": "Systemic Risk Score vs Portfolio Weight",
        "x_label": "Systemic Risk Score",
        "y_label": "Portfolio Weight (%)",
        "y_format": "percent",
        "data_source": "latest_governance_run",
        "target_date": latest_governance_run.get("target_date"),
        "grid": {"horizontal": True, "vertical": True},
        "series": [
            {
                "name": "Governance allocation",
                "label": "Ticker",
                "data": points,
                "highlightScope": {"highlight": "item", "fade": "global"},
            }
        ],
    }


def resolve_governance_plot_followup(
    session_state: dict[str, Any], message: str
) -> dict[str, Any] | None:
    state = deepcopy(session_state or {})
    latest_run = state.get("latest_governance_run")
    if not isinstance(latest_run, dict) or not latest_run:
        return None

    pending = state.get("pending_plot_request")
    requested_axes = _is_systemic_risk_weight_request(message)
    pending_is_governance_scatter = isinstance(pending, dict) and pending == {
        "chart_type": "scatter",
        "x": "systemic_risk_score",
        "y": "portfolio_weight",
        "source": "latest_governance_run",
        "label": "ticker",
    }

    if requested_axes:
        pending = {
            "chart_type": "scatter",
            "x": "systemic_risk_score",
            "y": "portfolio_weight",
            "source": "latest_governance_run",
            "label": "ticker",
        }
        state["pending_plot_request"] = pending
        logger.info("Stored pending_plot_request for session %s: %s", state.get("session_id"), pending)
    elif is_short_followup(message) and pending_is_governance_scatter:
        logger.info(
            "Resolved short follow-up from pending_plot_request for session %s: %s",
            state.get("session_id"),
            message,
        )
    elif _GENERIC_SCATTER_REQUEST.search(str(message or "")):
        return {
            "action": "clarify_axes",
            "response": (
                "Which governance metrics should I use for the scatter plot? "
                "For example: systemic risk score vs portfolio weight."
            ),
            "state": state,
        }
    else:
        return None

    plot_spec = build_systemic_risk_weight_plot(latest_run)
    if not plot_spec["series"][0]["data"]:
        return {
            "action": "unavailable",
            "response": "The latest governance run has no tickers with both a systemic risk score and a portfolio weight.",
            "state": state,
        }
    return {
        "action": "plot",
        "response": "Using the latest saved governance run, here is Systemic Risk Score vs Portfolio Weight.",
        "plot_spec": plot_spec,
        "state": state,
    }
