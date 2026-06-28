from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


U1_TECH_TICKERS = [
    "AAPL", "ADBE", "ADI", "AMAT", "AMD", "AVGO", "CRM", "CSCO", "IBM", "INTU",
    "KLAC", "LRCX", "MSFT", "MU", "NOW", "NVDA", "ORCL", "PANW", "QCOM", "TXN",
]

KNOWN_UNIVERSE_TICKERS = {
    "U1": U1_TECH_TICKERS,
}

PROJECT_POLICY_MEMORY = {
    "advisory_only_language": True,
    "forbidden_trading_language": ["buy", "sell", "trade signal"],
    "chart_system": "MUI",
    "disallowed_chart_systems": ["Google Charts"],
    "render_missing_data_plots": False,
    "u1_resolution": "U1 must resolve to all 20 U1 tickers unless the user requests a subset.",
    "smart_view_scope": "Smart View should show selected plots only, not all 88 plots.",
    "bar_plot_preservation": "Bar chart requests must preserve plot_id, universe, chart_type, and bar_mode.",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_session_state(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "active_universe": None,
        "active_tickers": [],
        "active_ticker_count": 0,
        "active_sector": None,
        "active_date_range": {"start": None, "end": None},
        "active_weights": {
            "type": "missing",
            "weights": {},
            "source": None,
            "approved_by_user": False,
        },
        "active_advisory_weights": {
            "available": False,
            "weights": {},
            "source": None,
            "target_date": None,
        },
        "active_analysis_id": None,
        "last_successful_analysis": None,
        "last_successful_plot": None,
        "latest_governance_run": None,
        "pending_plot_request": None,
        "approved_proxies": [],
        "last_intent": None,
        "last_sub_intent": None,
        "last_plot_id": None,
        "last_chart_type": None,
        "last_bar_mode": None,
        "last_plot_status": None,
        "continuity_memory": {"strategy": "bounded_priority_recency_memory", "anchors": [], "top": {}},
        "conversation_state": {
            "messages": [],
            "summary": "",
            "current_topic": None,
            "selected_tickers": [],
            "current_strategy": None,
            "comparison_target": None,
            "current_metrics": [],
            "last_run_id": None,
            "user_intent": None,
            "resolved_question": None,
            "safety_context": "advisory_only_no_trading_execution",
        },
        "current_plan": None,
        "last_plan": None,
        "last_modules_called": [],
        "last_modules_skipped": [],
        "missing_inputs": [],
        "pending_action": None,
        "warnings": [],
        "last_user_message": None,
        "last_response_summary": None,
        "last_plot": None,
        "updated_at": utc_now_iso(),
    }


def clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(state)
