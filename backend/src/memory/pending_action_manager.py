from __future__ import annotations

from typing import Any


YES_MARKERS = {
    "yes",
    "y",
    "ok",
    "okay",
    "sure",
    "do it",
    "use them",
    "use it",
    "use proxy",
    "use the proxy",
    "use equal weight",
    "use equal-weight",
    "use equal weight proxy",
    "use equal-weight proxy",
    "then plot",
    "plot it",
    "use them and plot",
    "go ahead",
}
NO_MARKERS = {"no", "n", "cancel", "stop", "do not", "don't", "never mind", "nevermind"}

EXPLICIT_NEW_COMMAND_MARKERS = (
    "list ",
    "fetch ",
    "show ",
    "get ",
    "calculate ",
    "compute ",
    "generate ",
    "plot ",
    "run ",
)

EXPLICIT_NEW_COMMAND_PHRASES = (
    "list the tickers",
    "list tickers",
    "fetch the tickers",
    "fetch tickers",
    "fetch the u",
    "show universe",
    "show the universe",
    "show tickers",
    "show the tickers",
    "calculate real values",
    "calculate the real values",
    "compute real values",
    "real current weights",
)


def is_explicit_new_command(message: str) -> bool:
    normalized = " ".join(str(message or "").lower().split())
    if not normalized:
        return False
    if normalized in YES_MARKERS or normalized in NO_MARKERS:
        return False
    if any(phrase in normalized for phrase in EXPLICIT_NEW_COMMAND_PHRASES):
        return True
    return any(normalized.startswith(marker) for marker in EXPLICIT_NEW_COMMAND_MARKERS)


def classify_confirmation(message: str) -> str | None:
    normalized = " ".join(str(message or "").lower().split())
    if not normalized:
        return None
    if normalized in YES_MARKERS:
        return "yes"
    if normalized in NO_MARKERS:
        return "no"
    return None


def build_equal_weight_pending_action(reason: str, target_plot_id: str, tickers: list[str], universe: str | None) -> dict[str, Any]:
    return {
        "type": "use_equal_weight_proxy",
        "reason": reason,
        "target_plot_id": target_plot_id,
        "requires_confirmation": True,
        "target_universe": universe,
        "target_tickers": list(tickers),
    }


def apply_pending_action(state: dict[str, Any], message: str) -> tuple[dict[str, Any], str | None]:
    pending = state.get("pending_action")
    if not pending:
        return state, None

    if is_explicit_new_command(message):
        state["pending_action"] = None
        state["missing_inputs"] = []
        return state, None

    confirmation = classify_confirmation(message)
    if confirmation == "no":
        state["pending_action"] = None
        state["missing_inputs"] = []
        state["last_plot_status"] = "cancelled"
        return state, "cancelled"

    if confirmation != "yes":
        return state, None

    if pending.get("type") == "use_equal_weight_proxy":
        tickers = [str(t).strip().upper() for t in pending.get("target_tickers", []) if str(t).strip()]
        weight = round(100.0 / len(tickers), 10) if tickers else 0.0
        state["active_weights"] = {
            "type": "equal_weight_proxy",
            "weights": {ticker: weight for ticker in tickers},
            "source": "pending_action_user_approved",
            "approved_by_user": True,
        }
        state["last_plot_id"] = pending.get("target_plot_id")
        state["last_chart_type"] = "bar"
        state["last_plot_status"] = "ready"
        state["missing_inputs"] = []
        state["pending_action"] = None
        return state, "executed"

    return state, None
