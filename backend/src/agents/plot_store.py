"""
plot_store.py — Shared in-memory store for interactive plot data.

Isolated in its own module to avoid circular imports between:
  live_data_tools → custom_plot_tool → price_series_tool → live_data_tools

Both live_data_tools and custom_plot_tool import GLOBAL_PLOT_IDS from here.
"""

# session_id → {"title": str, "data": list[{"date", "ticker", "value"}]}

from typing import Any

GLOBAL_PLOT_IDS: dict[str, Any] = {}
GLOBAL_PLOT_DATA: dict[str, Any] = {}


def register_plot(plot_id: str, plot_spec: dict[str, Any], session_id: str | None = None) -> None:
    """Make a generated PlotSpec immediately fetchable by the chat UI."""
    clean_plot_id = str(plot_id or "").strip()
    if not clean_plot_id:
        return

    GLOBAL_PLOT_DATA[clean_plot_id] = plot_spec

    clean_session_id = str(session_id or "").strip()
    if not clean_session_id:
        return

    if clean_session_id not in GLOBAL_PLOT_IDS:
        GLOBAL_PLOT_IDS[clean_session_id] = []
    if isinstance(GLOBAL_PLOT_IDS[clean_session_id], str):
        GLOBAL_PLOT_IDS[clean_session_id] = [GLOBAL_PLOT_IDS[clean_session_id]]
    if clean_plot_id not in GLOBAL_PLOT_IDS[clean_session_id]:
        GLOBAL_PLOT_IDS[clean_session_id].append(clean_plot_id)

