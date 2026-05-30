"""
plot_store.py — Shared in-memory store for interactive plot data.

Isolated in its own module to avoid circular imports between:
  live_data_tools → custom_plot_tool → price_series_tool → live_data_tools

Both live_data_tools and custom_plot_tool import GLOBAL_PLOT_IDS from here.
"""

# session_id → {"title": str, "data": list[{"date", "ticker", "value"}]}

from typing import Any
GLOBAL_PLOT_IDS: dict[str, Any] = {}
