from __future__ import annotations


def acknowledge_requested_plots(plot_ids: list[str]) -> str:
    unique_ids = []
    for plot_id in plot_ids:
        if plot_id and plot_id not in unique_ids:
            unique_ids.append(plot_id)
    return "Generated or queued plots acknowledged: " + ", ".join(unique_ids)
