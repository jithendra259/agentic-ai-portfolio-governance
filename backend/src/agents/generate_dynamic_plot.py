import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.agents.plot_store import GLOBAL_PLOT_IDS


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
SUPPORTED_PLOTS = {"heatmap", "pie", "line", "bar", "network"}

# ---------------------------------------------------------------------------
# Palette used by the frontend InlineChart component (must match COLORS in
# InlineChart.jsx so the legend colours are consistent)
# ---------------------------------------------------------------------------
PALETTE = [
    "#3b82f6",  # blue
    "#10b981",  # emerald
    "#f59e0b",  # amber
    "#ef4444",  # red
    "#8b5cf6",  # purple
    "#ec4899",  # pink
    "#06b6d4",  # cyan
    "#f97316",  # orange
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_dict(data: dict) -> dict:
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        loaded = json.loads(data)
        if isinstance(loaded, dict):
            return loaded
    raise ValueError("Plot data must be a dictionary or a JSON object string.")


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return cleaned.lower() or "plot"


def _apply_dark_theme() -> None:
    plt.style.use("dark_background")
    sns.set_theme(style="darkgrid", palette="crest")
    plt.rcParams.update(
        {
            "figure.facecolor": "#0b1020",
            "axes.facecolor": "#111827",
            "axes.edgecolor": "#9ca3af",
            "axes.labelcolor": "#e5e7eb",
            "xtick.color": "#d1d5db",
            "ytick.color": "#d1d5db",
            "grid.color": "#374151",
            "text.color": "#f3f4f6",
        }
    )


def _save_current_plot(title: str, plot_type: str) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{_slugify(plot_type)}_{_slugify(title)}_{timestamp}.png"
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close()
    return f"/outputs/{filename}"


# ---------------------------------------------------------------------------
# Data extractors (unchanged — used only for PNG fallback types)
# ---------------------------------------------------------------------------

def _extract_matrix(data: dict) -> pd.DataFrame:
    matrix = (
        data.get("matrix")
        or data.get("correlation_matrix")
        or data.get("covariance_matrix")
        or data.get("values")
    )
    if matrix is None:
        raise ValueError("Heatmap data must include 'matrix', 'correlation_matrix', or 'covariance_matrix'.")

    df = pd.DataFrame(matrix)
    if df.empty:
        raise ValueError("Heatmap matrix is empty.")
    return df.astype(float)


def _extract_network_payload(data: dict) -> tuple[list[dict], dict[str, float]]:
    edges = data.get("holder_edges") or data.get("edges") or []
    if not isinstance(edges, list):
        raise ValueError("Network plot data must include 'holder_edges' as a list.")

    risk_scores = data.get("risk_scores") or data.get("scores") or {}
    if risk_scores and not isinstance(risk_scores, dict):
        raise ValueError("Network plot risk scores must be a dictionary.")

    return edges, {str(k).upper(): float(v) for k, v in risk_scores.items()}


# ---------------------------------------------------------------------------
# PlotSpec builders for MUI-native chart types
# ---------------------------------------------------------------------------

def _build_line_spec(data: dict, title: str) -> dict:
    """
    Build a PlotSpec for a time-series line chart.

    Accepted input shapes:
      {"price_history": {ticker: [{date, close}, ...], ...}}
      {"series": {name: [{date, value}, ...], ...}}
      {name: [{date, close}, ...], ...}          (bare dict)
    """
    raw = data.get("price_history") or data.get("series") or data
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Line chart data must contain a price_history or series mapping.")

    series = []
    for i, (name, rows) in enumerate(raw.items()):
        if isinstance(rows, dict):
            rows = [{"date": k, "close": v} for k, v in rows.items()]
        if not isinstance(rows, list) or not rows:
            continue
        frame = pd.DataFrame(rows)
        date_col = next((c for c in ("date", "Date") if c in frame.columns), None)
        val_col = next((c for c in ("close", "Close", "value", "Value") if c in frame.columns), None)
        if date_col is None or val_col is None:
            continue
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame[val_col] = pd.to_numeric(frame[val_col], errors="coerce")
        frame = frame.dropna(subset=[date_col, val_col]).sort_values(date_col)
        if frame.empty:
            continue
        pts = [
            {"x": row[date_col].strftime("%Y-%m-%d"), "y": round(float(row[val_col]), 6)}
            for _, row in frame.iterrows()
        ]
        series.append({"name": str(name).upper(), "color": PALETTE[i % len(PALETTE)], "data": pts})

    if not series:
        raise ValueError("No valid series found for line chart.")

    return {
        "plot_type": "line",
        "title": title,
        "x_label": "Date",
        "x_type": "time",
        "y_label": "Price",
        "series": series,
    }


def _build_bar_spec(data: dict, title: str) -> dict:
    """
    Build a PlotSpec for a bar chart.

    Accepted input shapes:
      {"scores": {ticker: value, ...}}
      {"risk_scores": {ticker: value, ...}}
      {ticker: value, ...}                       (bare dict of floats)
    """
    raw = data.get("scores") or data.get("risk_scores") or data
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Bar chart data must contain a scores or risk_scores mapping.")

    items = [(str(k), float(v)) for k, v in raw.items() if v is not None]
    if not items:
        raise ValueError("Bar chart scores are empty.")
    items.sort(key=lambda t: t[1], reverse=True)

    pts = [{"x": k, "y": round(v, 6)} for k, v in items]
    return {
        "plot_type": "bar",
        "title": title,
        "x_label": "Ticker",
        "x_type": "band",
        "y_label": "Score",
        "series": [{"name": "Score", "color": PALETTE[0], "data": pts}],
    }


def _build_pie_spec(data: dict, title: str) -> dict:
    """
    Build a PlotSpec for a pie / donut chart.

    Accepted input shapes:
      {"weights": {label: weight, ...}}
      {"optimal_weights": {label: weight, ...}}
      {label: weight, ...}                       (bare dict of floats)
    """
    raw = data.get("weights") or data.get("optimal_weights") or data
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Pie chart data must contain a weights mapping.")

    items = [(str(k), float(v)) for k, v in raw.items() if v and float(v) > 0]
    if not items:
        raise ValueError("Pie chart weights are empty after filtering non-positive values.")
    items.sort(key=lambda t: t[1], reverse=True)

    pts = [
        {"x": k, "y": round(v, 6), "color": PALETTE[i % len(PALETTE)]}
        for i, (k, v) in enumerate(items)
    ]
    return {
        "plot_type": "pie",
        "title": title,
        "series": [{"name": "Allocation", "data": pts}],
    }


# ---------------------------------------------------------------------------
# Main tool — MUI-native for line/bar/pie; PNG fallback for heatmap/network
# ---------------------------------------------------------------------------

@tool
def generate_financial_plot(
    data: dict,
    plot_type: str,
    title: str,
    config: RunnableConfig = None,
) -> str:
    """
    Generate a financial chart from structured data.

    For plot_type = line / bar / pie  → stores an interactive PlotSpec in
    GLOBAL_PLOT_DATA so the MUI frontend renders an interactive chart in the
    chat bubble (no PNG saved).

    For plot_type = heatmap / network → falls back to saving a PNG (no MUI X
    Charts equivalent exists yet) and returns a markdown image link.
    """
    try:
        payload = _coerce_dict(data)
        normalized = str(plot_type or "").strip().lower()
        plot_title = str(title or "Financial Plot").strip()

        if normalized not in SUPPORTED_PLOTS:
            return (
                f"Unable to generate plot: unsupported plot type '{plot_type}'. "
                f"Supported types are: {', '.join(sorted(SUPPORTED_PLOTS))}."
            )

        # --- MUI-native interactive chart types ---
        if normalized == "line":
            spec = _build_line_spec(payload, plot_title)
        elif normalized == "bar":
            spec = _build_bar_spec(payload, plot_title)
        elif normalized == "pie":
            spec = _build_pie_spec(payload, plot_title)
        else:
            spec = None  # falls through to PNG path below

        if spec is not None:
            import uuid
            from src.memory.mongodb_memory_layer import MongoMemoryManager
            
            plot_id = str(uuid.uuid4())
            try:
                mongo = MongoMemoryManager()
                mongo.store_plot(plot_id, spec, ttl_days=1)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Failed to store plot in MongoDB: {e}")

            session_id = (
                config.get("configurable", {}).get("thread_id", "default")
                if config
                else "default"
            )
            from src.agents.plot_store import GLOBAL_PLOT_IDS
            if session_id not in GLOBAL_PLOT_IDS:
                GLOBAL_PLOT_IDS[session_id] = []
            if isinstance(GLOBAL_PLOT_IDS[session_id], str):
                GLOBAL_PLOT_IDS[session_id] = [GLOBAL_PLOT_IDS[session_id]]
            GLOBAL_PLOT_IDS[session_id].append(plot_id)
            return f"Chart ready: {plot_title}"

        # --- PNG fallback for heatmap / network ---
        _apply_dark_theme()
        fig, ax = plt.subplots(figsize=(10, 6))

        if normalized == "heatmap":
            matrix = _extract_matrix(payload)
            sns.heatmap(matrix, cmap="mako", center=0, annot=False, linewidths=0.25, ax=ax)
            ax.set_title(plot_title, fontsize=14, fontweight="bold")

        elif normalized == "network":
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(11, 8))
            edges, risk_scores = _extract_network_payload(payload)
            graph = nx.Graph()

            stock_nodes = set(risk_scores.keys())
            for edge in edges:
                ticker = str(edge.get("ticker", "")).upper()
                holder = str(edge.get("holder", "")).strip()
                weight = float(edge.get("weight", 0.0))
                if ticker:
                    stock_nodes.add(ticker)
                if ticker and holder:
                    graph.add_node(ticker, bipartite=0)
                    graph.add_node(holder, bipartite=1)
                    graph.add_edge(ticker, holder, weight=weight)

            for ticker in stock_nodes:
                graph.add_node(ticker, bipartite=0)

            if graph.number_of_nodes() == 0:
                raise ValueError("Network plot data did not include any valid nodes.")

            positions = nx.spring_layout(graph, seed=42, k=0.8)
            stock_list = [n for n, a in graph.nodes(data=True) if a.get("bipartite") == 0]
            holder_list = [n for n, a in graph.nodes(data=True) if a.get("bipartite") == 1]
            stock_sizes = [900 + 1800 * float(risk_scores.get(n, 0.0)) for n in stock_list]

            nx.draw_networkx_nodes(graph, positions, nodelist=stock_list, node_color="#22d3ee",
                                   node_size=stock_sizes, edgecolors="#e5e7eb", linewidths=1.2, ax=ax)
            if holder_list:
                nx.draw_networkx_nodes(graph, positions, nodelist=holder_list, node_color="#f59e0b",
                                       node_size=700, edgecolors="#e5e7eb", linewidths=1.0, ax=ax)
            nx.draw_networkx_edges(graph, positions, alpha=0.35, width=1.2, edge_color="#6b7280", ax=ax)
            nx.draw_networkx_labels(graph, positions, font_size=8, font_color="#f9fafb", ax=ax)
            ax.set_title(plot_title, fontsize=14, fontweight="bold")
            ax.set_axis_off()

        plot_path = _save_current_plot(plot_title, normalized)
        return f"Chart ready: ![{plot_title}]({plot_path})"

    except Exception as e:
        plt.close("all")
        return f"Unable to generate plot due to a formatting or rendering error: {type(e).__name__}: {str(e)}"
