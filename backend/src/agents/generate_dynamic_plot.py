import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
SUPPORTED_PLOTS = {"heatmap", "pie", "line", "bar", "network"}


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


def _extract_weights(data: dict) -> pd.Series:
    weights = data.get("weights") or data.get("optimal_weights") or data
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Pie chart data must include a non-empty 'weights' mapping.")

    series = pd.Series(weights, dtype=float)
    series = series[series > 0].sort_values(ascending=False)
    if series.empty:
        raise ValueError("Pie chart weights are empty after filtering non-positive values.")
    return series


def _extract_price_history(data: dict) -> pd.DataFrame:
    price_history = data.get("price_history") or data.get("series") or data
    if not isinstance(price_history, dict) or not price_history:
        raise ValueError("Line chart data must include a 'price_history' mapping.")

    series_map = {}
    for ticker, rows in price_history.items():
        if isinstance(rows, dict):
            frame = pd.DataFrame(
                [{"date": key, "close": value} for key, value in rows.items()]
            )
        else:
            frame = pd.DataFrame(rows)

        if frame.empty:
            continue

        date_col = "date" if "date" in frame.columns else "Date" if "Date" in frame.columns else None
        value_col = "close" if "close" in frame.columns else "Close" if "Close" in frame.columns else None
        if date_col is None or value_col is None:
            continue

        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
        frame = frame.dropna(subset=[date_col, value_col]).sort_values(date_col)
        if frame.empty:
            continue

        series_map[str(ticker).upper()] = frame.set_index(date_col)[value_col]

    if not series_map:
        raise ValueError("No valid historical price series were found for the line chart.")

    return pd.DataFrame(series_map).sort_index()


def _extract_scores(data: dict) -> pd.Series:
    scores = data.get("scores") or data.get("risk_scores") or data
    if not isinstance(scores, dict) or not scores:
        raise ValueError("Bar chart data must include a non-empty 'scores' mapping.")

    series = pd.Series(scores, dtype=float).sort_values(ascending=False)
    if series.empty:
        raise ValueError("Bar chart scores are empty.")
    return series


def _extract_network_payload(data: dict) -> tuple[list[dict], dict[str, float]]:
    edges = data.get("holder_edges") or data.get("edges") or []
    if not isinstance(edges, list):
        raise ValueError("Network plot data must include 'holder_edges' as a list.")

    risk_scores = data.get("risk_scores") or data.get("scores") or {}
    if risk_scores and not isinstance(risk_scores, dict):
        raise ValueError("Network plot risk scores must be a dictionary.")

    return edges, {str(k).upper(): float(v) for k, v in risk_scores.items()}


@tool
def generate_financial_plot(data: dict, plot_type: str, title: str) -> str:
    """
    Generate a dark-theme financial plot from structured data and save it to outputs/.
    Returns a markdown image link so the UI can render the chart.
    """
    try:
        payload = _coerce_dict(data)
        normalized_plot_type = str(plot_type or "").strip().lower()
        plot_title = str(title or "Financial Plot").strip()

        if normalized_plot_type not in SUPPORTED_PLOTS:
            return (
                "Unable to generate plot: unsupported plot type "
                f"'{plot_type}'. Supported types are: {', '.join(sorted(SUPPORTED_PLOTS))}."
            )

        _apply_dark_theme()
        fig, ax = plt.subplots(figsize=(10, 6))

        if normalized_plot_type == "heatmap":
            matrix = _extract_matrix(payload)
            sns.heatmap(matrix, cmap="mako", center=0, annot=False, linewidths=0.25, ax=ax)
            ax.set_title(plot_title, fontsize=14, fontweight="bold")

        elif normalized_plot_type == "pie":
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(8, 8))
            weights = _extract_weights(payload)
            colors = sns.color_palette("crest", n_colors=len(weights))
            ax.pie(
                weights.values,
                labels=weights.index.tolist(),
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                wedgeprops={"edgecolor": "#111827", "linewidth": 1.0},
                textprops={"color": "#f3f4f6"},
            )
            ax.set_title(plot_title, fontsize=14, fontweight="bold")

        elif normalized_plot_type == "line":
            prices = _extract_price_history(payload)
            sns.lineplot(data=prices, dashes=False, linewidth=2.0, ax=ax)
            ax.set_title(plot_title, fontsize=14, fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("Close Price")
            ax.tick_params(axis="x", rotation=30)
            ax.legend(title="Ticker", frameon=False)

        elif normalized_plot_type == "bar":
            scores = _extract_scores(payload)
            sns.barplot(x=scores.index.tolist(), y=scores.values.tolist(), palette="crest", ax=ax)
            ax.set_title(plot_title, fontsize=14, fontweight="bold")
            ax.set_xlabel("Ticker")
            ax.set_ylabel("Risk Score")
            ax.tick_params(axis="x", rotation=20)

        elif normalized_plot_type == "network":
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
            stock_list = [node for node, attrs in graph.nodes(data=True) if attrs.get("bipartite") == 0]
            holder_list = [node for node, attrs in graph.nodes(data=True) if attrs.get("bipartite") == 1]
            stock_sizes = [
                900 + 1800 * float(risk_scores.get(node, 0.0))
                for node in stock_list
            ]

            nx.draw_networkx_nodes(
                graph,
                positions,
                nodelist=stock_list,
                node_color="#22d3ee",
                node_size=stock_sizes,
                edgecolors="#e5e7eb",
                linewidths=1.2,
                ax=ax,
            )
            if holder_list:
                nx.draw_networkx_nodes(
                    graph,
                    positions,
                    nodelist=holder_list,
                    node_color="#f59e0b",
                    node_size=700,
                    edgecolors="#e5e7eb",
                    linewidths=1.0,
                    ax=ax,
                )

            nx.draw_networkx_edges(graph, positions, alpha=0.35, width=1.2, edge_color="#6b7280", ax=ax)
            nx.draw_networkx_labels(graph, positions, font_size=8, font_color="#f9fafb", ax=ax)
            ax.set_title(plot_title, fontsize=14, fontweight="bold")
            ax.set_axis_off()

        plot_path = _save_current_plot(plot_title, normalized_plot_type)
        return f"Plot generated successfully: ![{plot_title}]({plot_path})"

    except Exception as e:
        plt.close("all")
        return f"Unable to generate plot due to a formatting or rendering error: {type(e).__name__}: {str(e)}"
