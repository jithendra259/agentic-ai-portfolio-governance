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
from src.decision.advisory_labels import normalize_advisory_language


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
SUPPORTED_PLOTS = {
    "heatmap",
    "pie",
    "line",
    "bar",
    "network",
    "scatter",
    "sparkline",
    "sankey",
    "candlestick",
    "funnel",
    "radar",
    "gauge",
    "radial_bar",
    "radial_line",
}

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
    data = _normalize_heatmap_payload(data)
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


def _normalize_heatmap_payload(data: dict) -> dict:
    if not isinstance(data, dict):
        return data
    if data.get("matrix") is not None:
        normalized = dict(data)
        normalized.setdefault("metadata", {})
        normalized["metadata"].setdefault("heatmap_type", data.get("heatmap_type", "generic"))
        return normalized

    heatmap_sources = (
        ("correlation_heatmap", "correlation", "correlation"),
        ("covariance_heatmap", "covariance", "covariance"),
        ("missing_data", "missing", None),
    )
    for key, heatmap_type, value_field in heatmap_sources:
        rows = data.get(key)
        if not isinstance(rows, list) or not rows:
            continue
        matrix: dict[str, dict[str, float]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            x_key = row.get("tickerX") or row.get("ticker") or row.get("x")
            y_key = row.get("tickerY") or row.get("date") or row.get("y")
            if x_key is None or y_key is None:
                continue
            value_key = value_field
            if value_key is None:
                value_key = next((candidate for candidate in row if candidate not in {"ticker", "tickerX", "tickerY", "date", "x", "y"}), None)
            if value_key is None:
                continue
            try:
                value = float(row[value_key])
            except (TypeError, ValueError):
                continue
            matrix.setdefault(str(x_key), {})[str(y_key)] = value
        if matrix:
            normalized = dict(data)
            normalized["matrix"] = matrix
            normalized["metadata"] = {**data.get("metadata", {}), "heatmap_type": heatmap_type}
            return normalized
    return data


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

    The returned PlotSpec can include all MUI X Charts line features.
    The backend decides which features to enable; the frontend reads them.
    All feature fields are optional for backward compatibility.
    """
    # ── Extract the raw time-series mapping ──
    raw = data.get("price_history") or data.get("series") or data
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Line chart data must contain a price_history or series mapping.")

    # ── Per-series feature overrides passed from the tool caller ──
    series_overrides = data.get("series_config", {})  # {name: {area, curve, …}}

    series = []
    total_points = 0
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
        total_points += len(pts)

        # Build per-series entry with all MUI X line chart features
        upper_name = str(name).upper()
        overrides = series_overrides.get(name, series_overrides.get(upper_name, {}))

        entry = {
            "name": upper_name,
            "label": overrides.get("label", upper_name),
            "color": overrides.get("color", PALETTE[i % len(PALETTE)]),
            "data": pts,
        }

        # ── Area fill ──
        if overrides.get("area") is not None:
            entry["area"] = bool(overrides["area"])
        # ── Area baseline ──
        if "baseline" in overrides:
            entry["baseline"] = overrides["baseline"]  # 'min', 'max', or number
        # ── Stacking ──
        if "stack" in overrides:
            entry["stack"] = overrides["stack"]
        if "stackOffset" in overrides:
            entry["stackOffset"] = overrides["stackOffset"]
        # ── Curve interpolation ──
        if "curve" in overrides:
            entry["curve"] = overrides["curve"]
        # ── Marks ──
        if overrides.get("showMark") is not None:
            entry["showMark"] = bool(overrides["showMark"])
        if "shape" in overrides:
            entry["shape"] = overrides["shape"]
        # ── Connect nulls ──
        if overrides.get("connectNulls") is not None:
            entry["connectNulls"] = bool(overrides["connectNulls"])
        # ── Highlight scope ──
        if "highlightScope" in overrides:
            entry["highlightScope"] = overrides["highlightScope"]
        # ── Disable highlight ──
        if overrides.get("disableHighlight") is not None:
            entry["disableHighlight"] = bool(overrides["disableHighlight"])
        # ── Value format ──
        if "value_format" in overrides:
            entry["value_format"] = overrides["value_format"]
        # ── Y-axis binding ──
        if "yAxisId" in overrides:
            entry["yAxisId"] = overrides["yAxisId"]

        series.append(entry)

    if not series:
        raise ValueError("No valid series found for line chart.")

    # ── Build the PlotSpec with all backend-decided features ──
    spec = {
        "plot_type": "line",
        "title": title,
        "x_label": data.get("x_label", "Date"),
        "x_type": data.get("x_type", "time"),
        "y_label": data.get("y_label", "Price"),
        "series": series,
    }

    # ── Grid (default: horizontal lines for readability) ──
    spec["grid"] = data.get("grid", {"horizontal": True})

    # ── Global curve interpolation (smoother lines for financial data) ──
    if "curve" in data:
        spec["curve"] = data["curve"]
    else:
        spec["curve"] = "monotoneX"

    # ── Skip animation (useful for large datasets or SSR) ──
    if data.get("skipAnimation") is not None:
        spec["skipAnimation"] = bool(data["skipAnimation"])

    # ── Global highlight scope ──
    if "highlightScope" in data:
        spec["highlightScope"] = data["highlightScope"]
    else:
        # Default: highlight the hovered series, fade the rest
        spec["highlightScope"] = {"highlight": "series", "fade": "global"}

    # ── Experimental features (position-based pointer interaction) ──
    if "experimentalFeatures" in data:
        spec["experimentalFeatures"] = data["experimentalFeatures"]
    else:
        spec["experimentalFeatures"] = {"enablePositionBasedPointerInteraction": True}

    # ── Y-axis format ──
    if "y_format" in data:
        spec["y_format"] = data["y_format"]

    # ── Multi-axis support ──
    if "yAxis" in data and isinstance(data["yAxis"], list):
        y_axes = []
        for axis in data["yAxis"]:
            ax = {"id": axis.get("id", "default-y-axis")}
            if "label" in axis:
                ax["label"] = axis["label"]
            if "position" in axis:
                ax["position"] = axis["position"]
            if "value_format" in axis:
                ax["value_format"] = axis["value_format"]
            if "width" in axis:
                ax["width"] = axis["width"]
            # ── Y-axis colorMap ──
            if "colorMap" in axis:
                ax["colorMap"] = axis["colorMap"]
            # ── Y-axis domainLimit ──
            ax["domainLimit"] = axis.get("domainLimit", "nice")
            y_axes.append(ax)
        spec["yAxis"] = y_axes

    # ── Recession bands overlay ──
    if "recessions" in data:
        spec["recessions"] = data["recessions"]

    if "animation" in data and isinstance(data["animation"], dict):
        spec["animation"] = data["animation"]

    return spec


def _build_bar_spec(data: dict, title: str) -> dict:
    """
    Build a PlotSpec for a bar chart.

    Accepted input shapes:
      {"scores": {ticker: value, ...}}
      {"risk_scores": {ticker: value, ...}}
      {ticker: value, ...}                       (bare dict of floats)
      {"categories": [...], "series": [{"name": ..., "data": [...]}, ...]}

    The returned PlotSpec can include all MUI X Charts bar features.
    The backend decides which features to enable; the frontend reads them.
    All feature fields are optional for backward compatibility.
    """
    # ── Detect multi-series shape ──
    if "categories" in data and "series" in data and isinstance(data["series"], list):
        categories = data["categories"]
        series = []
        series_overrides = data.get("series_config", {})
        for i, s in enumerate(data["series"]):
            name = s.get("name", f"Series {i+1}")
            overrides = series_overrides.get(name, {})
            entry = {
                "name": name,
                "label": overrides.get("label", s.get("label", name)),
                "color": overrides.get("color", s.get("color", PALETTE[i % len(PALETTE)])),
                "data": [{"x": cat, "y": val} for cat, val in zip(categories, s.get("data", []))],
            }
            # ── Stacking ──
            if s.get("stack") or overrides.get("stack"):
                entry["stack"] = overrides.get("stack", s.get("stack"))
            if s.get("stackOffset") or overrides.get("stackOffset"):
                entry["stackOffset"] = overrides.get("stackOffset", s.get("stackOffset"))
            # ── Bar labels ──
            if s.get("barLabel") is not None or overrides.get("barLabel") is not None:
                entry["barLabel"] = overrides.get("barLabel", s.get("barLabel"))
            if s.get("barLabelPlacement") or overrides.get("barLabelPlacement"):
                entry["barLabelPlacement"] = overrides.get("barLabelPlacement", s.get("barLabelPlacement"))
            # ── Min bar size ──
            if s.get("minBarSize") is not None or overrides.get("minBarSize") is not None:
                entry["minBarSize"] = overrides.get("minBarSize", s.get("minBarSize"))
            # ── Highlight scope ──
            if "highlightScope" in overrides:
                entry["highlightScope"] = overrides["highlightScope"]
            elif "highlightScope" in s:
                entry["highlightScope"] = s["highlightScope"]
            series.append(entry)
    else:
        # ── Single-series shape: scores dict ──
        raw = data.get("scores") or data.get("risk_scores") or data
        # Filter out known non-data keys
        skip_keys = {
            "plot_type", "title", "config", "layout", "borderRadius", "grid",
            "skipAnimation", "highlightScope", "categoryGapRatio", "barGapRatio",
            "x_label", "y_label", "y_format", "xAxis", "yAxis", "series_config",
        }
        if not isinstance(raw, dict) or not raw:
            raise ValueError("Bar chart data must contain a scores or risk_scores mapping.")

        items = []
        for k, v in raw.items():
            if k in skip_keys:
                continue
            try:
                items.append((str(k), float(v)))
            except (TypeError, ValueError):
                continue
        if not items:
            raise ValueError("Bar chart scores are empty.")
        items.sort(key=lambda t: t[1], reverse=True)

        pts = [{"x": k, "y": round(v, 6)} for k, v in items]
        series = [{"name": "Score", "label": "Score", "color": PALETTE[0], "data": pts}]

    if not series:
        raise ValueError("No valid series found for bar chart.")

    # ── Build the PlotSpec with all backend-decided features ──
    spec = {
        "plot_type": "bar",
        "title": title,
        "x_label": data.get("x_label", "Category"),
        "x_type": data.get("x_type", "band"),
        "y_label": data.get("y_label", "Value"),
        "series": series,
    }

    # ── Layout: vertical (default) or horizontal ──
    if "layout" in data:
        spec["layout"] = data["layout"]

    # ── Border radius (rounded bar corners, default 4 for polished look) ──
    spec["borderRadius"] = data.get("borderRadius", 4)

    # ── Grid (default: horizontal lines for readability) ──
    spec["grid"] = data.get("grid", {"horizontal": True})

    # ── Category gap ratio ──
    if "categoryGapRatio" in data:
        spec["categoryGapRatio"] = data["categoryGapRatio"]

    # ── Bar gap ratio ──
    if "barGapRatio" in data:
        spec["barGapRatio"] = data["barGapRatio"]

    # ── Skip animation ──
    if data.get("skipAnimation") is not None:
        spec["skipAnimation"] = bool(data["skipAnimation"])

    # ── Global highlight scope ──
    if "highlightScope" in data:
        spec["highlightScope"] = data["highlightScope"]
    else:
        spec["highlightScope"] = {"highlight": "item", "fade": "global"}

    # ── Y-axis format ──
    if "y_format" in data:
        spec["y_format"] = data["y_format"]

    # ── X-axis config (colorMap, tickPlacement, tickLabelPlacement) ──
    if "xAxis" in data and isinstance(data["xAxis"], list):
        spec["xAxis"] = data["xAxis"]

    # ── Y-axis config (colorMap, width, label) ──
    if "yAxis" in data and isinstance(data["yAxis"], list):
        spec["yAxis"] = data["yAxis"]

    if "animation" in data and isinstance(data["animation"], dict):
        spec["animation"] = data["animation"]

    return spec


def _build_pie_spec(data: dict, title: str) -> dict:
    """
    Build a PlotSpec for a pie / donut chart.

    Accepted input shapes:
      {"series": [{"name": ..., "data": [{"x": ..., "y": ...}], ...}]} (multi-series/nested)
      {"weights": {label: weight, ...}}                                (single-series)
      {"optimal_weights": {label: weight, ...}}                        (single-series)
      {label: weight, ...}                                             (bare dict of floats)
    """
    series_list = []
    
    # 1. Multi-series structure
    if "series" in data and isinstance(data["series"], list):
        for i, s in enumerate(data["series"]):
            name = s.get("name", f"Series {i+1}")
            
            raw_data = s.get("data", [])
            pts = []
            for j, pt in enumerate(raw_data):
                if not isinstance(pt, dict):
                    continue
                # Normalize point fields: id/x for label, value/y for numeric value
                id_val = pt.get("id") or pt.get("x") or f"slice-{j}"
                val_val = pt.get("value") if pt.get("value") is not None else pt.get("y", 0.0)
                
                pt_entry = {
                    "id": id_val,
                    "value": float(val_val),
                }
                if "color" in pt:
                    pt_entry["color"] = pt["color"]
                if "label" in pt:
                    pt_entry["label"] = pt["label"]
                if "labelMarkType" in pt:
                    pt_entry["labelMarkType"] = pt["labelMarkType"]
                pts.append(pt_entry)
            
            # Sort data values if requested
            sorting = s.get("sorting", "none")
            if sorting == "asc":
                pts.sort(key=lambda x: x["value"])
            elif sorting == "desc":
                pts.sort(key=lambda x: x["value"], reverse=True)

            series_entry = {
                "name": name,
                "data": pts
            }
            
            # Forward all valid pie series properties
            props = [
                "innerRadius", "outerRadius", "paddingAngle", "cornerRadius",
                "startAngle", "endAngle", "cx", "cy", "arcLabel", "arcLabelMinAngle",
                "arcLabelRadius", "highlightScope", "faded", "highlighted",
                "valueFormatter", "sorting"
            ]
            for prop in props:
                if prop in s:
                    series_entry[prop] = s[prop]
            
            series_list.append(series_entry)
            
    # 2. Single-series structure (legacy weights / optimal_weights / bare dict)
    else:
        raw = data.get("weights") or data.get("optimal_weights") or data
        skip_keys = {
            "plot_type", "title", "config", "series_config", "innerRadius", "outerRadius",
            "paddingAngle", "cornerRadius", "startAngle", "endAngle", "cx", "cy",
            "arcLabel", "arcLabelMinAngle", "arcLabelRadius", "highlightScope",
            "faded", "highlighted", "sorting", "skipAnimation", "hideLegend", "colors",
        }
        
        if not isinstance(raw, dict):
            raise ValueError("Pie chart data must contain a weights mapping or series list.")
            
        items = []
        for k, v in raw.items():
            if k in skip_keys:
                continue
            try:
                items.append((str(k), float(v)))
            except (TypeError, ValueError):
                continue
                
        if not items:
            raise ValueError("Pie chart weights are empty.")
            
        # Default behavior: sort descending
        items.sort(key=lambda t: t[1], reverse=True)
        
        pts = [
            {
                "id": k, 
                "value": round(v, 6), 
                "color": PALETTE[i % len(PALETTE)]
            }
            for i, (k, v) in enumerate(items)
        ]
        
        series_entry = {
            "name": "Allocation",
            "data": pts
        }
        
        # Forward any series properties passed directly to data
        props = [
            "innerRadius", "outerRadius", "paddingAngle", "cornerRadius",
            "startAngle", "endAngle", "cx", "cy", "arcLabel", "arcLabelMinAngle",
            "arcLabelRadius", "highlightScope", "faded", "highlighted",
            "valueFormatter", "sorting"
        ]
        for prop in props:
            if prop in data:
                series_entry[prop] = data[prop]
                
        series_list.append(series_entry)
        
    spec = {
        "plot_type": "pie",
        "title": title,
        "series": series_list,
    }
    
    # Top-level general chart props
    if "skipAnimation" in data:
        spec["skipAnimation"] = bool(data["skipAnimation"])
    if "hideLegend" in data:
        spec["hideLegend"] = bool(data["hideLegend"])
    if "colors" in data and isinstance(data["colors"], list):
        spec["colors"] = data["colors"]
        
    if "animation" in data and isinstance(data["animation"], dict):
        spec["animation"] = data["animation"]

    return spec


def _build_scatter_spec(data: dict, title: str) -> dict:
    """
    Build a PlotSpec for a scatter chart.

    Accepted data shapes:
      {"series": [{"name": "Group A", "data": [{"x": 1.0, "y": 2.0, "z": 5.0, "id": 0}], "markerSize": 3, "highlightScope": ...}]}
      {"data": [{"x": 1.0, "y": 2.0}], "name": "Series 1"} (shorthand for single-series)
    """
    series_list = []
    raw_series = data.get("series")
    
    # 1. Shorthand single-series format
    if not raw_series and "data" in data and isinstance(data["data"], list):
        raw_series = [{"name": data.get("name", "Series 1"), "data": data["data"]}]
        
    if raw_series and isinstance(raw_series, list):
        for i, s in enumerate(raw_series):
            name = s.get("name", f"Series {i+1}")
            raw_pts = s.get("data", [])
            pts = []
            for j, pt in enumerate(raw_pts):
                if not isinstance(pt, dict):
                    continue
                x_val = pt.get("x", 0.0)
                y_val = pt.get("y", 0.0)
                pt_entry = {
                    "x": float(x_val),
                    "y": float(y_val),
                    "id": pt.get("id") if pt.get("id") is not None else f"pt-{j}"
                }
                if "z" in pt:
                    pt_entry["z"] = float(pt["z"])
                pts.append(pt_entry)
                
            series_entry = {
                "name": name,
                "label": s.get("label", name),
                "color": s.get("color", PALETTE[i % len(PALETTE)]),
                "data": pts
            }
            if "markerSize" in s:
                series_entry["markerSize"] = s["markerSize"]
            if "highlightScope" in s:
                series_entry["highlightScope"] = s["highlightScope"]
                
            series_list.append(series_entry)
    else:
        raise ValueError("Scatter chart data must contain a series list or data list.")
        
    spec = {
        "plot_type": "scatter",
        "title": title,
        "x_label": data.get("x_label", "X"),
        "y_label": data.get("y_label", "Y"),
        "series": series_list,
    }
    
    # Optional axes config
    if "xAxis" in data and isinstance(data["xAxis"], list):
        spec["xAxis"] = data["xAxis"]
    if "yAxis" in data and isinstance(data["yAxis"], list):
        spec["yAxis"] = data["yAxis"]
    if "zAxis" in data and isinstance(data["zAxis"], list):
        spec["zAxis"] = data["zAxis"]
        
    # Optional general chart settings
    if "grid" in data:
        spec["grid"] = data["grid"]
    else:
        # Default: grids are enabled on both axes for scatter positioning reference
        spec["grid"] = {"horizontal": True, "vertical": True}
        
    if "skipAnimation" in data:
        spec["skipAnimation"] = bool(data["skipAnimation"])
    if "hideLegend" in data:
        spec["hideLegend"] = bool(data["hideLegend"])
    if "hitAreaRadius" in data:
        spec["hitAreaRadius"] = data["hitAreaRadius"]
    if "colors" in data and isinstance(data["colors"], list):
        spec["colors"] = data["colors"]
        
    if "animation" in data and isinstance(data["animation"], dict):
        spec["animation"] = data["animation"]

    return spec


def _build_sparkline_spec(data: dict, title: str) -> dict:
    """
    Build a PlotSpec for a sparkline chart.

    Accepted data shape:
      {"data": [10, 15, 8, 12, 20], "plotType": "line", ...}
    """
    raw_data = data.get("data", [])
    if not isinstance(raw_data, list):
        raise ValueError("Sparkline data must be a list of numbers.")
        
    spec = {
        "plot_type": "sparkline",
        "title": title,
        "data": [float(x) for x in raw_data],
    }
    
    # Forward optional sparkline-specific properties
    if "plotType" in data:
        spec["plotType"] = data["plotType"]
    if "area" in data:
        spec["area"] = bool(data["area"])
    if "curve" in data:
        spec["curve"] = data["curve"]
    if "color" in data:
        spec["color"] = data["color"]
        
    # Defaults showHighlight and showTooltip to True
    spec["showHighlight"] = bool(data.get("showHighlight", True))
    spec["showTooltip"] = bool(data.get("showTooltip", True))
    
    if "xAxis" in data:
        spec["xAxis"] = data["xAxis"]
    if "yAxis" in data:
        spec["yAxis"] = data["yAxis"]
    if "baseline" in data:
        spec["baseline"] = data["baseline"]
    if "height" in data:
        spec["height"] = data["height"]
        
    if "animation" in data and isinstance(data["animation"], dict):
        spec["animation"] = data["animation"]

    return spec


def _build_sankey_spec(data: dict, title: str) -> dict:
    """
    Build a PlotSpec for a Sankey chart.

    Expected data shape:
      {
        "nodes": [{"id": "A", "label": "Source A", "color": "#e57373"}],
        "links": [{"source": "A", "target": "B", "value": 15, "color": ...}],
        "nodeOptions": {"align": "justify", "width": 15, "padding": 10, "showLabels": True, "sort": "auto"},
        "linkOptions": {"color": "source", "opacity": 0.6, "showValues": True, "curveCorrection": 10},
        "valueFormatter": "currency" / "percent" / "raw"
      }
    """
    raw_nodes = data.get("nodes", [])
    raw_links = data.get("links", [])
    
    if not isinstance(raw_links, list) or not raw_links:
        raise ValueError("Sankey chart data must contain a links list.")
        
    nodes = []
    for n in raw_nodes:
        if not isinstance(n, dict) or "id" not in n:
            continue
        node_entry = {"id": str(n["id"])}
        if "label" in n:
            node_entry["label"] = str(n["label"])
        if "color" in n:
            node_entry["color"] = str(n["color"])
        nodes.append(node_entry)
        
    links = []
    for l in raw_links:
        if not isinstance(l, dict) or "source" not in l or "target" not in l or "value" not in l:
            continue
        link_entry = {
            "source": str(l["source"]),
            "target": str(l["target"]),
            "value": float(l["value"])
        }
        if "color" in l:
            link_entry["color"] = str(l["color"])
        links.append(link_entry)
        
    spec = {
        "plot_type": "sankey",
        "title": title,
        "nodes": nodes,
        "links": links
    }
    
    # Forward optional nodeOptions & linkOptions
    if "nodeOptions" in data and isinstance(data["nodeOptions"], dict):
        spec["nodeOptions"] = data["nodeOptions"]
    if "linkOptions" in data and isinstance(data["linkOptions"], dict):
        spec["linkOptions"] = data["linkOptions"]
    if "valueFormatter" in data:
        spec["valueFormatter"] = data["valueFormatter"]
    if "height" in data:
        spec["height"] = data["height"]
        
    if "animation" in data and isinstance(data["animation"], dict):
        spec["animation"] = data["animation"]

    return spec


def _build_candlestick_spec(data: dict, title: str) -> dict:
    """
    Build a PlotSpec for a Candlestick chart.

    Expected data shape:
      {"series": [{"name": "AAPL", "data": [{"date": "2026-05-25", "open": 180.2, "high": 182.5, "low": 179.8, "close": 181.9}]}]}
      or
      {"data": [{"date": "2026-05-25", "open": 180.2, ...}], "name": "AAPL"}
      or
      {"prices": {"AAPL": [{"date": "2026-05-25", "open": 180.2, ...}]}} (from price tool output)
    """
    raw = data.get("prices") or data.get("series") or data.get("data") or data
    if isinstance(raw, dict) and not any(k in raw for k in ("series", "data", "prices")):
        # If it's a dict like {"AAPL": [...]}
        raw_series = []
        for ticker, rows in raw.items():
            if isinstance(rows, list):
                raw_series.append({"name": ticker, "data": rows})
    elif isinstance(raw, list):
        if raw and all(isinstance(x, dict) and "data" in x for x in raw):
            raw_series = raw
        else:
            raw_series = [{"name": data.get("name", "Stock"), "data": raw}]
    elif isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        raw_series = [{"name": raw.get("name", "Stock"), "data": raw["data"]}]
    else:
        raw_series = raw

    if not isinstance(raw_series, list) or not raw_series:
        raise ValueError("Candlestick chart data must contain a list of series or data rows.")

    series_list = []
    for i, s in enumerate(raw_series):
        if not isinstance(s, dict):
            continue
        name = s.get("name", f"Series {i+1}")
        raw_pts = s.get("data", [])
        pts = []
        for pt in raw_pts:
            if not isinstance(pt, dict):
                continue
            date_val = pt.get("date") or pt.get("Date") or pt.get("x")
            if not date_val:
                continue
            
            o = pt.get("open") if pt.get("open") is not None else pt.get("Open")
            h = pt.get("high") if pt.get("high") is not None else pt.get("High")
            l = pt.get("low") if pt.get("low") is not None else pt.get("Low")
            c = pt.get("close") if pt.get("close") is not None else pt.get("Close")
            v = pt.get("volume") if pt.get("volume") is not None else pt.get("Volume")
            
            if o is None or h is None or l is None or c is None:
                continue

            pts.append({
                "date": str(date_val),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": int(v) if v is not None else None,
            })
            
        if pts:
            series_list.append({
                "name": name,
                "label": s.get("label", name),
                "color": s.get("color", PALETTE[i % len(PALETTE)]),
                "data": pts
            })

    if not series_list:
        raise ValueError("No valid OHLC data points found for candlestick chart.")

    spec = {
        "plot_type": "candlestick",
        "title": title,
        "series": series_list,
        "x_label": data.get("x_label", "Date"),
        "y_label": data.get("y_label", "Price"),
    }

    if "height" in data:
        spec["height"] = data["height"]
    if "animation" in data and isinstance(data["animation"], dict):
        spec["animation"] = data["animation"]

    return spec


def _build_network_spec(data: dict, title: str) -> dict:
    """
    Build a PlotSpec for a Network graph, pre-computing positions on the backend
    via networkx spring layout.
    """
    edges, risk_scores = _extract_network_payload(data)

    import networkx as nx
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

    node_positions = {}
    if graph.number_of_nodes() > 0:
        positions = nx.spring_layout(graph, seed=42, k=0.8)
        node_positions = {str(node): [float(p[0]), float(p[1])] for node, p in positions.items()}

    # Standardize nodes list for frontend
    nodes = []
    for node, attrs in graph.nodes(data=True):
        bipartite = attrs.get("bipartite", 0)
        is_stock = (bipartite == 0)
        nodes.append({
            "id": str(node),
            "is_stock": is_stock,
            "risk_score": float(risk_scores.get(node, 0.0)) if is_stock else None,
        })

    spec = {
        "plot_type": "network",
        "title": title,
        "nodes": nodes,
        "edges": [
            {
                "source": str(edge.get("ticker", "")).upper(),
                "target": str(edge.get("holder", "")).strip(),
                "weight": float(edge.get("weight", 0.0)),
            }
            for edge in edges if edge.get("ticker") and edge.get("holder")
        ],
        "node_positions": node_positions,
    }

    if "height" in data:
        spec["height"] = data["height"]

    return spec


def _build_heatmap_spec(data: dict, title: str) -> dict:
    data = _normalize_heatmap_payload(data)
    try:
        df = _extract_matrix(data)
    except Exception:
        # Fallback if matrix missing
        return None

    x_categories = list(df.columns)
    y_categories = list(df.index)
    
    series_data = []
    for i, col in enumerate(x_categories):
        for j, row in enumerate(y_categories):
            val = df.at[row, col]
            if not pd.isna(val):
                series_data.append([i, j, float(val)])
                
    spec = {
        "plot_type": "heatmap",
        "title": title,
        "matrix": data.get("matrix"),
        "metadata": data.get("metadata", {"heatmap_type": "generic"}),
        "xAxis": [{"data": [str(c) for c in x_categories]}],
        "yAxis": [{"data": [str(r) for r in y_categories]}],
        "series": [{"data": series_data}]
    }
    
    if "height" in data:
        spec["height"] = data["height"]
    else:
        spec["height"] = max(400, len(y_categories) * 40 + 100)
        
    return spec


def _build_funnel_spec(data: dict, title: str) -> dict:
    raw = data.get("series") or data.get("data") or data.get("stages") or data
    if isinstance(raw, dict):
        raw_items = [{"id": key, "label": key, "value": value} for key, value in raw.items()]
    elif isinstance(raw, list) and raw and isinstance(raw[0], dict) and "data" in raw[0]:
        raw_items = raw[0].get("data", [])
    else:
        raw_items = raw if isinstance(raw, list) else []

    items = []
    for index, item in enumerate(raw_items):
        if isinstance(item, dict):
            value = item.get("value", item.get("y"))
            label = item.get("label", item.get("id", item.get("x", f"Stage {index + 1}")))
            color = item.get("color", PALETTE[index % len(PALETTE)])
        else:
            value = item
            label = f"Stage {index + 1}"
            color = PALETTE[index % len(PALETTE)]
        try:
            items.append({
                "id": str(label),
                "label": str(label),
                "value": float(value),
                "color": color,
            })
        except (TypeError, ValueError):
            continue

    if not items:
        raise ValueError("Funnel chart data must include stages with numeric values.")

    spec = {
        "plot_type": "funnel",
        "title": title,
        "series": [{
            "label": data.get("label", "Funnel"),
            "data": items,
            "layout": data.get("layout", "vertical"),
            "curve": data.get("curve", "linear"),
            "borderRadius": data.get("borderRadius", 6),
            "variant": data.get("variant", "filled"),
        }],
        "height": data.get("height", 360),
    }
    if "hideLegend" in data:
        spec["hideLegend"] = bool(data["hideLegend"])
    return spec


def _build_radar_spec(data: dict, title: str) -> dict:
    metrics = data.get("metrics") or data.get("radar", {}).get("metrics") or data.get("categories")
    raw_series = data.get("series") or []
    if not metrics and raw_series:
        first = raw_series[0]
        if isinstance(first, dict) and isinstance(first.get("data"), dict):
            metrics = list(first["data"].keys())
    if not metrics:
        raise ValueError("Radar chart data must include metrics/categories.")

    normalized_series = []
    for index, series in enumerate(raw_series):
        if not isinstance(series, dict):
            continue
        raw_values = series.get("data", [])
        if isinstance(raw_values, dict):
            values = [float(raw_values.get(metric, 0.0)) for metric in metrics]
        else:
            values = [float(value) for value in raw_values[:len(metrics)]]
        normalized_series.append({
            "label": series.get("label", series.get("name", f"Series {index + 1}")),
            "data": values,
            "color": series.get("color", PALETTE[index % len(PALETTE)]),
            "fillArea": bool(series.get("fillArea", True)),
        })

    if not normalized_series:
        raise ValueError("Radar chart data must include at least one numeric series.")

    return {
        "plot_type": "radar",
        "title": title,
        "radar": {"metrics": [str(metric) for metric in metrics]},
        "series": normalized_series,
        "height": data.get("height", 360),
        "hideLegend": bool(data.get("hideLegend", False)),
    }


def _build_gauge_spec(data: dict, title: str) -> dict:
    value = data.get("value", data.get("score"))
    if value is None:
        raise ValueError("Gauge chart data must include a numeric value or score.")
    value = float(value)
    value_min = float(data.get("valueMin", data.get("min", 0)))
    value_max = float(data.get("valueMax", data.get("max", 100)))
    return {
        "plot_type": "gauge",
        "title": title,
        "value": value,
        "valueMin": value_min,
        "valueMax": value_max,
        "startAngle": data.get("startAngle", -110),
        "endAngle": data.get("endAngle", 110),
        "height": data.get("height", 260),
        "text": data.get("text"),
    }


def _build_radial_series_spec(data: dict, title: str, plot_type: str) -> dict:
    categories = data.get("categories") or data.get("rotationAxis", [{}])[0].get("data")
    raw_series = data.get("series") or []
    if not categories and raw_series:
        first = raw_series[0]
        if isinstance(first, dict) and isinstance(first.get("data"), dict):
            categories = list(first["data"].keys())
    if not categories:
        raise ValueError(f"{plot_type} chart data must include categories.")

    normalized_series = []
    for index, series in enumerate(raw_series):
        if not isinstance(series, dict):
            continue
        raw_values = series.get("data", [])
        if isinstance(raw_values, dict):
            values = [float(raw_values.get(category, 0.0)) for category in categories]
        else:
            values = [float(value) for value in raw_values[:len(categories)]]
        entry = {
            "label": series.get("label", series.get("name", f"Series {index + 1}")),
            "data": values,
            "color": series.get("color", PALETTE[index % len(PALETTE)]),
        }
        for key in ("stack", "stackOffset", "curve", "closePath", "area", "layout"):
            if key in series:
                entry[key] = series[key]
        normalized_series.append(entry)

    if not normalized_series:
        raise ValueError(f"{plot_type} chart data must include at least one numeric series.")

    return {
        "plot_type": plot_type,
        "title": title,
        "categories": [str(category) for category in categories],
        "series": normalized_series,
        "height": data.get("height", 360),
        "hideLegend": bool(data.get("hideLegend", False)),
        "grid": data.get("grid", {"radius": True, "rotation": True}),
    }

# ---------------------------------------------------------------------------
# Main tool — MUI-native for line/bar/pie/heatmap; PNG fallback for network
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

    For plot_type = line / bar / pie / scatter / sparkline / sankey / candlestick / network / heatmap → stores an interactive PlotSpec in
    GLOBAL_PLOT_DATA so the MUI frontend renders an interactive chart in the
    chat bubble (no PNG saved).

    For plot_type = network → falls back to saving a PNG if not handled yet.

    Args:
        data: dict with chart data. Shape depends on plot_type:
            LINE: {"dates": [...], "series": {"AAPL": [...], "MSFT": [...]}, ...}
                  Optional keys: curve, grid, highlightScope, yAxis, experimentalFeatures,
                  series_config (per-series overrides: area, baseline, stack, showMark, etc.)

            BAR (single-series): {"scores": {"AAPL": 0.85, "MSFT": 0.72}}
            BAR (multi-series):  {"categories": ["Q1","Q2"], "series": [{"name":"Revenue", "data":[10,20], "stack":"financials"}]}
                  Optional keys: layout ("horizontal"/"vertical"), borderRadius (number),
                  grid, categoryGapRatio, barGapRatio, highlightScope, skipAnimation,
                  xAxis (list with colorMap/tickPlacement), yAxis (list with colorMap),
                  series_config (per-series: barLabel, barLabelPlacement, minBarSize, stack, etc.)

            PIE (single-series): {"weights": {"AAPL": 0.15, "MSFT": 0.12}}
                  Optional keys: innerRadius, outerRadius, paddingAngle, cornerRadius,
                  startAngle, endAngle, cx, cy, arcLabel ("value"/"label"/"formattedValue"/"percent"/"label-percent"),
                  arcLabelMinAngle, arcLabelRadius, highlightScope, faded, highlighted, sorting ("asc"/"desc"/"none"),
                  skipAnimation, hideLegend, colors (custom color palette list)
            PIE (multi-series/nested): {"series": [{"name": "layer1", "data": [{"id": "A", "value": 10}], "innerRadius": 0, "outerRadius": 50}, ...]}

            SCATTER (multi-series): {"series": [{"name": "Group A", "data": [{"x": 1.0, "y": 2.0, "z": 5.0, "id": 0}], "markerSize": 3}]}
            SCATTER (single-series): {"data": [{"x": 1.0, "y": 2.0}], "name": "Series 1"}
                  Optional keys: grid ({"horizontal": bool, "vertical": bool}), skipAnimation,
                  hideLegend, hitAreaRadius (number/"item"), colors, xAxis, yAxis, zAxis (list configs)

            SPARKLINE: {"data": [10, 15, 8, 12, 20]}
                  Optional keys: plotType ("line"/"bar"), area, curve ("linear"/"natural"/"step"/"monotoneX"),
                  color, showHighlight, showTooltip, xAxis (config), yAxis (config), baseline, height

            SANKEY: {"nodes": [{"id": "A", "label": "Label", "color": "#hex"}], "links": [{"source": "A", "target": "B", "value": 10, "color": "#hex"}]}
                  Optional keys: nodeOptions (dict with align, width, padding, showLabels, sort),
                  linkOptions (dict with color, opacity, showValues, curveCorrection),
                  valueFormatter ("currency" / "percent" / "raw"), height (number)

            CANDLESTICK: {"series": [{"name": "AAPL", "data": [{"date": "2026-05-25", "open": 180.2, "high": 182.5, "low": 179.8, "close": 181.9, "volume": 1200000}]}]}
                  Optional keys: height (number)

            NETWORK: {"holder_edges": [{"ticker": "AAPL", "holder": "Vanguard Group", "weight": 0.08}], "risk_scores": {"AAPL": 0.65}}
                  Optional keys: height (number)

            All interactive plot types (LINE, BAR, PIE, SCATTER, SPARKLINE, SANKEY, CANDLESTICK, NETWORK) also support:
                  animation: optional dict:
                      "duration": duration string (e.g., "1.5s", "800ms")
                      "delay": delay string (e.g., "0.5s")
                      "easing": easing curve (e.g., "ease-in-out", "cubic-bezier(...)")
                      "animatedLabels": boolean (defaults to true; applies to bar labels)

        plot_type: One of "line", "bar", "pie", "scatter", "sparkline", "sankey", "candlestick", "network", "heatmap".
        title: Chart title string.
    """
    try:
        payload = _coerce_dict(data)
        
        normalized = str(plot_type or "").strip().lower()
        # Check if analysis_cache_key is provided to load from cache
        cache_key = payload.get("analysis_cache_key")
        if cache_key:
            from src.agents.price_series_tool import load_cached_analysis_dataset
            cached_data = load_cached_analysis_dataset(cache_key)
            if cached_data:
                if normalized == "heatmap":
                    returns_dict = cached_data.get("returns", {})
                    dates_dict = cached_data.get("return_dates_by_ticker", {})
                    df_list = []
                    for ticker, returns in returns_dict.items():
                        dates = dates_dict.get(ticker, [])
                        if dates and returns:
                            df_list.append(pd.Series(returns, index=pd.to_datetime(dates), name=ticker))
                    if df_list:
                        df = pd.concat(df_list, axis=1).sort_index().dropna()
                        if not df.empty:
                            payload["correlation_matrix"] = df.corr().to_dict()
                else:
                    metric = payload.get("metric", "prices")
                    if metric == "returns" or "return" in payload.get("y_label", "").lower():
                        # Reconstruct returns series for plotting
                        series_data = {}
                        for ticker in cached_data.get("tickers_included", []):
                            dates = cached_data.get("return_dates_by_ticker", {}).get(ticker, [])
                            returns = cached_data.get("returns", {}).get(ticker, [])
                            series_data[ticker] = [
                                {"date": d, "value": r} for d, r in zip(dates, returns)
                            ]
                        payload["series"] = series_data
                        if "y_label" not in payload:
                            payload["y_label"] = "Log Return"
                    else:
                        # Reconstruct prices series
                        prices_data = cached_data.get("prices", {})
                        payload["price_history"] = prices_data
                        payload["prices"] = prices_data
                        if "y_label" not in payload:
                            payload["y_label"] = "Price"

        normalized = str(plot_type or "").strip().lower()
        plot_title = normalize_advisory_language(str(title or "Financial Plot").strip())

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
        elif normalized == "scatter":
            spec = _build_scatter_spec(payload, plot_title)
        elif normalized == "sparkline":
            spec = _build_sparkline_spec(payload, plot_title)
        elif normalized == "sankey":
            spec = _build_sankey_spec(payload, plot_title)
        elif normalized == "candlestick":
            spec = _build_candlestick_spec(payload, plot_title)
        elif normalized == "network":
            spec = _build_network_spec(payload, plot_title)
        elif normalized == "heatmap":
            spec = _build_heatmap_spec(payload, plot_title)
        elif normalized == "funnel":
            spec = _build_funnel_spec(payload, plot_title)
        elif normalized == "radar":
            spec = _build_radar_spec(payload, plot_title)
        elif normalized == "gauge":
            spec = _build_gauge_spec(payload, plot_title)
        elif normalized == "radial_bar":
            spec = _build_radial_series_spec(payload, plot_title, "radial_bar")
        elif normalized == "radial_line":
            spec = _build_radial_series_spec(payload, plot_title, "radial_line")
        else:
            spec = None  # falls through to PNG path below

        if spec is not None:
            import uuid
            from src.memory.mongodb_memory_layer import MongoMemoryManager
            
            plot_id = str(uuid.uuid4())
            stored = False
            try:
                mongo = MongoMemoryManager()
                stored = bool(mongo.store_plot(plot_id, spec, ttl_days=365))
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Failed to store plot in MongoDB: {e}")

            if not stored:
                return (
                    "Unable to generate plot: visualization storage is unavailable, "
                    "so no interactive chart was attached."
                )

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
