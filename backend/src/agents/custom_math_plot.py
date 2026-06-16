import ast
import math
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.agents.plot_store import register_plot


ALLOWED_FUNCTIONS = {
    "abs": abs,
    "acos": math.acos,
    "asin": math.asin,
    "atan": math.atan,
    "ceil": math.ceil,
    "cos": math.cos,
    "cosh": math.cosh,
    "exp": math.exp,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "max": max,
    "min": min,
    "pow": pow,
    "round": round,
    "sin": math.sin,
    "sinh": math.sinh,
    "sqrt": math.sqrt,
    "tan": math.tan,
    "tanh": math.tanh,
}

ALLOWED_CONSTANTS = {
    "e": math.e,
    "pi": math.pi,
    "tau": math.tau,
}

ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)


def _safe_eval_formula(formula: str, x_value: float) -> float:
    tree = ast.parse(str(formula), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"Unsupported formula syntax: {node.__class__.__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
                raise ValueError("Only approved math functions are allowed in formulas.")
        if isinstance(node, ast.Name) and node.id not in {"x", *ALLOWED_FUNCTIONS.keys(), *ALLOWED_CONSTANTS.keys()}:
            raise ValueError(f"Unknown formula variable: {node.id}")

    namespace = {"x": float(x_value), **ALLOWED_FUNCTIONS, **ALLOWED_CONSTANTS}
    value = eval(compile(tree, "<math-plot-formula>", "eval"), {"__builtins__": {}}, namespace)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("Formula produced a non-finite numeric value.")
    return float(value)


def _linspace(start: float, end: float, points: int) -> list[float]:
    if points < 2:
        raise ValueError("points must be at least 2.")
    if points > 1000:
        raise ValueError("points cannot exceed 1000 for chat rendering.")
    step = (float(end) - float(start)) / (points - 1)
    return [float(start) + step * index for index in range(points)]


@tool
def generate_custom_math_plot(
    formulas: list[dict[str, Any]],
    title: str,
    x_start: float = 0,
    x_end: float = 10,
    points: int = 100,
    x_label: str = "x",
    y_label: str = "Value",
    plot_type: str = "line",
    config: RunnableConfig = None,
) -> str:
    """
    Generate an interactive custom mathematical plot from one or more formulas.

    Use this when the user asks for a custom plot based on a formula or derived
    mathematical relationship, for example y=x**2, y=sin(x), log growth,
    payoff curves, custom risk curves, or any non-database synthetic function.

    Args:
        formulas: [{"name": "Quadratic", "formula": "x**2"}, ...].
        title: Chart title.
        x_start: First x value.
        x_end: Last x value.
        points: Number of sampled x values, max 1000.
        x_label: X-axis label.
        y_label: Y-axis label.
        plot_type: "line" or "scatter".
    """
    normalized_plot_type = str(plot_type or "line").strip().lower()
    if normalized_plot_type not in {"line", "scatter"}:
        return "Unable to generate custom math plot: plot_type must be 'line' or 'scatter'."
    if not isinstance(formulas, list) or not formulas:
        return "Unable to generate custom math plot: provide at least one formula."

    try:
        xs = _linspace(float(x_start), float(x_end), int(points))
        series = []
        for index, item in enumerate(formulas):
            if not isinstance(item, dict):
                raise ValueError("Each formula entry must be an object with name and formula.")
            name = str(item.get("name") or f"Series {index + 1}")
            formula = str(item.get("formula") or "").strip()
            if not formula:
                raise ValueError(f"Missing formula for {name}.")
            data = [{"x": round(x, 8), "y": round(_safe_eval_formula(formula, x), 8)} for x in xs]
            series.append(
                {
                    "name": name,
                    "label": name,
                    "data": data,
                    "showMark": bool(item.get("showMark", normalized_plot_type == "scatter")),
                }
            )

        spec = {
            "plot_type": normalized_plot_type,
            "title": str(title or "Custom Math Plot"),
            "x_label": x_label,
            "y_label": y_label,
            "x_type": "linear",
            "grid": {"horizontal": True, "vertical": True},
            "series": series,
        }
        if normalized_plot_type == "line":
            spec["curve"] = "linear"
            spec["highlightScope"] = {"highlight": "series", "fade": "global"}
            spec["experimentalFeatures"] = {"enablePositionBasedPointerInteraction": True}

        plot_id = str(uuid.uuid4())
        session_id = (
            config.get("configurable", {}).get("thread_id", "default")
            if config
            else "default"
        )
        register_plot(plot_id, spec, session_id)
        return f"Custom math plot ready: {spec['title']}"
    except Exception as exc:
        return f"Unable to generate custom math plot: {exc}"
