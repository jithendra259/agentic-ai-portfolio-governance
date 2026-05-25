from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path
import uuid

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from src.agents.price_series_tool import load_cached_analysis_dataset

import sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from config import CONFIG

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
logger = logging.getLogger(__name__)

_OLLAMA_MODEL = (os.getenv("PORTFOLIO_OLLAMA_MODEL") or CONFIG.LLM_MODEL).strip()

_CODE_GEN_SYSTEM = """\
You are a Python data visualisation expert. Write complete, runnable matplotlib code.

STRICT RULES:
1. Allowed imports ONLY: matplotlib, matplotlib.pyplot as plt, seaborn as sns, pandas as pd, numpy as np, json
2. The variable `data` (a Python dict) is already defined before your code runs. Use it directly.
3. The variable `output_path` (a string) is already defined. Save with:
   plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
4. Apply dark theme at the top: plt.style.use("dark_background")
5. Set rcParams for dark axes:
   plt.rcParams.update({"figure.facecolor": "#0b1020", "axes.facecolor": "#111827",
                        "axes.labelcolor": "#e5e7eb", "xtick.color": "#d1d5db",
                        "ytick.color": "#d1d5db", "text.color": "#f3f4f6"})
6. Call plt.close() as the very last line.
7. `data['prices']` is dict[ticker, list[dict]]: {"AAPL": [{"date": "...", "close": 150}, ...]}.
   `data['returns']` is dict[ticker, list[float]].
   When creating DataFrames, extract the numerical values correctly:
   df = pd.DataFrame({tkr: [obs['close'] for obs in series] for tkr, series in data['prices'].items()}, index=pd.to_datetime(data['price_dates']))
8. If the x-axis is a time-series, format it to show monthly or quarterly ticks (e.g., 'Jan 2022') to avoid overlapping labels.
9. Output ONLY raw Python code. No markdown fences. No explanation. No comments.
"""


import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

_CODE_CACHE = {}

def _ask_llm_for_code(description: str, data_summary: str, error_context: str = "") -> str:
    # Check cache first for successful prompt patterns
    cache_key = hashlib.md5(f"{description}|{data_summary}".encode()).hexdigest()
    if not error_context and cache_key in _CODE_CACHE:
        logger.info("Custom Plot Tool: Found cached Python generation for this description.")
        return _CODE_CACHE[cache_key]

    llm = ChatOllama(model=_OLLAMA_MODEL, temperature=0.1, num_ctx=2048)
    user_content = f"Plot request: {description}\n\nData structure:\n{data_summary}"
    if error_context:
        user_content += f"\n\nYour previous attempt failed with this error — fix it:\n{error_context}"

    response = llm.invoke(
        [
            {"role": "system", "content": _CODE_GEN_SYSTEM},
            {"role": "user", "content": user_content},
        ]
    )
    code = response.content.strip()

    if code.startswith("```"):
        lines = code.splitlines()
        code = "\n".join(line for line in lines if not line.startswith("```"))

    clean_code = code.strip()
    
    # Store successful generation pattern if there was no error
    if not error_context:
        # We only cache it provisionally here — we could invalidate it if it fails in execution
        _CODE_CACHE[cache_key] = clean_code

    return clean_code


def _execute_plot_code(code: str, data: dict, output_path: str) -> str | None:
    """
    Run generated code in an isolated subprocess.
    Returns an error string on failure, or None on success.
    """
    preamble = textwrap.dedent(f"""\
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        import json

        data = {json.dumps(data, default=str)}
        output_path = {json.dumps(output_path)}

    """)

    full_script = preamble + code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(full_script)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return (result.stderr or result.stdout or "Unknown error")[-800:]
        return None
    except subprocess.TimeoutExpired:
        return "Plot generation timed out after 30 seconds."
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _summarise_data(data: dict) -> str:
    """
    Build a compact description of the data dict for the code-gen prompt.
    Avoids blowing up the context window with large datasets.
    """
    lines: list[str] = []
    for key, value in list(data.items())[:15]:
        if isinstance(value, list):
            sample = str(value[:3])[: 120]
            lines.append(f"  {key}: list[{len(value)}], first items: {sample}")
        elif isinstance(value, dict):
            lines.append(f"  {key}: dict, keys={list(value.keys())[:10]}")
        elif isinstance(value, (int, float)):
            lines.append(f"  {key}: number = {value}")
        else:
            lines.append(f"  {key}: {str(value)[:120]}")
    return "\n".join(lines) or "  (empty dict)"


def _resolve_plot_data(data: dict) -> dict:
    if not isinstance(data, dict):
        return data

    cache_key = data.get("analysis_cache_key")
    if not cache_key:
        return data

    cached_dataset = load_cached_analysis_dataset(str(cache_key))
    if cached_dataset is None:
        raise ValueError(
            "The cached analysis dataset is no longer available. Please fetch the price series again."
        )
    return cached_dataset


def _wants_correlation_heatmap(description: str) -> bool:
    normalized = str(description or "").lower()
    return "heatmap" in normalized and (
        "correlation" in normalized or "corr" in normalized
    )


def _wants_covariance_heatmap(description: str) -> bool:
    normalized = str(description or "").lower()
    return "heatmap" in normalized and (
        "covariance" in normalized or "cov" in normalized
    )


def _wants_close_return_plot(description: str) -> bool:
    normalized = str(description or "").lower()
    wants_return = "return" in normalized or "returns" in normalized
    wants_time_plot = any(word in normalized for word in ("plot", "chart", "graph", "line"))
    return wants_return and wants_time_plot and (
        "close" in normalized
        or "closing" in normalized
        or "price" in normalized
        or "daily" in normalized
    )


def _prices_to_aligned_frame(data: dict) -> pd.DataFrame:
    prices = data.get("prices")
    if not isinstance(prices, dict) or not prices:
        raise ValueError("Correlation heatmap data must include cached price series.")

    series_map = {}
    coverage_starts = []
    coverage_ends = []
    for ticker, rows in prices.items():
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

        series = frame.set_index(date_col)[value_col]
        series_map[str(ticker).upper()] = series
        coverage_starts.append(series.index.min())
        coverage_ends.append(series.index.max())

    if len(series_map) < 2:
        raise ValueError("At least two valid price series are required for a heatmap.")

    common_start = max(coverage_starts)
    common_end = min(coverage_ends)
    if common_start > common_end:
        raise ValueError("The requested tickers do not have an overlapping date range.")

    aligned = pd.concat(series_map.values(), axis=1, keys=series_map.keys()).sort_index()
    aligned = aligned[(aligned.index >= common_start) & (aligned.index <= common_end)]
    aligned = aligned.ffill().dropna(how="any")
    if len(aligned) < 3 or aligned.shape[1] < 2:
        raise ValueError("Not enough overlapping price observations remained after alignment.")
    return aligned


def _render_stat_heatmap(
    matrix: pd.DataFrame,
    title: str,
    *,
    center: float | None,
    cmap: str,
) -> None:
    annot = matrix.shape[0] <= 12 and matrix.shape[1] <= 12
    sns.heatmap(
        matrix,
        cmap=cmap,
        center=center,
        annot=annot,
        fmt=".2f" if annot else "",
        linewidths=0.35,
        linecolor="#6b7280",
        square=True,
        cbar_kws={"shrink": 0.78},
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=14)
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)


def _try_render_deterministic_heatmap(data: dict, description: str, output_path: str) -> str | None:
    wants_corr = _wants_correlation_heatmap(description)
    wants_cov = _wants_covariance_heatmap(description)
    if not wants_corr and not wants_cov:
        return None

    prices = _prices_to_aligned_frame(data)
    log_returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if log_returns.empty or len(log_returns) < 2:
        raise ValueError("Not enough clean log returns remained after alignment.")

    start_date = log_returns.index.min().strftime("%Y-%m-%d")
    end_date = log_returns.index.max().strftime("%Y-%m-%d")
    tickers = ", ".join(log_returns.columns)

    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#0b1020",
            "axes.facecolor": "#111827",
            "axes.edgecolor": "#9ca3af",
            "axes.labelcolor": "#e5e7eb",
            "xtick.color": "#d1d5db",
            "ytick.color": "#d1d5db",
            "text.color": "#f3f4f6",
        }
    )

    if wants_corr and wants_cov:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        plt.sca(axes[0])
        _render_stat_heatmap(log_returns.corr(), "Pearson Correlation Matrix (Log Returns)", center=0.0, cmap="coolwarm")
        plt.sca(axes[1])
        _render_stat_heatmap(log_returns.cov(), "Covariance Matrix (Log Returns)", center=None, cmap="mako")
        fig.suptitle(f"{tickers} | {start_date} to {end_date}", fontsize=10, y=0.99, color="#d1d5db")
    elif wants_cov:
        plt.figure(figsize=(10, 8))
        _render_stat_heatmap(log_returns.cov(), "Covariance Matrix (Log Returns)", center=None, cmap="mako")
        plt.xlabel(f"{start_date} to {end_date}")
    else:
        plt.figure(figsize=(10, 8))
        _render_stat_heatmap(log_returns.corr(), "Pearson Correlation Matrix (Log Returns)", center=0.0, cmap="coolwarm")
        plt.xlabel(f"{start_date} to {end_date}")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close("all")
    return (
        f"Plot generated successfully: ![Correlation heatmap](/outputs/{Path(output_path).name})\n\n"
        f"Computed from {len(log_returns)} aligned daily log-return observations across {log_returns.shape[1]} tickers."
    )


def _returns_to_frame(data: dict) -> pd.DataFrame:
    returns = data.get("returns")
    if isinstance(returns, dict) and returns:
        series_map = {}
        dates_by_ticker = data.get("return_dates_by_ticker") if isinstance(data.get("return_dates_by_ticker"), dict) else {}
        aligned_dates = data.get("return_dates") if isinstance(data.get("return_dates"), list) else []

        for ticker, values in returns.items():
            if not isinstance(values, list) or not values:
                continue

            dates = dates_by_ticker.get(ticker) or aligned_dates
            if len(dates) != len(values):
                continue

            series = pd.Series(
                pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float),
                index=pd.to_datetime(dates, errors="coerce"),
                name=str(ticker).upper(),
            ).dropna()
            if not series.empty:
                series_map[str(ticker).upper()] = series

        if series_map:
            frame = pd.concat(series_map.values(), axis=1, keys=series_map.keys()).sort_index()
            return frame.dropna(how="all")

    prices = _prices_to_aligned_frame(data)
    return np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")


def _try_render_deterministic_return_plot(data: dict, description: str, output_path: str) -> str | None:
    if not _wants_close_return_plot(description):
        return None

    returns = _returns_to_frame(data)
    if returns.empty:
        raise ValueError("Not enough clean close-return observations were available to plot.")

    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#0b1020",
            "axes.facecolor": "#111827",
            "axes.edgecolor": "#9ca3af",
            "axes.labelcolor": "#e5e7eb",
            "xtick.color": "#d1d5db",
            "ytick.color": "#d1d5db",
            "text.color": "#f3f4f6",
        }
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    for ticker in returns.columns:
        ax.plot(returns.index, returns[ticker], linewidth=1.4, label=ticker)

    start_date = returns.index.min().strftime("%Y-%m-%d")
    end_date = returns.index.max().strftime("%Y-%m-%d")
    ticker_text = ", ".join(map(str, returns.columns))
    ax.axhline(0, color="#9ca3af", linewidth=0.9, alpha=0.7)
    ax.set_title(f"Daily Close Returns: {ticker_text}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(f"{start_date} to {end_date}")
    ax.set_ylabel("Daily log return")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.18, linewidth=0.7)
    if len(returns.columns) > 1:
        ax.legend(loc="best", frameon=False)
    fig.autofmt_xdate(rotation=25)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close("all")
    return (
        f"Plot generated successfully: ![Daily close returns](/outputs/{Path(output_path).name})\n\n"
        f"Computed from {len(returns)} daily close-return observations for {ticker_text}."
    )


@tool
def generate_custom_plot(data: dict, description: str) -> str:
    """
    Generate any matplotlib/seaborn plot the user requests using AI-generated code.
    """
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        resolved_data = _resolve_plot_data(data)
        
        safe_description = description.replace("\n", " ").replace("\r", "")
        short_title = safe_description[:80].strip().rstrip(".")

        # Fast path: Full image cache
        data_cache_key = data.get("analysis_cache_key")
        if data_cache_key:
            plot_sig = hashlib.md5(f"{data_cache_key}|{short_title}".encode()).hexdigest()
            fast_path_filename = f"custom_cached_{plot_sig}.png"
            if (OUTPUT_DIR / fast_path_filename).exists():
                logger.info("Custom plot image cache HIT — bypassing LLM and execution entirely.")
                return f"Plot generated successfully: ![{short_title}](/outputs/{fast_path_filename})"
        else:
            unique_id = str(uuid.uuid4())[:8]
            fast_path_filename = f"custom_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{unique_id}.png"

        output_path = str(OUTPUT_DIR / fast_path_filename)
        data_summary = _summarise_data(resolved_data)
        logger.info("Generating custom plot: %s", description[:80])

        deterministic_result = _try_render_deterministic_heatmap(
            resolved_data,
            description,
            output_path,
        )
        if deterministic_result is not None:
            return deterministic_result

        deterministic_result = _try_render_deterministic_return_plot(
            resolved_data,
            description,
            output_path,
        )
        if deterministic_result is not None:
            return deterministic_result

        # First attempt
        code = _ask_llm_for_code(description, data_summary)
        error = _execute_plot_code(code, resolved_data, output_path)

        # Invalidate code cache if the first generation resulted in an error
        if error:
            cache_key = hashlib.md5(f"{description}|{data_summary}".encode()).hexdigest()
            _CODE_CACHE.pop(cache_key, None)
            logger.warning("Custom plot first attempt failed — retrying. Error: %s", error[:200])
            code = _ask_llm_for_code(description, data_summary, error_context=error)
            error = _execute_plot_code(code, resolved_data, output_path)

        if error:
            return (
                f"Unable to generate the requested plot after two attempts.\n"
                f"Last error: {error}\n\n"
                f"Try rephrasing your request or provide the data in a simpler format."
            )

        if not Path(output_path).exists():
            return (
                "Plot code ran without errors but no image was saved. "
                "The generated code may have missed the plt.savefig(output_path) call."
            )

        return f"Plot generated successfully: ![{short_title}](/outputs/{Path(output_path).name})"

    except Exception as exc:
        logger.exception("generate_custom_plot failed unexpectedly")
        return f"Unable to generate plot due to an internal error: {type(exc).__name__}: {exc}"
