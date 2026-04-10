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
