"""Sandboxed Dynamic Python Code Execution Engine for Portfolio Calculations.

Allows specialized agents and users to execute dynamic mathematical and statistical code
on aligned portfolio price and return data with strict security guardrails.
"""

from __future__ import annotations

import ast
import contextlib
import io
import logging
import math
import sys
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from src.agents.live_data_tools import _normalize_tickers
from src.agents.math_eda_tools import _fetch_price_df

logger = logging.getLogger(__name__)

# Disallowed AST nodes and attributes for security
BANNED_MODULES = {
    "os", "sys", "subprocess", "socket", "urllib", "requests", "shutil",
    "pathlib", "importlib", "builtins", "__builtin__", "pickle", "ctypes"
}

BANNED_ATTRIBUTES = {
    "__subclasses__", "__bases__", "__class__", "__globals__", "__code__",
    "__closure__", "__builtins__", "__import__", "system", "spawn", "popen"
}


class CodeSecurityValidator(ast.NodeVisitor):
    """Inspects AST to ensure the dynamic code only uses safe mathematical operations."""

    def __init__(self):
        self.errors: List[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if base_mod in BANNED_MODULES:
                self.errors.append(f"Security Policy: Import of module '{alias.name}' is prohibited.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_mod = node.module.split(".")[0]
            if base_mod in BANNED_MODULES:
                self.errors.append(f"Security Policy: Import from module '{node.module}' is prohibited.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in BANNED_ATTRIBUTES:
            self.errors.append(f"Security Policy: Access to private attribute '{node.attr}' is prohibited.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "compile", "open", "input", "__import__"}:
                self.errors.append(f"Security Policy: Direct invocation of '{node.func.id}()' is prohibited.")
        self.generic_visit(node)


def validate_python_code_security(code: str) -> Optional[str]:
    """Parse and validate python code for safety. Returns error string if invalid."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax Error in generated code: {e}"

    validator = CodeSecurityValidator()
    validator.visit(tree)
    if validator.errors:
        return "\n".join(validator.errors)
    return None


@tool("execute_dynamic_portfolio_code")
def execute_dynamic_portfolio_code(
    code: str,
    tickers: list[str],
    start_date: str = "2020-01-01",
    end_date: str = "2024-01-01",
) -> str:
    """Execute dynamic mathematical Python code on portfolio price and return data.
    
    The code runs in a sandboxed environment with pre-loaded variables:
      - `prices`: pandas DataFrame of historical aligned closing prices (columns = tickers).
      - `returns`: pandas DataFrame of log returns.
      - `tickers`: list of ticker symbols.
      - `np`: numpy library.
      - `pd`: pandas library.
      - `scipy`: scipy library.
      - `math`: math module.
    
    To output results, either use `print(...)` or assign your final answer/table/dictionary to variable `result`.

    Args:
        code: Python code string to execute.
        tickers: List of ticker symbols (e.g. ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']).
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.
    """
    clean_tickers = _normalize_tickers(tickers)
    if not clean_tickers:
        return "ERROR: Please specify at least one valid ticker symbol."

    # 1. Security Check
    security_err = validate_python_code_security(code)
    if security_err:
        return f"CODE_SECURITY_VIOLATION:\n{security_err}"

    # 2. Fetch Market Data
    prices = _fetch_price_df(clean_tickers, start_date=start_date, end_date=end_date)
    if prices.empty:
        return f"ERROR: Could not retrieve price history for {clean_tickers} between {start_date} and {end_date}."

    returns = np.log(prices / prices.shift(1)).dropna()

    # 3. Prepare Safe Execution Scope
    safe_globals: Dict[str, Any] = {
        "__builtins__": {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        },
        "np": np,
        "numpy": np,
        "pd": pd,
        "pandas": pd,
        "math": math,
        "prices": prices.copy(),
        "returns": returns.copy(),
        "tickers": list(prices.columns),
    }

    # Include scipy if available
    try:
        import scipy
        import scipy.stats as stats
        import scipy.optimize as opt
        safe_globals["scipy"] = scipy
        safe_globals["stats"] = stats
        safe_globals["opt"] = opt
    except ImportError:
        pass

    # 4. Execute with stdout capture
    stdout_buffer = io.StringIO()
    local_scope: Dict[str, Any] = {}
    exec_error: Optional[Exception] = None

    def _worker():
        nonlocal exec_error
        try:
            with contextlib.redirect_stdout(stdout_buffer):
                exec(code, safe_globals, local_scope)
        except Exception as e:
            exec_error = e

    t = threading.Thread(target=_worker)
    t.start()
    t.join(timeout=10.0)

    if t.is_alive():
        return "EXECUTION_TIMEOUT: Code execution exceeded maximum allowable runtime of 10.0 seconds."

    if exec_error is not None:
        return f"EXECUTION_ERROR: {type(exec_error).__name__}: {exec_error}"

    output_str = stdout_buffer.getvalue().strip()
    result_val = local_scope.get("result")

    parts = []
    if output_str:
        parts.append(f"**Execution Output:**\n```\n{output_str}\n```")
    if result_val is not None:
        if isinstance(result_val, (pd.DataFrame, pd.Series)):
            parts.append(f"**Result Data:**\n\n{result_val.to_markdown()}")
        else:
            parts.append(f"**Result Value:** `{result_val}`")

    if not parts:
        return "Code executed successfully with no output returned. (Tip: Use `print(...)` or set `result = ...` to return data.)"

    return "\n\n".join(parts)
