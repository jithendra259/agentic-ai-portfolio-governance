from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def normalize_ticker_list(tickers: str | list[str], default_tickers: list[str]) -> list[str]:
    values = tickers.split(",") if isinstance(tickers, str) else tickers
    cleaned = []
    for ticker in values:
        symbol = str(ticker or "").strip().upper()
        if symbol and symbol not in cleaned:
            cleaned.append(symbol)
    return cleaned or default_tickers


def first_present(row: pd.Series | dict, names: list[str], default: Any = np.nan) -> Any:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        numeric = float(value)
        if not np.isfinite(numeric):
            return default
        return numeric
    except (TypeError, ValueError):
        return default


def round_float(value: Any, digits: int = 6) -> float | None:
    numeric = safe_float(value)
    return round(numeric, digits) if numeric is not None else None


def recordify(df: pd.DataFrame, max_rows: int = 700) -> list[dict[str, Any]]:
    if df.empty:
        return []
    working = df.copy()
    if len(working) > max_rows:
        step = max(1, len(working) // max_rows)
        working = working.iloc[::step].copy()
        if working.index[-1] != df.index[-1]:
            working = pd.concat([working, df.tail(1)]).drop_duplicates()

    records = []
    for row in working.to_dict(orient="records"):
        clean_row = {}
        for key, value in row.items():
            if isinstance(value, (pd.Timestamp, datetime)):
                clean_row[key] = value.strftime("%Y-%m-%d")
            elif pd.isna(value):
                clean_row[key] = None
            elif isinstance(value, np.integer):
                clean_row[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                clean_row[key] = round_float(value)
            else:
                clean_row[key] = value
        records.append(clean_row)
    return records
