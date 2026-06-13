from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.analytics.stock_eda.serialization import recordify, round_float


def descriptive_summary(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    rows = []
    for group_key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        base = {col: value for col, value in zip(group_cols, group_key)}
        for metric in metrics:
            series = pd.to_numeric(group[metric], errors="coerce").dropna()
            if series.empty:
                continue
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "count": int(series.count()),
                    "mean": round_float(series.mean()),
                    "sd": round_float(series.std(ddof=1)),
                    "min": round_float(series.min()),
                    "q1": round_float(series.quantile(0.25)),
                    "median": round_float(series.median()),
                    "q3": round_float(series.quantile(0.75)),
                    "max": round_float(series.max()),
                    "skewness": round_float(series.skew()),
                    "kurtosis": round_float(series.kurt()),
                }
            )
    return rows


def outlier_rows(df: pd.DataFrame, metrics: list[str]) -> list[dict[str, Any]]:
    rows = []
    for ticker, group in df.groupby("ticker"):
        for metric in metrics:
            series = pd.to_numeric(group[metric], errors="coerce")
            std = series.std(ddof=1)
            if not std or not np.isfinite(std):
                continue
            z_scores = (series - series.mean()) / std
            flagged = group.loc[z_scores.abs() > 2.0, ["date", "ticker", "sector", "company"]].copy()
            for index, base_row in flagged.iterrows():
                rows.append(
                    {
                        "date": base_row["date"].strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "sector": base_row["sector"],
                        "company": base_row["company"],
                        "metric": metric,
                        "value": round_float(group.loc[index, metric]),
                        "z_score": round_float(z_scores.loc[index]),
                    }
                )
    return rows[:500]


def seasonal_summary(df: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            observations=("ticker", "count"),
            average_close=("close", "mean"),
            average_volume=("volume", "mean"),
            average_volatility=("volatility", "mean"),
            total_market_return=("market_return", "sum"),
            average_market_return=("market_return", "mean"),
        )
        .reset_index()
    )
    sort_cols = [col for col in ["sector", "ticker", "year", "month_number", "quarter", "day_of_week"] if col in grouped.columns]
    if sort_cols:
        grouped = grouped.sort_values(sort_cols)
    return recordify(grouped, max_rows=700)


def time_series_summary(df: pd.DataFrame, group_cols: list[str], max_rows: int = 700) -> list[dict[str, Any]]:
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            average_close=("close", "mean"),
            average_volume=("volume", "mean"),
            average_volatility=("volatility", "mean"),
            average_market_return=("market_return", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )
    return recordify(grouped, max_rows=max_rows)
