from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.analytics.stock_eda.data_loader import extract_ohlcv_from_db, generate_fallback_ohlcv
from src.analytics.stock_eda.metadata import DEFAULT_FULL_EDA_TICKERS
from src.analytics.stock_eda.serialization import normalize_ticker_list, recordify, round_float
from src.analytics.stock_eda.summaries import descriptive_summary, outlier_rows, seasonal_summary, time_series_summary


def build_full_stock_eda_payload(
    tickers: str = "AAPL,JPM,MSFT,BAC,META,WFC,GOOG,HSBC",
    start_date: str = "2019-01-01",
    end_date: str = "2023-12-31",
    get_collection: Callable[[], object | None] | None = None,
    generate_price_series: Callable[[list[str], str, str], dict[str, pd.Series]] | None = None,
) -> dict:
    if get_collection is None or generate_price_series is None:
        raise ValueError("get_collection and generate_price_series are required for full stock EDA.")

    ticker_list = normalize_ticker_list(tickers, DEFAULT_FULL_EDA_TICKERS)
    df, is_mock, metadata = extract_ohlcv_from_db(ticker_list, start_date, end_date, get_collection)
    if df.empty:
        df = generate_fallback_ohlcv(ticker_list, start_date, end_date, generate_price_series)
        is_mock = True
        metadata = {"source": "fallback_sample_ohlcv_simulation", "missing_tickers": []}
    else:
        missing_tickers = metadata.get("missing_tickers", [])
        if missing_tickers:
            fallback_df = generate_fallback_ohlcv(missing_tickers, start_date, end_date, generate_price_series)
            if not fallback_df.empty:
                fallback_df["data_source"] = "fallback_sample_ohlcv_simulation"
                df["data_source"] = metadata.get("source", "historical_prices")
                df = pd.concat([df, fallback_df], ignore_index=True)
                is_mock = True
                metadata = {
                    **metadata,
                    "source": "mixed_historical_and_fallback_sample_ohlcv",
                    "fallback_tickers": missing_tickers,
                    "missing_tickers": [],
                }

    df = _prepare_features(df, metadata)
    numeric_metrics = ["open", "high", "low", "close", "adjusted_close", "volume", "market_return", "volatility"]
    descriptive_by_ticker = descriptive_summary(df, ["ticker", "sector", "company"], numeric_metrics)
    descriptive_by_sector = descriptive_summary(df, ["sector"], numeric_metrics)

    ordered_cols = list(df.columns)
    missing_summary = []
    for ticker, ticker_df in df.groupby("ticker"):
        missing_cells = int(ticker_df[ordered_cols].isna().sum().sum())
        missing_summary.append(
            {
                "ticker": ticker,
                "rows": int(len(ticker_df)),
                "missing_cells": missing_cells,
                "complete_rate": round_float(1.0 - (missing_cells / max(1, len(ticker_df) * len(ordered_cols))), 4),
                "first_date": ticker_df["date"].min().strftime("%Y-%m-%d"),
                "last_date": ticker_df["date"].max().strftime("%Y-%m-%d"),
            }
        )

    return {
        "is_mock": bool(is_mock),
        "tickers": sorted(df["ticker"].unique().tolist()),
        "date_range": {
            "start": df["date"].min().strftime("%Y-%m-%d") if not df.empty else start_date,
            "end": df["date"].max().strftime("%Y-%m-%d") if not df.empty else end_date,
        },
        "metadata": metadata,
        "schema": {
            "market_return": "(Close - Open) / Close * 100",
            "volatility": "(High - Low) / Close * 100",
            "seasonal_grain": "year, quarter, month, day_of_week",
            "outlier_rule": "absolute z-score greater than 2 by ticker and metric",
        },
        "prepared_rows_sample": recordify(df, max_rows=250),
        "missing_values": missing_summary,
        "descriptive_statistics_by_ticker": descriptive_by_ticker,
        "descriptive_statistics_by_sector": descriptive_by_sector,
        "outliers": outlier_rows(df, numeric_metrics),
        "seasonal_summaries": {
            "year": seasonal_summary(df, ["sector", "year"]),
            "quarter": seasonal_summary(df, ["sector", "quarter"]),
            "month": seasonal_summary(df, ["sector", "month_number", "month"]),
            "day_of_week": seasonal_summary(df, ["sector", "day_of_week"]),
            "ticker_year": seasonal_summary(df, ["ticker", "sector", "year"]),
            "ticker_month": seasonal_summary(df, ["ticker", "sector", "month_number", "month"]),
        },
        "time_series": {
            "sector_daily": time_series_summary(df, ["date", "sector"]),
            "ticker_daily": time_series_summary(df, ["date", "ticker", "sector"], max_rows=900),
        },
    }


def _prepare_features(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    working = df.copy()
    if "data_source" not in working.columns:
        working["data_source"] = metadata.get("source", "historical_prices")
    for col_name in ["open", "high", "low", "close", "adjusted_close", "volume"]:
        working[col_name] = pd.to_numeric(working[col_name], errors="coerce")
    working = working.dropna(subset=["date", "open", "high", "low", "close"])
    working["market_return"] = np.where(working["close"] != 0, (working["close"] - working["open"]) / working["close"] * 100, np.nan)
    working["volatility"] = np.where(working["close"] != 0, ((working["high"] - working["low"]) / working["close"]) * 100, np.nan)
    working["year"] = working["date"].dt.year.astype(str)
    working["month_number"] = working["date"].dt.month
    working["month"] = working["date"].dt.month_name()
    working["quarter"] = "Q" + working["date"].dt.quarter.astype(str)
    working["day"] = working["date"].dt.day.astype(int)
    working["day_of_week"] = working["date"].dt.day_name()

    ordered_cols = [
        "date", "open", "high", "low", "close", "volume", "adjusted_close",
        "ticker", "sector", "company", "data_source", "market_return", "volatility",
        "year", "month_number", "month", "quarter", "day", "day_of_week",
    ]
    return working[ordered_cols].sort_values(["ticker", "date"])
