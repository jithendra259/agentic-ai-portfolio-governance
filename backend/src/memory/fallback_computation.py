from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from api.analytics_router import get_portfolio_prices
from src.decision.concentration_metrics import compute_concentration_metrics
from src.memory.session_state import KNOWN_UNIVERSE_TICKERS


DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2024-12-31"

U1_SECTOR_MAP = {ticker: "Technology" for ticker in KNOWN_UNIVERSE_TICKERS.get("U1", [])}
DEFAULT_SECTOR_MAP = {
    **U1_SECTOR_MAP,
    "AMZN": "Consumer Discretionary",
    "JPM": "Financials",
    "XOM": "Energy",
    "UNH": "Health Care",
    "LLY": "Health Care",
    "V": "Financials",
    "MA": "Financials",
    "COST": "Consumer Staples",
    "HD": "Consumer Discretionary",
    "PG": "Consumer Staples",
    "NFLX": "Communication Services",
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "BAC": "Financials",
}


def compute_equal_weights(tickers: list[str], percent: bool = True) -> dict[str, float]:
    clean = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not clean:
        return {}
    value = (100.0 if percent else 1.0) / len(clean)
    return {ticker: value for ticker in clean}


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {str(t).upper(): float(w) for t, w in (weights or {}).items() if str(t).strip()}
    total = sum(clean.values())
    if total <= 0:
        return {}
    if total > 1.5:
        return {ticker: weight / total for ticker, weight in clean.items()}
    return clean


def compute_sector_weights(ticker_weights: dict[str, float], sector_map: dict[str, str] | None = None) -> dict[str, float]:
    sectors = sector_map or DEFAULT_SECTOR_MAP
    normalized = normalize_weights(ticker_weights)
    out: dict[str, float] = {}
    for ticker, weight in normalized.items():
        sector = sectors.get(ticker, "Other")
        out[sector] = out.get(sector, 0.0) + weight
    return out


def compute_hhi_bundle(ticker_weights: dict[str, float], sector_map: dict[str, str] | None = None) -> dict[str, float]:
    return compute_concentration_metrics(normalize_weights(ticker_weights), sector_map or DEFAULT_SECTOR_MAP)


def compute_allocation_change(current_weights: dict[str, float], advisory_weights: dict[str, float]) -> dict[str, float]:
    current = normalize_weights(current_weights)
    advisory = normalize_weights(advisory_weights)
    tickers = sorted(set(current) | set(advisory))
    return {ticker: advisory.get(ticker, 0.0) - current.get(ticker, 0.0) for ticker in tickers}


def compute_returns_and_covariance(
    tickers: list[str],
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    clean = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not clean:
        return {"status": "unavailable", "reason": "tickers missing"}

    df_prices, is_mock = get_portfolio_prices(",".join(clean), start_date or DEFAULT_START_DATE, end_date or DEFAULT_END_DATE)
    df_returns = np.log(df_prices / df_prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if df_returns.empty or len(df_returns) < 2:
        return {"status": "unavailable", "reason": "not enough return observations"}

    covariance = df_returns.cov() * 252.0
    return {
        "status": "success",
        "tickers": list(df_returns.columns),
        "returns_df": df_returns,
        "covariance_matrix": covariance,
        "date_range": {
            "start": start_date or DEFAULT_START_DATE,
            "end": end_date or DEFAULT_END_DATE,
        },
        "data_source": "historical_returns" if not is_mock else "fallback_sample_price_simulation",
        "fallback_used": bool(is_mock),
    }


def compute_risk_contribution(
    tickers: list[str],
    weights: dict[str, float],
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    returns_bundle = compute_returns_and_covariance(tickers, start_date, end_date)
    if returns_bundle.get("status") != "success":
        return returns_bundle

    cov_df: pd.DataFrame = returns_bundle["covariance_matrix"]
    normalized_weights = normalize_weights(weights)
    aligned_tickers = [ticker for ticker in cov_df.columns if ticker in normalized_weights]
    if not aligned_tickers:
        return {"status": "unavailable", "reason": "weights do not overlap covariance tickers"}

    cov = cov_df.loc[aligned_tickers, aligned_tickers].to_numpy(dtype=float)
    w = np.array([normalized_weights[ticker] for ticker in aligned_tickers], dtype=float)
    w = w / w.sum()
    denominator = float(w.T @ cov @ w)
    if denominator <= 0 or not np.isfinite(denominator):
        return {"status": "unavailable", "reason": "portfolio variance is not finite"}

    marginal = cov @ w
    contributions = w * marginal / denominator
    data = [
        {
            "ticker": ticker,
            "risk_contribution_percent": float(contributions[index] * 100.0),
            "allocation_percent": float(w[index] * 100.0),
        }
        for index, ticker in enumerate(aligned_tickers)
    ]
    data.sort(key=lambda row: row["risk_contribution_percent"], reverse=True)
    return {
        "status": "success",
        "plot_id": "plot_55_risk_contribution_by_ticker",
        "chart_type": "bar",
        "bar_mode": "horizontal",
        "data": data,
        "covariance_matrix": cov_df.to_dict(),
        "data_source": returns_bundle["data_source"],
        "fallback_used": returns_bundle["fallback_used"],
        "formula": "risk_contribution_i = w_i * (Sigma w)_i / (w^T Sigma w)",
        "confidence": "Medium" if returns_bundle["fallback_used"] else "High",
        "limitations": [
            "Actual current holdings were not available; proxy weights are used when active_weights.type is equal_weight_proxy."
        ],
    }


def compute_var_cvar_from_returns(portfolio_returns: np.ndarray, alpha: float = 0.95) -> dict[str, float]:
    if portfolio_returns.size == 0:
        return {"var_95": 0.0, "cvar_95": 0.0}
    losses = -portfolio_returns
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size else var
    return {"var_95": var, "cvar_95": cvar}


def compute_drawdown_from_returns(portfolio_returns: np.ndarray) -> dict[str, float]:
    if portfolio_returns.size == 0:
        return {"maximum_drawdown": 0.0}
    equity = np.exp(np.cumsum(portfolio_returns))
    drawdown = 1.0 - equity / np.maximum.accumulate(equity)
    return {"maximum_drawdown": float(np.max(drawdown))}
