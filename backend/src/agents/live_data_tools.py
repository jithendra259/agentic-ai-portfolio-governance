import json
import logging
import os
import re
import time
import warnings
from functools import lru_cache
from importlib import import_module
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from config import CONFIG
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pymongo import MongoClient
from pymongo.errors import AutoReconnect, NetworkTimeout, PyMongoError
from src.memory.mongodb_memory_layer import MongoMemoryManager

load_dotenv(root_dir / ".env", encoding="utf-8-sig")
load_dotenv()


DB_NAME = "Stock_data"
COLLECTION_NAME = "ticker"
logger = logging.getLogger(__name__)
LOOKUP_CACHE_TTL_SECONDS = 300
_LOOKUP_CACHE = {}
_MEMORY_MANAGER = None


class _LazyModule:
    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = import_module(self.module_name)
        return self._module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


cp = _LazyModule("cvxpy")


def _get_memory_manager() -> MongoMemoryManager:
    global _MEMORY_MANAGER
    if _MEMORY_MANAGER is None:
        _MEMORY_MANAGER = MongoMemoryManager()
    return _MEMORY_MANAGER


class _LazyMemoryManager:
    def __getattr__(self, name: str):
        return getattr(_get_memory_manager(), name)


memory_manager = _LazyMemoryManager()


def _generate_financial_plot():
    from src.agents.generate_dynamic_plot import generate_financial_plot
    return generate_financial_plot


def _get_mongo_uri() -> str:
    mongo_uri = (os.getenv("MONGO_URI") or "").strip()
    if not mongo_uri:
        load_dotenv(root_dir / ".env", override=False, encoding="utf-8-sig")
        mongo_uri = (os.getenv("MONGO_URI") or "").strip()
    return mongo_uri


@lru_cache(maxsize=1)
def _get_client():
    mongo_uri = _get_mongo_uri()
    if not mongo_uri:
        raise ValueError("MONGO_URI is not set in the environment.")

    return MongoClient(
        mongo_uri,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=15000,
        socketTimeoutMS=60000,
        maxPoolSize=20,
        retryReads=True,
        retryWrites=True,
        appname="agentic-ai-portfolio-governance-tools",
    )


@lru_cache(maxsize=1)
def _ensure_indexes():
    collection = _get_client()[DB_NAME][COLLECTION_NAME]
    collection.create_index("ticker", background=True)
    collection.create_index("universes", background=True)
    collection.create_index("info.sector", background=True)
    collection.create_index("sector", background=True)
    return True


def _get_collection():
    _ensure_indexes()
    return _get_client()[DB_NAME][COLLECTION_NAME]


def _refresh_collection():
    _ensure_indexes.cache_clear()
    _get_client.cache_clear()
    return _get_collection()


def _find_documents_with_retry(
    query: dict,
    projection: Optional[dict] = None,
    sort: Optional[tuple[str, int]] = None,
    limit: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_time_ms: Optional[int] = 15000,
    attempts: int = 2,
    retry_delay_seconds: float = 1.5,
):
    last_error = None
    collection = _get_collection()

    for attempt in range(1, attempts + 1):
        try:
            cursor = collection.find(query, projection or {})
            if sort is not None:
                cursor = cursor.sort(sort[0], sort[1])
            if limit is not None:
                cursor = cursor.limit(max(1, int(limit)))
            if batch_size is not None:
                cursor = cursor.batch_size(max(1, int(batch_size)))
            if max_time_ms is not None:
                cursor = cursor.max_time_ms(max(1, int(max_time_ms)))
            return list(cursor)
        except (NetworkTimeout, AutoReconnect, PyMongoError) as exc:
            last_error = exc
            logger.warning(
                "Mongo query attempt %s/%s failed with %s: %s",
                attempt,
                attempts,
                type(exc).__name__,
                exc,
            )
            if attempt >= attempts:
                break
            time.sleep(retry_delay_seconds)
            collection = _refresh_collection()

    if last_error is not None:
        raise last_error
    return []


def _find_price_documents_with_retry(
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    keep_ohlcv: bool = False,
):
    cleaned = _normalize_tickers(tickers)
    if not cleaned:
        return []

    if start_date and end_date:
        date_expr = {"$ifNull": ["$$hp.Date", "$$hp.date"]}
        cond = {
            "$and": [
                {"$gte": [date_expr, str(start_date)]},
                {"$lte": [date_expr, str(end_date)]},
            ]
        }
        projection = {
            "ticker": 1,
            "historical_prices": {
                "$filter": {
                    "input": "$historical_prices",
                    "as": "hp",
                    "cond": cond,
                }
            },
        }
    else:
        projection = {
            "ticker": 1,
            "historical_prices.Date": 1,
            "historical_prices.date": 1,
            "historical_prices.Close": 1,
            "historical_prices.close": 1,
        }

    if keep_ohlcv and not (start_date and end_date):
        projection.update({
            "historical_prices.Open": 1,
            "historical_prices.open": 1,
            "historical_prices.High": 1,
            "historical_prices.high": 1,
            "historical_prices.Low": 1,
            "historical_prices.low": 1,
            "historical_prices.Volume": 1,
            "historical_prices.volume": 1,
        })

    return _find_documents_with_retry(
        {"ticker": {"$in": cleaned}},
        projection,
        batch_size=50,
        max_time_ms=20000,
    )


def _lookup_cache_key(name: str, *parts: str) -> tuple:
    normalized_parts = tuple(str(part).strip().upper() for part in parts)
    return (name,) + normalized_parts


def _get_lookup_cache(key: tuple) -> Optional[str]:
    cached = _LOOKUP_CACHE.get(key)
    if not cached:
        return None

    expires_at, payload = cached
    if expires_at <= time.monotonic():
        _LOOKUP_CACHE.pop(key, None)
        return None

    return payload


def _set_lookup_cache(key: tuple, payload: str, ttl_seconds: int = LOOKUP_CACHE_TTL_SECONDS) -> str:
    _LOOKUP_CACHE[key] = (time.monotonic() + ttl_seconds, payload)
    return payload


def _downsample_df(df: pd.DataFrame, target_points: int = 500) -> pd.DataFrame:
    """Downsample a DataFrame to approximately target_points while preserving the first and last rows."""
    if df.empty or len(df) <= target_points:
        return df

    step = (len(df) + target_points - 1) // target_points
    if step <= 1:
        return df

    # Always include the very last row to ensure the latest price is present
    downsampled = df.iloc[::step].copy()
    if downsampled.index[-1] != df.index[-1]:
        downsampled = pd.concat([downsampled, df.tail(1)]).drop_duplicates()
        
    return downsampled


def _extract_price_frame(doc: dict, downsample: bool = False, keep_ohlcv: bool = False) -> pd.DataFrame:
    historical_prices = doc.get("historical_prices", [])
    if not historical_prices:
        return pd.DataFrame()

    df = pd.DataFrame(historical_prices).copy()
    if df.empty:
        return df

    date_col = "Date" if "Date" in df.columns else "date" if "date" in df.columns else None
    close_col = "Close" if "Close" in df.columns else "close" if "close" in df.columns else None

    if date_col is None or close_col is None:
        return pd.DataFrame()

    if keep_ohlcv:
        cols_to_keep = [date_col, close_col]
        for col in ["Open", "open", "High", "high", "Low", "low", "Volume", "volume"]:
            if col in df.columns:
                cols_to_keep.append(col)
        df = df[cols_to_keep].rename(columns={date_col: "Date", close_col: "Close"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        for col in ["Open", "open", "High", "high", "Low", "low", "Volume", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    else:
        df = df[[date_col, close_col]].rename(columns={date_col: "Date", close_col: "Close"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    
    if downsample:
        df = _downsample_df(df)
        
    return df


def _get_effective_price_on_or_before(df: pd.DataFrame, target_date: pd.Timestamp):
    eligible = df[df["Date"] <= target_date]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def _normalize_tickers(tickers: list[str]) -> list[str]:
    return sorted({t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()})


def _build_price_frames(docs_by_ticker: dict[str, dict]) -> dict[str, pd.DataFrame]:
    return {ticker: _extract_price_frame(doc) for ticker, doc in docs_by_ticker.items()}


def _warn_drop_ticker(ticker: str, reason: str, dropped_tickers: list[dict]) -> None:
    message = f"Dropping {ticker} from governance pipeline: {reason}"
    logger.warning(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    dropped_tickers.append({"ticker": ticker, "reason": reason})


def _prepare_portfolio_inputs(
    docs_by_ticker: dict[str, dict],
    price_frames: dict[str, pd.DataFrame],
    cleaned_tickers: list[str],
    target_dt: pd.Timestamp,
    target_date: str,
    lookback_window: int = 90,
    min_history: int = 20,
) -> dict:
    valid_tickers = []
    dropped_tickers = []
    price_snapshot = []
    price_history = {}
    effective_dates = {}
    price_series = {}
    data_sources = {}

    for ticker in cleaned_tickers:
        doc = docs_by_ticker.get(ticker)
        source = "MongoDB"
        if not doc:
            df = _cached_yfinance_history_frame(
                ticker,
                (target_dt - pd.Timedelta(days=540)).strftime("%Y-%m-%d"),
                (target_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            source = "yfinance"
        else:
            df = price_frames.get(ticker, pd.DataFrame())

        if df.empty:
            _warn_drop_ticker(ticker, "no MongoDB or yfinance historical price series", dropped_tickers)
            continue

        eligible = df[df["Date"] <= target_dt].copy()
        if eligible.empty:
            _warn_drop_ticker(ticker, f"no price data on or before {target_date}", dropped_tickers)
            continue

        trailing_window = eligible.tail(lookback_window).copy()
        if len(trailing_window) < min_history:
            _warn_drop_ticker(
                ticker,
                f"insufficient {lookback_window}-day lookback history before {target_date}",
                dropped_tickers,
            )
            continue

        valid_tickers.append(ticker)
        data_sources[ticker] = source
        effective_row = eligible.iloc[-1]
        effective_date = effective_row["Date"].strftime("%Y-%m-%d")
        effective_dates[ticker] = effective_date
        price_snapshot.append(
            {
                "ticker": ticker,
                "close": round(float(effective_row["Close"]), 6),
                "effective_date": effective_date,
            }
        )
        price_history[ticker] = [
            {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "close": round(float(row["Close"]), 6),
            }
            for _, row in trailing_window.iterrows()
        ]
        price_series[ticker] = trailing_window.set_index("Date")["Close"].rename(ticker)

    overlapping_prices = pd.DataFrame()
    if price_series:
        overlapping_prices = pd.concat(price_series.values(), axis=1).sort_index()
        overlapping_prices = overlapping_prices.ffill().dropna(how="any")

    return {
        "valid_tickers": valid_tickers,
        "dropped_tickers": dropped_tickers,
        "price_snapshot": price_snapshot,
        "price_history": price_history,
        "effective_dates": effective_dates,
        "overlapping_prices": overlapping_prices,
        "data_sources": data_sources,
    }


def _build_network_analysis_payload(docs_by_ticker: dict[str, dict], valid_tickers: list[str]) -> dict:
    graph = nx.Graph()
    holder_edges = []
    missing_network_data = []

    for ticker in valid_tickers:
        graph.add_node(ticker, bipartite=0)
        doc = docs_by_ticker.get(ticker)
        holders = doc.get("graph_relationships", {}).get("institutional_holders", []) if doc else []

        if not holders:
            missing_network_data.append({"ticker": ticker, "reason": "no institutional holder data"})
            continue

        for holder in holders:
            holder_name = holder.get("Holder")
            pct_str = str(holder.get("pctHeld", "0")).replace("%", "").strip()
            try:
                weight = float(pct_str)
            except ValueError:
                weight = 0.0

            if holder_name and weight > 0:
                graph.add_node(holder_name, bipartite=1)
                graph.add_edge(ticker, holder_name, weight=weight)
                holder_edges.append(
                    {
                        "ticker": ticker,
                        "holder": holder_name,
                        "weight": round(weight, 6),
                    }
                )

    if not valid_tickers:
        return {
            "method": "No eligible tickers",
            "scores": {},
            "holder_edges": [],
            "missing_network_data": missing_network_data,
        }

    if graph.number_of_edges() == 0:
        scores = {ticker: 0.0 for ticker in valid_tickers}
        method = "No holder data available"
    else:
        try:
            centrality = nx.eigenvector_centrality(graph, max_iter=2000, weight="weight")
            method = "Eigenvector Centrality"
        except Exception:
            centrality = nx.degree_centrality(graph)
            method = "Degree Centrality fallback"

        stock_centrality = {node: float(centrality.get(node, 0.0)) for node in valid_tickers}
        c_series = pd.Series(stock_centrality, dtype=float)
        c_min, c_max = c_series.min(), c_series.max()
        if c_max > c_min:
            normalized = (c_series - c_min) / (c_max - c_min)
        else:
            normalized = pd.Series(0.0, index=c_series.index, dtype=float)

        scores = {ticker: round(float(normalized.get(ticker, 0.0)), 6) for ticker in valid_tickers}

    return {
        "method": method,
        "scores": scores,
        "holder_edges": holder_edges,
        "missing_network_data": missing_network_data,
    }


def _annual_to_daily_return(annual_return: float, periods: int = 252) -> float:
    value = float(annual_return)
    if value <= -1.0:
        raise ValueError("annual return target must be greater than -1")
    return float((1.0 + value) ** (1.0 / periods) - 1.0)


def _portfolio_audit_metrics(
    weights: dict[str, float],
    graph_scores: Optional[dict[str, float]],
    previous_weights: Optional[dict[str, float]],
    max_weight_constraint: float,
) -> dict:
    current = pd.Series(weights, dtype=float).clip(lower=0.0)
    total = float(current.sum())
    if current.empty or total <= 0:
        return {
            "hhi": None,
            "effective_number_of_holdings": None,
            "graph_exposure": None,
            "turnover": None,
            "max_observed_weight": None,
            "weight_cap_utilization": None,
        }
    current = current / total
    hhi = float(np.square(current).sum())
    graph = pd.Series(graph_scores or {}, dtype=float).reindex(current.index).fillna(0.0)
    turnover = None
    if previous_weights:
        previous = pd.Series(previous_weights, dtype=float).clip(lower=0.0)
        previous = previous.reindex(current.index).fillna(0.0)
        previous_total = float(previous.sum())
        if previous_total > 0:
            previous = previous / previous_total
            turnover = float(0.5 * (current - previous).abs().sum())
    maximum = float(current.max())
    return {
        "hhi": hhi,
        "effective_number_of_holdings": float(1.0 / hhi) if hhi > 0 else None,
        "graph_exposure": float(current @ graph),
        "turnover": turnover,
        "max_observed_weight": maximum,
        "weight_cap_utilization": (
            float(maximum / max_weight_constraint) if max_weight_constraint > 0 else None
        ),
    }


_LIGHTWEIGHT_OPTIMIZATION_FIELDS = {
    "optimization_type",
    "risk_tolerance",
    "weights",
    "expected_annualized_return",
    "expected_cvar_95",
    "target_annual_return_floor",
    "target_daily_return_floor",
    "target_return_constraint_used",
    "fallback_applied",
    "fallback_reason",
    "solver_name",
    "solver_status",
    "objective_value",
    "max_weight_constraint",
    "max_observed_weight",
    "weight_cap_utilization",
    "hhi",
    "effective_number_of_holdings",
    "turnover",
    "graph_exposure",
    "instability_index",
    "lambda_t",
    "effective_window_start",
    "effective_window_end",
    "historical_pricing_dates",
    "graph_scores_used",
}


def _lightweight_optimization_payload(payload: Optional[dict]) -> dict:
    source = payload if isinstance(payload, dict) else {}
    return {key: source.get(key) for key in _LIGHTWEIGHT_OPTIMIZATION_FIELDS if key in source}


def _build_optimization_payload(
    overlapping_prices: pd.DataFrame,
    effective_dates: dict[str, str],
    target_date: str,
    risk_tolerance: str = "moderate",
    network_scores: Optional[dict[str, float]] = None,
    previous_weights: Optional[dict[str, float]] = None,
    lambda_max: float = 1.0,
    k: float = 10.0,
    i_thresh: float = 0.85,
) -> dict:
    if overlapping_prices.empty or overlapping_prices.shape[1] < 2:
        return {
            "status": "error",
            "message": (
                f"Not enough overlapping historical prices remained to run optimization for {target_date}."
            ),
        }

    if len(overlapping_prices) < 20:
        return {
            "status": "error",
            "message": (
                f"Fewer than 20 overlapping historical observations remained before {target_date}."
            ),
        }

    log_returns = np.log(overlapping_prices / overlapping_prices.shift(1))
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if log_returns.empty or len(log_returns) < 20:
        return {
            "status": "error",
            "message": f"Insufficient clean return history remained to run optimization for {target_date}.",
        }

    asset_names = list(log_returns.columns)
    returns_matrix = log_returns.to_numpy()
    num_periods, num_assets = returns_matrix.shape
    beta = 0.95
    c_vector = np.array(
        [float((network_scores or {}).get(ticker, 0.0)) for ticker in asset_names],
        dtype=float,
    )

    mean_daily_returns = log_returns.mean().to_numpy()
    mean_annual_returns = mean_daily_returns * 252.0
    covariance_matrix = log_returns.cov().round(6)
    correlation_matrix = log_returns.corr().round(6)

    mean_volatility = float(log_returns.std().mean() * np.sqrt(252.0))
    # Approximation of correlation
    upper_tri = correlation_matrix.to_numpy()[np.triu_indices(correlation_matrix.shape[0], k=1)]
    mean_correlation = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0
    # Approximation of drawdown
    cum_returns = np.exp(log_returns.cumsum())
    max_cum_returns = cum_returns.cummax()
    drawdowns = (cum_returns - max_cum_returns) / max_cum_returns.replace(0.0, np.nan)
    mean_drawdown = float(np.abs(drawdowns.min()).mean())
    
    vol_norm = float(np.clip((mean_volatility - 0.10) / (0.60 - 0.10), 0.0, 1.0))
    corr_norm = float(np.clip((mean_correlation - 0.10) / (0.80 - 0.10), 0.0, 1.0))
    dd_norm = float(np.clip((mean_drawdown - 0.05) / (0.40 - 0.05), 0.0, 1.0))

    raw_instability_index = 0.4 * vol_norm + 0.3 * corr_norm + 0.3 * dd_norm
    instability_index = float(np.clip(raw_instability_index, 0.0, 1.0))

    profile = (risk_tolerance or "moderate").strip().lower()
    if profile not in {"conservative", "moderate", "aggressive"}:
        profile = "moderate"

    percentile_map = {
        "conservative": 25,
        "moderate": 50,
        "aggressive": 75,
    }
    target_annual_return = float(np.percentile(mean_annual_returns, percentile_map[profile]))
    try:
        target_daily_return = _annual_to_daily_return(target_annual_return)
    except ValueError as exc:
        return {
            "status": "error",
            "message": f"Unable to use the annual return target for {target_date}: {exc}.",
        }

    weights = cp.Variable(num_assets)
    alpha = cp.Variable()
    tail_excess = cp.Variable(num_periods, nonneg=True)

    portfolio_returns = returns_matrix @ weights
    losses = -portfolio_returns
    cvar_95 = alpha + (1.0 / ((1.0 - beta) * num_periods)) * cp.sum(tail_excess)
    lambda_t = float(lambda_max / (1.0 + np.exp(-k * (instability_index - i_thresh))))
    graph_penalty = lambda_t * (c_vector @ weights)

    max_weight_limit = max(0.15, 1.2 / num_assets)
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= max_weight_limit,
        tail_excess >= losses - alpha,
        mean_daily_returns @ weights >= target_daily_return,
    ]

    solver_name = None
    solver_status = None

    def solve_with(active_constraints):
        nonlocal solver_name, solver_status
        candidate = cp.Problem(cp.Minimize(cvar_95 + graph_penalty), active_constraints)
        for solver in [cp.CLARABEL, cp.OSQP, cp.SCS]:
            try:
                candidate.solve(solver=solver, verbose=False)
            except Exception:
                continue
            solver_status = str(candidate.status or "")
            if candidate.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and weights.value is not None:
                solver_name = str(solver)
                return candidate, True
        solver_status = str(candidate.status or solver_status or "solver_error")
        return candidate, False

    problem, solved = solve_with(constraints)
    fallback_applied = False
    fallback_reason = None
    target_return_constraint_used = True

    if not solved or weights.value is None:
        fallback_applied = True
        fallback_reason = "profile return floor was infeasible; solved without a return-floor constraint"
        target_return_constraint_used = False
        capped_fallback_constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            weights <= max_weight_limit,
            tail_excess >= losses - alpha,
        ]
        problem, solved = solve_with(capped_fallback_constraints)

    if not solved or weights.value is None:
        return {
            "status": "error",
            "message": (
                f"Historical CVaR optimization could not find a capped solution for {target_date}. "
                f"Solver status: {solver_status or problem.status}."
            ),
            "max_weight_constraint": round(max_weight_limit, 6),
            "fallback_applied": fallback_applied,
            "fallback_reason": (
                "profile return floor was infeasible and no capped fallback solution was available"
                if fallback_applied
                else "no capped solution was available"
            ),
        }

    optimal_weights = np.maximum(np.asarray(weights.value).reshape(-1), 0.0)
    optimal_weights[optimal_weights < 0.01] = 0.0

    if float(optimal_weights.sum()) <= 0:
        return {
            "status": "error",
            "message": (
                f"Historical CVaR optimization failed for {target_date}: all weights were negligible after cleanup."
            ),
        }

    optimal_weights = optimal_weights / optimal_weights.sum()
    if float(optimal_weights.max()) > max_weight_limit + 1e-6:
        return {
            "status": "error",
            "message": (
                f"Historical CVaR optimization cleanup exceeded the maximum-weight constraint "
                f"for {target_date}."
            ),
            "max_weight_constraint": round(max_weight_limit, 6),
        }
    weights_map = {
        ticker: round(float(weight), 6)
        for ticker, weight in sorted(zip(asset_names, optimal_weights), key=lambda item: item[1], reverse=True)
        if weight > 0
    }

    expected_annualized_return = float(mean_annual_returns @ optimal_weights)
    realized_portfolio_returns = returns_matrix @ optimal_weights
    portfolio_losses = -realized_portfolio_returns
    var_95 = float(np.quantile(portfolio_losses, beta))
    tail_losses = portfolio_losses[portfolio_losses >= var_95]
    expected_cvar_95 = float(tail_losses.mean()) if len(tail_losses) > 0 else var_95

    graph_scores_used = {
        ticker: round(float(score), 6)
        for ticker, score in zip(asset_names, c_vector)
    }
    audit_metrics = _portfolio_audit_metrics(
        weights=weights_map,
        graph_scores=graph_scores_used,
        previous_weights=previous_weights,
        max_weight_constraint=max_weight_limit,
    )
    logger.info(
        "Governance optimizer target_date=%s solver=%s status=%s cap=%.6f fallback=%s reason=%s",
        target_date,
        solver_name,
        solver_status,
        max_weight_limit,
        fallback_applied,
        fallback_reason or "none",
    )

    return {
        "status": "success",
        "optimization_type": "graph_regularized_cvar",
        "risk_tolerance": profile,
        "weights": weights_map,
        "expected_annualized_return": round(expected_annualized_return, 6),
        "expected_cvar_95": round(expected_cvar_95, 6),
        "target_annual_return_floor": round(target_annual_return, 6),
        "target_daily_return_floor": round(target_daily_return, 10),
        "target_return_constraint_used": target_return_constraint_used,
        "fallback_applied": fallback_applied,
        "fallback_reason": fallback_reason,
        "solver_name": solver_name,
        "solver_status": solver_status,
        "max_weight_constraint": round(max_weight_limit, 6),
        "instability_index": round(instability_index, 6),
        "lambda_t": round(lambda_t, 6),
        "graph_scores_used": graph_scores_used,
        "objective_value": round(float(problem.value), 6) if problem.value is not None else None,
        "effective_window_start": overlapping_prices.index.min().strftime("%Y-%m-%d"),
        "effective_window_end": overlapping_prices.index.max().strftime("%Y-%m-%d"),
        "historical_pricing_dates": {
            ticker: effective_dates[ticker]
            for ticker in asset_names
            if ticker in effective_dates
        },
        "correlation_matrix": correlation_matrix.to_dict(),
        "covariance_matrix": covariance_matrix.to_dict(),
        **audit_metrics,
    }


def _generate_inline_governance_plots(
    target_date: str,
    weights: dict[str, float],
    network_payload: dict,
    config=None,
) -> list[str]:
    generated_plots = []
    plot_requests = []
    risk_scores = network_payload.get("scores", {}) if isinstance(network_payload, dict) else {}
    holder_edges = network_payload.get("holder_edges", []) if isinstance(network_payload, dict) else {}

    if weights:
        plot_requests.append(
            {
                "plot_type": "pie",
                "title": f"Advisory Allocation Weights as of {target_date}",
                "data": {"weights": weights},
            }
        )

    if risk_scores or holder_edges:
        plot_requests.append(
            {
                "plot_type": "network",
                "title": f"Institutional Risk Network as of {target_date}",
                "data": {
                    "holder_edges": holder_edges,
                    "risk_scores": risk_scores,
                },
            }
        )

    for request in plot_requests:
        try:
            plot_output = _generate_financial_plot().invoke(
                {
                    "data": request["data"],
                    "plot_type": request["plot_type"],
                    "title": request["title"],
                },
                config=config,
            )
        except Exception as exc:
            logger.warning("Unable to generate %s plot inline: %s", request["plot_type"], exc)
            continue

        if isinstance(plot_output, str) and plot_output.startswith("Chart ready"):
            generated_plots.append(plot_output)
            continue

        # Legacy PNG fallback (heatmap / network)
        if isinstance(plot_output, str) and "![" in plot_output:
            generated_plots.append(plot_output)
            continue

        logger.warning(
            "Plot tool returned unexpected response for %s: %s",
            request["plot_type"],
            plot_output,
        )

    return generated_plots


def _build_lightweight_governance_payload(
    status: str,
    message: str,
    target_date: str,
    valid_tickers: list[str],
    dropped_tickers: Optional[list[dict]] = None,
    systemic_risk: Optional[dict] = None,
    optimization: Optional[dict] = None,
    generated_plots: Optional[list[str]] = None,
    data_sources: Optional[dict[str, str]] = None,
) -> dict:
    return {
        "status": status,
        "message": message,
        "target_date": str(target_date or ""),
        "valid_tickers": valid_tickers or [],
        "dropped_tickers": dropped_tickers or [],
        "data_sources": data_sources or {},
        "systemic_risk": systemic_risk or {"method": "Unavailable", "scores": {}},
        "optimization": optimization or {},
        "generated_plots": generated_plots or [],
    }


def _run_price_snapshot_from_frames(
    price_frames: dict[str, pd.DataFrame],
    cleaned_tickers: list[str],
    target_dt: pd.Timestamp,
    target_date: str,
) -> str:
    lines = []
    missing = []

    for ticker in cleaned_tickers:
        df = price_frames.get(ticker, pd.DataFrame())
        if df.empty:
            missing.append(f"{ticker} (no historical price series)")
            continue

        row = _get_effective_price_on_or_before(df, target_dt)
        if row is None:
            missing.append(f"{ticker} (no price on or before {target_date})")
            continue

        lines.append(f"- {ticker}: close={row['Close']:.2f} on {row['Date'].strftime('%Y-%m-%d')}")

    if not lines:
        return (
            f"Unable to fetch historical prices for {target_date}. "
            f"No requested tickers had usable data on or before that date. "
            f"Missing details: {', '.join(missing) if missing else 'none'}"
        )

    response = [
        f"Historical closing prices on or immediately before {target_date}:",
        *lines,
    ]

    if missing:
        response.append("")
        response.append("Missing or unavailable:")
        response.extend(f"- {item}" for item in missing)

    return "\n".join(response)


def _run_network_analysis_from_docs(docs_by_ticker: dict[str, dict], cleaned_tickers: list[str]) -> str:
    graph = nx.Graph()
    stock_nodes = []
    missing = []

    for ticker in cleaned_tickers:
        doc = docs_by_ticker.get(ticker)
        if not doc:
            missing.append(f"{ticker} (ticker not found)")
            continue

        holders = doc.get("graph_relationships", {}).get("institutional_holders", [])
        if not holders:
            missing.append(f"{ticker} (no institutional holder data)")
            continue

        stock_nodes.append(ticker)
        graph.add_node(ticker, bipartite=0)

        for holder in holders:
            holder_name = holder.get("Holder")
            pct_str = str(holder.get("pctHeld", "0")).replace("%", "").strip()
            try:
                weight = float(pct_str)
            except ValueError:
                weight = 0.0

            if holder_name and weight > 0:
                graph.add_node(holder_name, bipartite=1)
                graph.add_edge(ticker, holder_name, weight=weight)

    if not stock_nodes:
        details = ", ".join(missing) if missing else "no eligible tickers"
        return f"Unable to analyze institutional network: {details}."

    try:
        centrality = nx.eigenvector_centrality(graph, max_iter=2000, weight="weight")
        method = "Eigenvector Centrality"
    except Exception:
        centrality = nx.degree_centrality(graph)
        method = "Degree Centrality fallback"

    stock_centrality = {node: score for node, score in centrality.items() if node in stock_nodes}
    if not stock_centrality:
        return "Unable to analyze institutional network: graph centrality could not be computed for the requested tickers."

    c_series = pd.Series(stock_centrality)
    c_min, c_max = c_series.min(), c_series.max()
    if c_max > c_min:
        normalized = (c_series - c_min) / (c_max - c_min)
    else:
        normalized = pd.Series(0.0, index=c_series.index)

    lines = [
        "Institutional Network Risk Analysis",
        f"Method used: {method}",
        "Normalized structural risk scores:",
    ]
    for ticker, score in normalized.sort_values(ascending=False).items():
        lines.append(f"- {ticker}: {score:.4f}")

    if missing:
        lines.append("")
        lines.append("Unavailable tickers:")
        lines.extend(f"- {item}" for item in missing)

    return "\n".join(lines)


def _run_historical_cvar_from_frames(
    price_frames: dict[str, pd.DataFrame],
    cleaned_tickers: list[str],
    target_dt: pd.Timestamp,
    target_date: str,
    risk_tolerance: str = "moderate",
) -> str:
    if len(cleaned_tickers) < 2:
        return "Unable to run historical CVaR optimization: please provide at least two valid tickers."

    price_series = {}
    effective_dates = {}
    missing = []

    for ticker in cleaned_tickers:
        df = price_frames.get(ticker, pd.DataFrame())
        if df.empty:
            missing.append(f"{ticker} (no historical price series)")
            continue

        eligible = df[df["Date"] <= target_dt].copy()
        if eligible.empty:
            missing.append(f"{ticker} (no data on or before {target_date})")
            continue

        effective_dates[ticker] = eligible["Date"].iloc[-1].strftime("%Y-%m-%d")
        trailing_window = eligible.tail(90).copy()

        if len(trailing_window) < 20:
            missing.append(f"{ticker} (insufficient history before {target_date})")
            continue

        series = trailing_window.set_index("Date")["Close"].rename(ticker)
        price_series[ticker] = series

    if len(price_series) < 2:
        return (
            f"Unable to run historical CVaR optimization for {target_date}: fewer than two tickers had enough "
            f"usable trailing history. Missing details: {', '.join(missing) if missing else 'none'}"
        )

    prices = pd.concat(price_series.values(), axis=1).sort_index()
    prices = prices.ffill().dropna(how="any")

    if len(prices) < 20 or prices.shape[1] < 2:
        return (
            f"Unable to run historical CVaR optimization for {target_date}: not enough overlapping historical "
            f"prices across the requested tickers."
        )

    log_returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if log_returns.empty or len(log_returns) < 20:
        return f"Unable to run historical CVaR optimization for {target_date}: insufficient clean return history."

    asset_names = list(log_returns.columns)
    returns_matrix = log_returns.to_numpy()
    num_periods, num_assets = returns_matrix.shape
    beta = 0.95

    mean_daily_returns = log_returns.mean().to_numpy()
    mean_annual_returns = mean_daily_returns * 252.0

    profile = (risk_tolerance or "moderate").strip().lower()
    if profile not in {"conservative", "moderate", "aggressive"}:
        profile = "moderate"

    percentile_map = {
        "conservative": 25,
        "moderate": 50,
        "aggressive": 75,
    }
    target_annual_return = float(np.percentile(mean_annual_returns, percentile_map[profile]))
    target_daily_return = target_annual_return / 252.0

    weights = cp.Variable(num_assets)
    alpha = cp.Variable()
    tail_excess = cp.Variable(num_periods, nonneg=True)

    portfolio_returns = returns_matrix @ weights
    losses = -portfolio_returns

    cvar_95 = alpha + (1.0 / ((1.0 - beta) * num_periods)) * cp.sum(tail_excess)

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= 0.15,
        tail_excess >= losses - alpha,
        mean_daily_returns @ weights >= target_daily_return,
    ]

    problem = cp.Problem(cp.Minimize(cvar_95), constraints)

    for solver in [cp.CLARABEL, cp.OSQP, cp.SCS]:
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                break
        except Exception:
            continue

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weights.value is None:
        return (
            f"Historical CVaR optimization could not find a stable solution for {target_date}. "
            f"Solver status: {problem.status}."
        )

    optimal_weights = np.maximum(np.asarray(weights.value).reshape(-1), 0.0)
    optimal_weights[optimal_weights < 0.01] = 0.0

    if float(optimal_weights.sum()) <= 0:
        return f"Historical CVaR optimization failed for {target_date}: all weights were negligible after cleanup."

    optimal_weights = optimal_weights / optimal_weights.sum()
    expected_annualized_return = float(mean_annual_returns @ optimal_weights)

    realized_portfolio_returns = returns_matrix @ optimal_weights
    portfolio_losses = -realized_portfolio_returns
    var_95 = float(np.quantile(portfolio_losses, beta))
    tail_losses = portfolio_losses[portfolio_losses >= var_95]
    expected_cvar_95 = float(tail_losses.mean()) if len(tail_losses) > 0 else var_95

    allocation_lines = []
    for ticker, weight in sorted(zip(asset_names, optimal_weights), key=lambda item: item[1], reverse=True):
        if weight > 0:
            allocation_lines.append(f"- {ticker}: {weight * 100:.2f}%")

    response = [
        "Historical CVaR Optimization Result",
        f"Target date requested: {target_date}",
        f"Effective price window end: {prices.index.max().strftime('%Y-%m-%d')}",
        f"Risk tolerance: {profile.capitalize()}",
        f"Universe: {', '.join(asset_names)}",
        f"Target annual return floor used in optimization: {target_annual_return * 100:.2f}%",
        "",
        "Optimal allocation weights:",
        *allocation_lines,
        "",
        f"Estimated/backtested annualized portfolio return: {expected_annualized_return * 100:.2f}%",
        f"Expected 95% CVaR (daily tail risk): {expected_cvar_95 * 100:.2f}%",
        "",
        "Historical pricing dates used at or before the target date:",
        *[f"- {ticker}: {effective_dates[ticker]}" for ticker in asset_names if ticker in effective_dates],
    ]

    if missing:
        response.extend(["", "Excluded or unavailable tickers:"])
        response.extend(f"- {item}" for item in missing)

    return "\n".join(response)


def _summarize_metrics(metrics: dict, max_items: int = 8) -> list[str]:
    if not isinstance(metrics, dict) or not metrics:
        return []

    lines = []
    for key, value in list(metrics.items())[:max_items]:
        lines.append(f"  - {key}: {value}")
    return lines


def _get_yfinance_module():
    try:
        import yfinance as yf
    except ImportError:
        return None
    return yf


def _get_yfinance_info(ticker_obj) -> dict:
    try:
        if hasattr(ticker_obj, "get_info"):
            info = ticker_obj.get_info()
        else:
            info = getattr(ticker_obj, "info", {})
    except Exception:
        info = {}
    return info if isinstance(info, dict) else {}


def _json_safe_value(value):
    if pd.isna(value) if not isinstance(value, (list, dict, tuple, set)) else False:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _frame_to_json_records(frame, max_rows: int = 5000) -> list[dict]:
    if frame is None:
        return []
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=frame.name or "value")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []

    normalized = frame.reset_index().copy()
    normalized.columns = [str(column) for column in normalized.columns]
    if len(normalized) > max_rows:
        normalized = normalized.tail(max_rows)
    normalized = normalized.replace({np.nan: None})
    records = normalized.to_dict(orient="records")
    return [
        {str(key): _json_safe_value(value) for key, value in row.items()}
        for row in records
    ]


def _series_to_json_records(series, max_rows: int = 5000) -> list[dict]:
    if series is None or not isinstance(series, pd.Series) or series.empty:
        return []
    frame = series.tail(max_rows).rename(series.name or "value").reset_index()
    frame.columns = [str(column) for column in frame.columns]
    return [
        {str(key): _json_safe_value(value) for key, value in row.items()}
        for row in frame.replace({np.nan: None}).to_dict(orient="records")
    ]


def _safe_yfinance_frame(ticker_obj, attr_name: str) -> list[dict]:
    try:
        value = getattr(ticker_obj, attr_name)
    except Exception as exc:
        logger.warning("yfinance %s fetch failed: %s", attr_name, exc)
        return []
    return _frame_to_json_records(value)


def _safe_yfinance_series(ticker_obj, attr_name: str) -> list[dict]:
    try:
        value = getattr(ticker_obj, attr_name)
    except Exception as exc:
        logger.warning("yfinance %s fetch failed: %s", attr_name, exc)
        return []
    return _series_to_json_records(value)


def _normalize_yfinance_data_type(data_type: str | None) -> str:
    normalized = str(data_type or "history").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "price": "history",
        "prices": "history",
        "ohlcv": "history",
        "close": "history",
        "profile": "info",
        "company": "info",
        "fundamentals": "financials",
        "income_statement": "financials",
        "balance": "balance_sheet",
        "cash_flow": "cashflow",
        "recommendation": "recommendations",
        "institutional_holders": "holders",
        "holder": "holders",
        "everything": "all",
        "any": "all",
        "auto": "all",
    }
    return aliases.get(normalized, normalized)


def _fetch_yfinance_market_payload(
    symbol: str,
    data_type: str,
    period: str = "1y",
    interval: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    option_expiration: str | None = None,
) -> dict | None:
    yf = _get_yfinance_module()
    if yf is None:
        return None

    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        return None

    normalized_type = _normalize_yfinance_data_type(data_type)
    try:
        ticker_obj = yf.Ticker(normalized_symbol)
    except Exception as exc:
        logger.warning("yfinance ticker initialization failed for %s: %s", normalized_symbol, exc)
        return None

    payload: dict[str, object] = {
        "symbol": normalized_symbol,
        "data_type": normalized_type,
        "period": period,
        "interval": interval,
        "start_date": start_date,
        "end_date": end_date,
        "source": "yfinance",
    }

    def add_history() -> None:
        try:
            history_kwargs = {"interval": interval or "1d", "auto_adjust": False}
            if start_date or end_date:
                if start_date:
                    history_kwargs["start"] = start_date
                if end_date:
                    history_kwargs["end"] = end_date
            else:
                history_kwargs["period"] = period or "1y"
            payload["history"] = _frame_to_json_records(ticker_obj.history(**history_kwargs))
        except Exception as exc:
            logger.warning("yfinance history fetch failed for %s: %s", normalized_symbol, exc)
            payload["history"] = []

    def add_info() -> None:
        info = _get_yfinance_info(ticker_obj)
        payload["info"] = {str(key): _json_safe_value(value) for key, value in info.items()}

    if normalized_type in {"history", "all"}:
        add_history()
    if normalized_type in {"info", "all"}:
        add_info()
    if normalized_type in {"financials", "all"}:
        payload["financials"] = _safe_yfinance_frame(ticker_obj, "financials")
    if normalized_type in {"balance_sheet", "all"}:
        payload["balance_sheet"] = _safe_yfinance_frame(ticker_obj, "balance_sheet")
    if normalized_type in {"cashflow", "all"}:
        payload["cashflow"] = _safe_yfinance_frame(ticker_obj, "cashflow")
    if normalized_type in {"dividends", "all"}:
        payload["dividends"] = _safe_yfinance_series(ticker_obj, "dividends")
    if normalized_type in {"splits", "all"}:
        payload["splits"] = _safe_yfinance_series(ticker_obj, "splits")
    if normalized_type in {"holders", "all"}:
        payload["major_holders"] = _safe_yfinance_frame(ticker_obj, "major_holders")
        payload["institutional_holders"] = _safe_yfinance_frame(ticker_obj, "institutional_holders")
        payload["mutualfund_holders"] = _safe_yfinance_frame(ticker_obj, "mutualfund_holders")
    if normalized_type in {"recommendations", "all"}:
        payload["recommendations"] = _safe_yfinance_frame(ticker_obj, "recommendations")
    if normalized_type in {"options", "all"}:
        try:
            expirations = list(getattr(ticker_obj, "options", []) or [])
            payload["options_expirations"] = expirations
            selected_expiration = option_expiration or (expirations[0] if normalized_type == "options" and expirations else None)
            if selected_expiration:
                chain = ticker_obj.option_chain(selected_expiration)
                payload["options"] = {
                    "expiration": selected_expiration,
                    "calls": _frame_to_json_records(getattr(chain, "calls", None)),
                    "puts": _frame_to_json_records(getattr(chain, "puts", None)),
                }
        except Exception as exc:
            logger.warning("yfinance options fetch failed for %s: %s", normalized_symbol, exc)
            payload["options_expirations"] = []

    if len(payload) <= 7:
        return None
    return payload


def _market_payload_summary(payload: dict, from_cache: bool = False) -> str:
    symbol = payload.get("symbol", "UNKNOWN")
    data_type = payload.get("data_type", "market_data")
    lines = [
        f"Market data for {symbol}",
        f"- Source: {payload.get('source', 'yfinance')}{' cache' if from_cache else ''}",
        f"- Data type: {data_type}",
    ]
    for key in [
        "history",
        "financials",
        "balance_sheet",
        "cashflow",
        "dividends",
        "splits",
        "recommendations",
        "major_holders",
        "institutional_holders",
        "mutualfund_holders",
    ]:
        value = payload.get(key)
        if isinstance(value, list):
            lines.append(f"- {key}: {len(value)} rows")
    info = payload.get("info")
    if isinstance(info, dict) and info:
        name = info.get("longName") or info.get("shortName") or symbol
        sector = info.get("sector") or "Unknown sector"
        industry = info.get("industry") or "Unknown industry"
        lines.append(f"- Company: {name}")
        lines.append(f"- Classification: {sector} / {industry}")
    expirations = payload.get("options_expirations")
    if isinstance(expirations, list):
        lines.append(f"- options_expirations: {len(expirations)} dates")
    options = payload.get("options")
    if isinstance(options, dict):
        calls = options.get("calls") if isinstance(options.get("calls"), list) else []
        puts = options.get("puts") if isinstance(options.get("puts"), list) else []
        lines.append(f"- options chain {options.get('expiration')}: {len(calls)} calls, {len(puts)} puts")
    lines.append("The full payload is cached for follow-up analysis.")
    return "\n".join(lines)


MARKET_DATA_REQUIREMENT_KEYWORDS = {
    "history": [
        "history",
        "historical",
        "ohlcv",
        "open",
        "high",
        "low",
        "close",
        "closing",
        "volume",
        "price trend",
        "chart",
    ],
    "info": [
        "quote",
        "current price",
        "market cap",
        "valuation",
        "pe",
        "p/e",
        "beta",
        "company",
        "profile",
        "sector",
        "industry",
        "business",
        "summary",
        "employees",
    ],
    "financials": [
        "financials",
        "income",
        "income statement",
        "revenue",
        "sales",
        "profit",
        "net income",
        "ebitda",
        "operating income",
    ],
    "balance_sheet": [
        "balance",
        "balance sheet",
        "assets",
        "liabilities",
        "equity",
        "debt",
        "cash",
    ],
    "cashflow": [
        "cash flow",
        "cashflow",
        "operating cash",
        "free cash",
        "capex",
        "capital expenditure",
    ],
    "dividends": ["dividend", "dividends", "yield"],
    "splits": ["split", "splits"],
    "holders": ["holder", "holders", "institution", "institutional", "mutual fund", "ownership"],
    "recommendations": ["recommendation", "recommendations", "analyst", "rating", "buy", "sell", "hold"],
    "options": ["option", "options", "calls", "puts", "strike", "expiry", "expiration"],
}


MARKET_METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "current_price": {
        "label": "Current price",
        "data_type": "info",
        "aliases": ["current price", "latest price", "quote", "stock price"],
        "info_keys": ["currentPrice", "regularMarketPrice", "previousClose"],
    },
    "market_cap": {
        "label": "Market cap",
        "data_type": "info",
        "aliases": ["market cap", "market capitalization", "valuation"],
        "info_keys": ["marketCap"],
    },
    "trailing_pe": {
        "label": "Trailing P/E",
        "data_type": "info",
        "aliases": ["trailing pe", "trailing p/e", "pe ratio", "p/e ratio"],
        "info_keys": ["trailingPE"],
    },
    "forward_pe": {
        "label": "Forward P/E",
        "data_type": "info",
        "aliases": ["forward pe", "forward p/e"],
        "info_keys": ["forwardPE"],
    },
    "beta": {
        "label": "Beta",
        "data_type": "info",
        "aliases": ["beta"],
        "info_keys": ["beta"],
    },
    "dividend_yield": {
        "label": "Dividend yield",
        "data_type": "info",
        "aliases": ["dividend yield", "yield"],
        "info_keys": ["dividendYield", "trailingAnnualDividendYield"],
    },
    "total_revenue": {
        "label": "Total revenue",
        "data_type": "financials",
        "aliases": ["revenue", "total revenue", "sales"],
        "statement_rows": ["total revenue", "revenue"],
    },
    "net_income": {
        "label": "Net income",
        "data_type": "financials",
        "aliases": ["net income", "profit", "earnings"],
        "statement_rows": ["net income", "net income common stockholders"],
    },
    "gross_profit": {
        "label": "Gross profit",
        "data_type": "financials",
        "aliases": ["gross profit"],
        "statement_rows": ["gross profit"],
    },
    "operating_income": {
        "label": "Operating income",
        "data_type": "financials",
        "aliases": ["operating income", "operating profit"],
        "statement_rows": ["operating income"],
    },
    "ebitda": {
        "label": "EBITDA",
        "data_type": "financials",
        "aliases": ["ebitda"],
        "statement_rows": ["ebitda", "normalized ebitda"],
    },
    "total_assets": {
        "label": "Total assets",
        "data_type": "balance_sheet",
        "aliases": ["total assets", "assets"],
        "statement_rows": ["total assets"],
    },
    "total_liabilities": {
        "label": "Total liabilities",
        "data_type": "balance_sheet",
        "aliases": ["total liabilities", "liabilities"],
        "statement_rows": ["total liabilities net minority interest", "total liabilities"],
    },
    "total_debt": {
        "label": "Total debt",
        "data_type": "balance_sheet",
        "aliases": ["total debt", "debt"],
        "statement_rows": ["total debt"],
    },
    "cash": {
        "label": "Cash and equivalents",
        "data_type": "balance_sheet",
        "aliases": ["cash", "cash equivalents"],
        "statement_rows": ["cash and cash equivalents", "cash cash equivalents and short term investments"],
    },
    "operating_cash_flow": {
        "label": "Operating cash flow",
        "data_type": "cashflow",
        "aliases": ["operating cash flow", "cash from operations"],
        "statement_rows": ["operating cash flow", "total cash from operating activities"],
    },
    "free_cash_flow": {
        "label": "Free cash flow",
        "data_type": "cashflow",
        "aliases": ["free cash flow", "fcf"],
        "statement_rows": ["free cash flow"],
    },
    "capital_expenditure": {
        "label": "Capital expenditure",
        "data_type": "cashflow",
        "aliases": ["capital expenditure", "capex"],
        "statement_rows": ["capital expenditure", "capital expenditures"],
    },
    "latest_close": {
        "label": "Latest close",
        "data_type": "history",
        "aliases": ["latest close", "closing price", "close price", "close"],
    },
}


def _normalize_market_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _infer_market_data_requirements(request: str = "", metrics: list[str] | None = None) -> tuple[list[str], list[str]]:
    text = _normalize_market_text(" ".join([request or "", " ".join(metrics or [])]))
    requested_types: set[str] = set()
    requested_metrics: list[str] = []

    if any(phrase in text for phrase in ["all data", "all available", "everything", "any kind of data", "complete data"]):
        requested_types.update(
            ["history", "info", "financials", "balance_sheet", "cashflow", "dividends", "splits", "holders", "recommendations", "options"]
        )

    for metric_key, definition in MARKET_METRIC_DEFINITIONS.items():
        aliases = definition.get("aliases", [])
        if any(_normalize_market_text(alias) in text for alias in aliases):
            requested_metrics.append(metric_key)
            requested_types.add(str(definition["data_type"]))

    for data_type, keywords in MARKET_DATA_REQUIREMENT_KEYWORDS.items():
        if any(_normalize_market_text(keyword) in text for keyword in keywords):
            requested_types.add(data_type)

    if not requested_types:
        requested_types.add("info")
    return sorted(requested_types), requested_metrics


def _fetch_cached_yfinance_market_payload(
    symbol: str,
    data_type: str,
    period: str = "1y",
    interval: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    option_expiration: str | None = None,
) -> tuple[dict | None, bool]:
    normalized_symbol = str(symbol or "").strip().upper()
    normalized_type = _normalize_yfinance_data_type(data_type)
    params = {"option_expiration": option_expiration or ""}
    cache_key = memory_manager.compute_market_data_cache_key(
        symbol=normalized_symbol,
        data_type=normalized_type,
        period=period,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        params=params,
    )
    cached = memory_manager.retrieve_market_data_cache(cache_key)
    if cached and isinstance(cached.get("payload"), dict):
        return cached["payload"], True

    payload = _fetch_yfinance_market_payload(
        symbol=normalized_symbol,
        data_type=normalized_type,
        period=period,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        option_expiration=option_expiration,
    )
    if payload:
        memory_manager.store_market_data_cache(
            cache_key=cache_key,
            symbol=normalized_symbol,
            data_type=normalized_type,
            payload=payload,
            period=period,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            source="yfinance",
            ttl_hours=1 if normalized_type == "options" else 24,
        )
    return payload, False


def _coerce_numeric(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_market_value(value: object) -> str:
    numeric = _coerce_numeric(value)
    if numeric is None:
        return str(value) if value not in (None, "") else "N/A"
    absolute = abs(numeric)
    if absolute >= 1_000_000_000_000:
        return f"{numeric / 1_000_000_000_000:.2f}T"
    if absolute >= 1_000_000_000:
        return f"{numeric / 1_000_000_000:.2f}B"
    if absolute >= 1_000_000:
        return f"{numeric / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{numeric:,.2f}"
    return f"{numeric:.4g}"


_CURRENCY_SYMBOLS = {
    "USD": "$",
    "INR": "₹",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CNY": "¥",
    "CAD": "C$",
    "AUD": "A$",
    "HKD": "HK$",
    "SGD": "S$",
}

_EXCHANGE_CURRENCY_HINTS = {
    ".NS": "INR",
    ".BO": "INR",
    ".TO": "CAD",
    ".V": "CAD",
    ".L": "GBP",
    ".PA": "EUR",
    ".DE": "EUR",
    ".F": "EUR",
    ".HK": "HKD",
    ".SS": "CNY",
    ".SZ": "CNY",
    ".T": "JPY",
    ".AX": "AUD",
    ".SI": "SGD",
}

_NON_MONEY_METRICS = {"trailing_pe", "forward_pe", "beta", "dividend_yield"}


def _is_money_metric(metric_key: str) -> bool:
    return metric_key not in _NON_MONEY_METRICS


def _currency_hint_for_symbol(symbol: str) -> str | None:
    normalized = str(symbol or "").strip().upper()
    for suffix, currency in _EXCHANGE_CURRENCY_HINTS.items():
        if normalized.endswith(suffix):
            return currency
    return None


def _currency_from_payloads(symbol: str, payloads: dict[str, dict]) -> str | None:
    for payload in payloads.values():
        info = payload.get("info") if isinstance(payload, dict) and isinstance(payload.get("info"), dict) else {}
        currency = info.get("financialCurrency") or info.get("currency") or info.get("tradeableCurrency")
        if currency:
            return str(currency).upper()
    return _currency_hint_for_symbol(symbol)


def _compact_number(value: float, currency: str | None = None) -> str:
    absolute = abs(value)
    prefix = _CURRENCY_SYMBOLS.get(str(currency or "").upper(), "")
    suffix_currency = "" if prefix else (f"{str(currency).upper()} " if currency else "")
    if absolute >= 1_000_000_000_000:
        return f"{prefix}{suffix_currency}{value / 1_000_000_000_000:.2f}T"
    if absolute >= 1_000_000_000:
        return f"{prefix}{suffix_currency}{value / 1_000_000_000:.2f}B"
    if absolute >= 1_000_000:
        return f"{prefix}{suffix_currency}{value / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{prefix}{suffix_currency}{value:,.2f}"
    return f"{prefix}{suffix_currency}{value:.4g}"


def _format_money_value(value: object, currency: str | None) -> str:
    numeric = _coerce_numeric(value)
    if numeric is None:
        return str(value) if value not in (None, "") else "N/A"
    normalized_currency = str(currency or "").upper() or None
    if normalized_currency == "USD":
        return _compact_number(numeric, normalized_currency)
    if normalized_currency == "INR":
        absolute = abs(numeric)
        if absolute >= 1_000_000_000_000:
            return f"₹{numeric / 1_000_000_000_000:.2f} lakh crore"
        if absolute >= 10_000_000:
            return f"₹{numeric / 10_000_000:,.2f} crore"
        return f"₹{numeric:,.2f} (INR)"
    if normalized_currency:
        prefix = _CURRENCY_SYMBOLS.get(normalized_currency, "")
        if prefix:
            return f"{prefix}{numeric:,.2f} {normalized_currency}"
        return f"{normalized_currency} {numeric:,.2f}"
    return f"{numeric:,.2f} (currency unknown)"


def _format_metric_value(value: object, metric_key: str, currency: str | None) -> str:
    if metric_key == "dividend_yield":
        numeric = _coerce_numeric(value)
        if numeric is None:
            return str(value) if value not in (None, "") else "N/A"
        if 0 <= numeric <= 1:
            numeric *= 100
        return f"{numeric:.2f}%"
    if _is_money_metric(metric_key):
        return _format_money_value(value, currency)
    return _format_market_value(value)


def _statement_record_label(row: dict) -> str:
    for key in ["index", "Breakdown", "breakdown", "lineItem", "Line Item", "metric"]:
        if key in row:
            return _normalize_market_text(row.get(key))
    for value in row.values():
        if isinstance(value, str):
            return _normalize_market_text(value)
    return ""


def _extract_statement_metric(rows: list[dict], row_names: list[str]) -> tuple[object, str | None]:
    normalized_targets = {_normalize_market_text(name) for name in row_names}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        label = _statement_record_label(row)
        if label not in normalized_targets:
            continue
        for key, value in row.items():
            if _normalize_market_text(key) in {"index", "breakdown", "lineitem", "line item", "metric"}:
                continue
            numeric = _coerce_numeric(value)
            if numeric is not None:
                return numeric, str(key)
    return None, None


def _extract_market_metric(payloads: dict[str, dict], metric_key: str) -> tuple[object, str | None]:
    definition = MARKET_METRIC_DEFINITIONS.get(metric_key, {})
    data_type = str(definition.get("data_type") or "")
    payload = payloads.get(data_type) or {}
    if data_type == "info":
        info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
        for info_key in definition.get("info_keys", []):
            if info_key in info and info.get(info_key) not in (None, ""):
                return info.get(info_key), info_key
        return None, None
    if data_type in {"financials", "balance_sheet", "cashflow"}:
        rows = payload.get(data_type) if isinstance(payload.get(data_type), list) else []
        return _extract_statement_metric(rows, list(definition.get("statement_rows", [])))
    if data_type == "history":
        rows = payload.get("history") if isinstance(payload.get("history"), list) else []
        for row in reversed(rows):
            if isinstance(row, dict) and _coerce_numeric(row.get("Close")) is not None:
                return row.get("Close"), str(row.get("Date") or "latest")
    return None, None


@tool
def get_market_data_bundle(
    symbols: list[str],
    request: str = "",
    metrics: list[str] | None = None,
    period: str = "1y",
    interval: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    option_expiration: str | None = None,
) -> str:
    """
    Infer required yfinance data classes for a multi-ticker request, fetch/cache them, and return comparable values when possible.

    Use this for broad or follow-up requests such as company data, fundamentals,
    revenue/net-income comparison, valuation values, dividends, holders,
    recommendations, options, or "all available data".
    """
    normalized_symbols = sorted({str(symbol or "").strip().upper() for symbol in symbols or [] if str(symbol or "").strip()})
    if not normalized_symbols:
        return "Unable to fetch market data bundle: no ticker symbols were provided."

    data_types, inferred_metrics = _infer_market_data_requirements(request=request, metrics=metrics)
    requested_metrics = list(dict.fromkeys([*(metrics or []), *inferred_metrics]))
    resolved_metric_keys = [
        metric for metric in requested_metrics if metric in MARKET_METRIC_DEFINITIONS
    ]
    if not resolved_metric_keys:
        for item in requested_metrics:
            normalized = _normalize_market_text(item)
            match = next(
                (
                    key
                    for key, definition in MARKET_METRIC_DEFINITIONS.items()
                    if normalized == key or any(normalized == _normalize_market_text(alias) for alias in definition.get("aliases", []))
                ),
                None,
            )
            if match and match not in resolved_metric_keys:
                resolved_metric_keys.append(match)

    if any(_is_money_metric(metric_key) for metric_key in resolved_metric_keys) and "info" not in data_types:
        data_types = sorted({*data_types, "info"})

    payloads_by_symbol: dict[str, dict[str, dict]] = {}
    cache_hits: list[str] = []
    fetch_failures: list[str] = []
    for symbol in normalized_symbols:
        payloads_by_symbol[symbol] = {}
        for data_type in data_types:
            payload, from_cache = _fetch_cached_yfinance_market_payload(
                symbol=symbol,
                data_type=data_type,
                period=period,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                option_expiration=option_expiration,
            )
            if payload:
                payloads_by_symbol[symbol][data_type] = payload
                if from_cache:
                    cache_hits.append(f"{symbol}:{data_type}")
            else:
                fetch_failures.append(f"{symbol}:{data_type}")

    lines = [
        "Market Data Bundle",
        f"- Symbols: {', '.join(normalized_symbols)}",
        f"- Data fetched: {', '.join(data_types)}",
        "- Source: yfinance with backend cache",
    ]
    currencies_by_symbol = {
        symbol: _currency_from_payloads(symbol, payloads_by_symbol.get(symbol, {}))
        for symbol in normalized_symbols
    }
    known_currencies = sorted({currency for currency in currencies_by_symbol.values() if currency})
    if any(_is_money_metric(metric_key) for metric_key in resolved_metric_keys):
        currency_parts = [f"{symbol}={currencies_by_symbol.get(symbol) or 'unknown'}" for symbol in normalized_symbols]
        lines.append(f"- Currency basis: {', '.join(currency_parts)}")
        if len(known_currencies) > 1:
            lines.append("- Warning: monetary values use mixed currencies. Compare only after FX conversion or normalization.")
    if cache_hits:
        lines.append(f"- Cache hits: {', '.join(cache_hits[:12])}{' ...' if len(cache_hits) > 12 else ''}")
    if fetch_failures:
        lines.append(f"- Unavailable payloads: {', '.join(fetch_failures[:12])}{' ...' if len(fetch_failures) > 12 else ''}")

    if resolved_metric_keys:
        header = ["Metric", *normalized_symbols]
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for metric_key in resolved_metric_keys:
            definition = MARKET_METRIC_DEFINITIONS[metric_key]
            row = [str(definition["label"])]
            for symbol in normalized_symbols:
                value, period_label = _extract_market_metric(payloads_by_symbol.get(symbol, {}), metric_key)
                formatted = _format_metric_value(value, metric_key, currencies_by_symbol.get(symbol))
                if period_label and formatted != "N/A":
                    formatted = f"{formatted} ({period_label})"
                row.append(formatted)
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("")
        lines.append("Comparable numeric metrics were not explicitly requested, so the required payloads were fetched and cached for the next analysis step.")

    return "\n".join(lines)


def _normalize_percent_like_value(value):
    try:
        if value in (None, "", "N/A"):
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if 0.0 <= numeric <= 1.0:
        return round(numeric * 100.0, 6)
    return round(numeric, 6)


def _history_frame_to_records(history: pd.DataFrame) -> list[dict]:
    if history is None or history.empty or "Close" not in history.columns:
        return []

    frame = history.reset_index().copy()
    if "Date" not in frame.columns:
        index_name = history.index.name or "Date"
        if index_name in frame.columns:
            frame = frame.rename(columns={index_name: "Date"})

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date")
    return [
        {"Date": row["Date"].strftime("%Y-%m-%d"), "Close": round(float(row["Close"]), 6)}
        for _, row in frame.iterrows()
    ]


@tool
def get_yfinance_market_data(
    symbol: str,
    data_type: str = "history",
    period: str = "1y",
    interval: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    option_expiration: str | None = None,
) -> str:
    """
    Fetch any available Yahoo Finance/yfinance data for a ticker, cache it in Supabase/Mongo, and return a compact summary.

    data_type supports: history, info, financials, balance_sheet, cashflow, dividends,
    splits, holders, recommendations, options, all.
    """
    normalized_symbol = str(symbol or "").strip().upper()
    normalized_type = _normalize_yfinance_data_type(data_type)
    if not normalized_symbol:
        return "Unable to fetch market data: no ticker symbol was provided."

    params = {"option_expiration": option_expiration or ""}
    cache_key = memory_manager.compute_market_data_cache_key(
        symbol=normalized_symbol,
        data_type=normalized_type,
        period=period,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        params=params,
    )
    cached = memory_manager.retrieve_market_data_cache(cache_key)
    if cached and isinstance(cached.get("payload"), dict):
        return _market_payload_summary(cached["payload"], from_cache=True)

    payload = _fetch_yfinance_market_payload(
        symbol=normalized_symbol,
        data_type=normalized_type,
        period=period,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        option_expiration=option_expiration,
    )
    if not payload:
        return (
            f"Unable to fetch {normalized_type} data for {normalized_symbol} from yfinance. "
            "The ticker may be unavailable, delisted, or the requested data type may not be exposed by Yahoo Finance."
        )

    ttl_hours = 1 if normalized_type == "options" else 24
    stored = memory_manager.store_market_data_cache(
        cache_key=cache_key,
        symbol=normalized_symbol,
        data_type=normalized_type,
        payload=payload,
        period=period,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        source="yfinance",
        ttl_hours=ttl_hours,
    )
    summary = _market_payload_summary(payload, from_cache=False)
    if stored:
        summary += "\n- Cache: stored for faster follow-up requests"
    else:
        summary += "\n- Cache: skipped because no backend cache store is currently available"
    return summary


def _cached_yfinance_history_frame(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> pd.DataFrame:
    normalized_ticker = str(ticker or "").strip().upper()
    if not normalized_ticker:
        return pd.DataFrame()

    cache_key = memory_manager.compute_market_data_cache_key(
        symbol=normalized_ticker,
        data_type="history",
        period="custom",
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        params={},
    )
    cached = memory_manager.retrieve_market_data_cache(cache_key)
    payload = cached.get("payload") if isinstance(cached, dict) else None
    if not isinstance(payload, dict):
        payload = _fetch_yfinance_market_payload(
            symbol=normalized_ticker,
            data_type="history",
            period="custom",
            interval=interval,
            start_date=start_date,
            end_date=end_date,
        )
        if isinstance(payload, dict):
            memory_manager.store_market_data_cache(
                cache_key=cache_key,
                symbol=normalized_ticker,
                data_type="history",
                payload=payload,
                period="custom",
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                source="yfinance",
                ttl_hours=24,
            )

    rows = (payload or {}).get("history")
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()

    date_col = next((col for col in frame.columns if str(col).lower() in {"date", "datetime"}), frame.columns[0])
    rename_map = {}
    for column in frame.columns:
        lower = str(column).lower()
        if lower == "open":
            rename_map[column] = "Open"
        elif lower == "high":
            rename_map[column] = "High"
        elif lower == "low":
            rename_map[column] = "Low"
        elif lower == "close":
            rename_map[column] = "Close"
        elif lower == "volume":
            rename_map[column] = "Volume"
    frame = frame.rename(columns=rename_map)
    frame["Date"] = pd.to_datetime(frame[date_col], errors="coerce", utc=True).dt.tz_localize(None)
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "Close" not in frame.columns:
        return pd.DataFrame()
    frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date")
    return frame[[column for column in ["Date", "Open", "High", "Low", "Close", "Volume"] if column in frame.columns]]


def _fetch_yfinance_snapshot_doc(ticker: str) -> Optional[dict]:
    yf = _get_yfinance_module()
    if yf is None:
        return None

    normalized_ticker = str(ticker or "").strip().upper()
    if not normalized_ticker:
        return None

    try:
        ticker_obj = yf.Ticker(normalized_ticker)
        info = _get_yfinance_info(ticker_obj)
        history = ticker_obj.history(period="max", auto_adjust=False)
    except Exception as exc:
        logger.warning("yfinance snapshot fallback failed for %s: %s", normalized_ticker, exc)
        return None

    if history is None or history.empty:
        return None

    key_stats = {
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "profit_margin": info.get("profitMargins"),
        "return_on_equity": info.get("returnOnEquity"),
        "dividend_yield": _normalize_percent_like_value(info.get("dividendYield")),
        "beta": info.get("beta"),
    }
    key_stats = {key: value for key, value in key_stats.items() if value not in (None, "", "N/A")}

    info_payload = {
        "company_name": info.get("longName") or info.get("shortName") or normalized_ticker,
        "shortName": info.get("shortName") or info.get("longName") or normalized_ticker,
        "longName": info.get("longName") or info.get("shortName") or normalized_ticker,
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "country": info.get("country", "Unknown"),
        "website": info.get("website", "N/A"),
        "summary": info.get("longBusinessSummary") or info.get("longSummary") or "",
    }

    return {
        "ticker": normalized_ticker,
        "shortName": info.get("shortName") or normalized_ticker,
        "longName": info.get("longName") or normalized_ticker,
        "universes": [],
        "historical_prices": _history_frame_to_records(history),
        "info": info_payload,
        "key_stats": key_stats,
        "financials": {},
        "graph_relationships": {},
        "analysis_and_estimates": {},
        "_source": "yfinance_fallback",
    }


def _fetch_yfinance_price_on_or_before(ticker: str, target_dt: pd.Timestamp) -> Optional[dict]:
    yf = _get_yfinance_module()
    if yf is None:
        return None

    normalized_ticker = str(ticker or "").strip().upper()
    if not normalized_ticker:
        return None

    try:
        history = yf.Ticker(normalized_ticker).history(
            start="1900-01-01",
            end=(target_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
        )
    except Exception as exc:
        logger.warning("yfinance price fallback failed for %s: %s", normalized_ticker, exc)
        return None

    records = _history_frame_to_records(history)
    if not records:
        return None

    frame = pd.DataFrame(records)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date")
    row = _get_effective_price_on_or_before(frame, target_dt)
    if row is None:
        return None

    return {
        "ticker": normalized_ticker,
        "close": round(float(row["Close"]), 6),
        "date": row["Date"].strftime("%Y-%m-%d"),
        "source": "yfinance fallback",
    }


def _format_stock_record(doc: dict) -> str:
    ticker = str(doc.get("ticker") or doc.get("symbol") or "UNKNOWN").upper()
    info = doc.get("info", {}) if isinstance(doc.get("info"), dict) else {}
    key_stats = doc.get("key_stats", {}) if isinstance(doc.get("key_stats"), dict) else {}
    financials = doc.get("financials", {}) if isinstance(doc.get("financials"), dict) else {}
    graph = doc.get("graph_relationships", {}) if isinstance(doc.get("graph_relationships"), dict) else {}
    analysis = doc.get("analysis_and_estimates", {}) if isinstance(doc.get("analysis_and_estimates"), dict) else {}
    universes = doc.get("universes", []) if isinstance(doc.get("universes"), list) else []

    price_df = _extract_price_frame(doc)
    price_summary = []
    if not price_df.empty:
        latest = price_df.iloc[-1]
        earliest = price_df.iloc[0]
        price_summary = [
            f"- Historical price coverage: {earliest['Date'].strftime('%Y-%m-%d')} to {latest['Date'].strftime('%Y-%m-%d')}",
            f"- Most recent stored close: {latest['Close']:.2f} on {latest['Date'].strftime('%Y-%m-%d')}",
            f"- Historical observations stored: {len(price_df)}",
        ]

    company_name = (
        info.get("company_name")
        or info.get("shortName")
        or info.get("longName")
        or doc.get("shortName")
        or doc.get("longName")
        or "Unknown Company"
    )

    lines = [
        f"Ticker: {ticker}",
        f"- Company: {company_name}",
        f"- Universes: {', '.join(universes) if universes else 'None stored'}",
        f"- Sector: {info.get('sector', 'Unknown')}",
        f"- Industry: {info.get('industry', 'Unknown')}",
        f"- Country: {info.get('country', 'Unknown')}",
        f"- Website: {info.get('website', 'N/A')}",
    ]

    lines.extend(price_summary)

    if key_stats:
        lines.append("- Key stats:")
        lines.extend(_summarize_metrics(key_stats))

    if financials:
        income_annual = len(financials.get("income_statement", {}).get("annual", []))
        income_quarterly = len(financials.get("income_statement", {}).get("quarterly", []))
        balance_annual = len(financials.get("balance_sheet", {}).get("annual", []))
        cashflow_annual = len(financials.get("cashflow", {}).get("annual", []))
        lines.extend(
            [
                "- Financial statement coverage:",
                f"  - Income statement periods: annual={income_annual}, quarterly={income_quarterly}",
                f"  - Balance sheet annual periods: {balance_annual}",
                f"  - Cash flow annual periods: {cashflow_annual}",
            ]
        )

    if graph:
        lines.extend(
            [
                "- Graph and ownership data:",
                f"  - Dividends stored: {len(graph.get('dividends', []))}",
                f"  - Splits stored: {len(graph.get('splits', []))}",
                f"  - Institutional holders stored: {len(graph.get('institutional_holders', []))}",
                f"  - Insider roster entries stored: {len(graph.get('insider_roster', []))}",
                f"  - Insider transactions stored: {len(graph.get('insider_transactions', []))}",
            ]
        )

    if analysis:
        lines.extend(
            [
                "- Analyst and estimates data:",
                f"  - Recommendations stored: {len(analysis.get('recommendations', []))}",
                f"  - Earnings estimate rows stored: {len(analysis.get('earnings_estimate', []))}",
            ]
        )

    summary = info.get("summary")
    if summary:
        lines.append(f"- Business summary: {summary[:500]}{'...' if len(summary) > 500 else ''}")

    source_label = doc.get("_source")
    if source_label == "yfinance_fallback":
        lines.append("- Data source: yfinance fallback")

    return "\n".join(lines)


@tool
def list_available_sectors() -> str:
    """List distinct sectors available in the MongoDB stock database."""
    try:
        cache_key = _lookup_cache_key("list_available_sectors")
        cached = _get_lookup_cache(cache_key)
        if cached:
            return cached

        collection = _get_collection()
        sectors = {
            sector.strip()
            for sector in collection.distinct("sector")
            if isinstance(sector, str) and sector.strip()
        }
        sectors.update(
            sector.strip()
            for sector in collection.distinct("info.sector")
            if isinstance(sector, str) and sector.strip()
        )

        if not sectors:
            return "No sectors were found in the MongoDB database."

        sector_list = sorted(sectors)
        lines = ["Here are the available sectors found in the database:"]
        lines.extend(f"- {sector}" for sector in sector_list)
        return _set_lookup_cache(cache_key, "\n".join(lines))

    except Exception as e:
        return f"Unable to list available sectors due to a database or query error: {type(e).__name__}: {str(e)}"


@tool
def list_available_universes() -> str:
    """List distinct universes available in the MongoDB stock database, with ticker counts and sector mix."""
    try:
        cache_key = _lookup_cache_key("list_available_universes")
        cached = _get_lookup_cache(cache_key)
        if cached:
            return cached

        collection = _get_collection()
        docs = list(collection.find({}, {"universes": 1, "sector": 1, "info.sector": 1}))

        universe_counts = {}
        universe_sector_counts = {}

        for doc in docs:
            info = doc.get("info", {}) if isinstance(doc.get("info"), dict) else {}
            sector = info.get("sector") or doc.get("sector") or "Unknown"
            universes = doc.get("universes", []) if isinstance(doc.get("universes"), list) else []

            for universe in {str(item).strip().upper() for item in universes if str(item).strip()}:
                universe_counts[universe] = universe_counts.get(universe, 0) + 1
                sector_map = universe_sector_counts.setdefault(universe, {})
                sector_map[sector] = sector_map.get(sector, 0) + 1

        if not universe_counts:
            return "No universes were found in the MongoDB database."

        universe_keys = sorted(
            universe_counts,
            key=lambda item: (int(item[1:]) if re.fullmatch(r"U\d+", item) else float("inf"), item),
        )

        lines = ["Here are the available universes found in the database:"]
        for universe in universe_keys:
            sector_map = universe_sector_counts.get(universe, {})
            dominant_sector = (
                max(sector_map.items(), key=lambda item: item[1])[0]
                if sector_map
                else "Unknown"
            )
            lines.append(
                f"- {universe}: {universe_counts[universe]} tickers, dominant sector {dominant_sector}"
            )

        return _set_lookup_cache(cache_key, "\n".join(lines))

    except Exception as e:
        return f"Unable to list available universes due to a database or query error: {type(e).__name__}: {str(e)}"


@tool
def get_stocks_by_universe(universe: str) -> str:
    """Fetch stocks from MongoDB that belong to a requested universe such as U1 or U7."""
    try:
        if not universe or not universe.strip():
            return "Unable to search by universe: no universe was provided."

        normalized_universe = universe.strip().upper()
        cache_key = _lookup_cache_key("get_stocks_by_universe", normalized_universe)
        cached = _get_lookup_cache(cache_key)
        if cached:
            return cached

        collection = _get_collection()
        docs = list(
            collection.find(
                {"universes": normalized_universe},
                {
                    "ticker": 1,
                    "symbol": 1,
                    "shortName": 1,
                    "longName": 1,
                    "info.company_name": 1,
                    "info.shortName": 1,
                    "info.longName": 1,
                    "info.sector": 1,
                },
            ).sort("ticker", 1)
        )

        if not docs:
            return f"No stocks matching the universe '{normalized_universe}' were found in the database."

        results = []
        seen = set()
        for doc in docs:
            info = doc.get("info", {}) if isinstance(doc.get("info"), dict) else {}
            ticker = str(doc.get("ticker") or doc.get("symbol") or "UNKNOWN").upper()
            company_name = (
                doc.get("shortName")
                or doc.get("longName")
                or info.get("company_name")
                or info.get("shortName")
                or info.get("longName")
                or "Unknown Company"
            )
            sector = info.get("sector", "Unknown")
            key = (ticker, company_name)
            if key in seen:
                continue
            seen.add(key)
            results.append((ticker, company_name, sector))

        results.sort(key=lambda item: item[0])
        lines = [f"Here are the stocks in universe {normalized_universe} found in the database:"]
        lines.extend(f"- {ticker}: {company_name} ({sector})" for ticker, company_name, sector in results)
        return _set_lookup_cache(cache_key, "\n".join(lines))

    except Exception as e:
        return f"Unable to search stocks by universe due to a database or query error: {type(e).__name__}: {str(e)}"


@tool
def plot_historical_prices(
    tickers: list[str],
    start_date: str = "2005-01-01",
    end_date: str = "2025-12-31",
    config: RunnableConfig = None,
) -> str:
    """Plot historical prices from local MongoDB for the requested tickers over a date range."""
    try:
        if not tickers:
            return "Unable to generate the historical price plot: no tickers were provided."

        cleaned_tickers = _normalize_tickers(tickers)
        if not cleaned_tickers:
            return "Unable to generate the historical price plot: no valid tickers were provided."

        start_dt = pd.to_datetime(start_date, format="%Y-%m-%d", errors="raise")
        end_dt = pd.to_datetime(end_date, format="%Y-%m-%d", errors="raise")
        if start_dt > end_dt:
            return "Unable to generate the historical price plot: start_date must be on or before end_date."

        docs = _find_price_documents_with_retry(
            cleaned_tickers,
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
        )

        docs_by_ticker = {str(doc.get("ticker", "")).upper(): doc for doc in docs}
        included = {}
        point_counts = {}
        excluded = []
        yfinance_used = []

        for ticker in cleaned_tickers:
            doc = docs_by_ticker.get(ticker)
            if not doc:
                df = _cached_yfinance_history_frame(
                    ticker,
                    start_dt.strftime("%Y-%m-%d"),
                    end_dt.strftime("%Y-%m-%d"),
                )
                if not df.empty:
                    yfinance_used.append(ticker)
            else:
                df = _extract_price_frame(doc)
            if df.empty:
                excluded.append(f"- {ticker}: no MongoDB or yfinance historical price series")
                continue

            filtered = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()
            if filtered.empty:
                excluded.append(
                    f"- {ticker}: no historical prices between {start_date} and {end_date}"
                )
                continue

            original_count = len(filtered)
            render_frame = _downsample_df(filtered, target_points=700)
            point_counts[ticker] = {
                "original": int(original_count),
                "rendered": int(len(render_frame)),
            }

            included[ticker] = [
                {
                    "date": row["Date"].strftime("%Y-%m-%d"),
                    "close": round(float(row["Close"]), 6),
                }
                for _, row in render_frame.iterrows()
            ]

        if not included:
            lines = [
                "Unable to generate the historical price plot because no requested tickers had usable data in the selected date range."
            ]
            if excluded:
                lines.extend(["", "Excluded tickers:"])
                lines.extend(excluded)
            return "\n".join(lines)

        # --- MUI-native interactive chart: build PlotSpec and store as list ---
        plot_title = f"Historical Price Comparison {start_date} to {end_date}"

        from src.agents.generate_dynamic_plot import PALETTE
        series = [
            {
                "name": ticker,
                "label": ticker,
                "color": PALETTE[i % len(PALETTE)],
                "data": [
                    {"x": row["date"], "y": row["close"]}
                    for row in rows
                ],
                "showMark": False,
                "connectNulls": True,
                "highlightScope": {"highlight": "series", "fade": "global"},
            }
            for i, (ticker, rows) in enumerate(included.items())
        ]
        total_rendered_points = sum(len(rows) for rows in included.values())
        spec = {
            "plot_type": "line",
            "title": plot_title,
            "x_label": "Date",
            "x_type": "time",
            "y_label": "Close Price (USD)",
            "series": series,
            "density": {
                "sampled": any(counts["rendered"] < counts["original"] for counts in point_counts.values()),
                "point_counts": point_counts,
                "rendered_points": int(total_rendered_points),
            },
            # ── MUI X Line Chart features (backend-decided) ──
            "grid": {"horizontal": True},
            "curve": "monotoneX",
            "highlightScope": {"highlight": "series", "fade": "global"},
            "experimentalFeatures": {"enablePositionBasedPointerInteraction": True},
            "skipAnimation": total_rendered_points > 500,
        }

        import uuid
        from src.memory.mongodb_memory_layer import MongoMemoryManager
        
        plot_id = str(uuid.uuid4())
        stored = False
        
        try:
            mongo = MongoMemoryManager()
            stored = bool(mongo.store_plot(plot_id, spec, ttl_days=365))
        except Exception as e:
            logger.error("Failed to store plot in MongoDB: %s", e)

        session_id = (
            config.get("configurable", {}).get("thread_id", "default")
            if config
            else "default"
        )
        from src.agents.plot_store import register_plot
        register_plot(plot_id, spec, session_id)
        if not stored:
            logger.warning(
                "plot_historical_prices: registered PlotSpec %s in process memory because persistent storage is unavailable",
                plot_id,
            )

        logger.info(
            "plot_historical_prices: stored PlotSpec with ID %s for session %s",
            plot_id,
            session_id,
        )

        coverage_lines = [
            (
                f"- {ticker}: {rows[0]['date']} to {rows[-1]['date']} "
                f"({point_counts.get(ticker, {}).get('original', len(rows))} observations, "
                f"{point_counts.get(ticker, {}).get('rendered', len(rows))} rendered)"
            )
            for ticker, rows in included.items()
            if rows
        ]
        response = [
            "Historical Price Plot",
            f"- Date range: {start_date} to {end_date}",
            f"- Included tickers: {', '.join(included.keys())}",
            "- Coverage used:",
            *coverage_lines,
            "",
            "Plot generated successfully: interactive chart will appear in the panel.",
        ]

        if excluded:
            response.extend(["", "Excluded tickers:"])
            response.extend(excluded)

        if yfinance_used:
            response.extend(["", "Source note:"])
            response.append(f"- yfinance used and cached for: {', '.join(yfinance_used)}")

        if len(included) > 12:
            response.extend(
                [
                    "",
                    "Note: this chart contains many tickers, so visual overlap can make it dense.",
                ]
            )

        return "\n".join(response)

    except Exception as e:
        return (
            "Unable to generate the historical price plot due to an internal error: "
            f"{type(e).__name__}: {str(e)}"
        )


@tool
def get_universe_overview(universe: str) -> str:
    """Summarize a universe with its tickers, sector mix, and dominant sector from MongoDB."""
    try:
        if not universe or not universe.strip():
            return "Unable to summarize universe: no universe was provided."

        normalized_universe = universe.strip().upper()
        cache_key = _lookup_cache_key("get_universe_overview", normalized_universe)
        cached = _get_lookup_cache(cache_key)
        if cached:
            return cached

        collection = _get_collection()
        docs = list(
            collection.find(
                {"universes": normalized_universe},
                {
                    "ticker": 1,
                    "info.company_name": 1,
                    "info.shortName": 1,
                    "info.longName": 1,
                    "info.sector": 1,
                    "sector": 1,
                },
            ).sort("ticker", 1)
        )

        if not docs:
            return f"No stocks matching the universe '{normalized_universe}' were found in the database."

        rows = []
        sector_counts = {}
        for doc in docs:
            info = doc.get("info", {}) if isinstance(doc.get("info"), dict) else {}
            ticker = str(doc.get("ticker", "UNKNOWN")).upper()
            company_name = (
                info.get("company_name")
                or info.get("shortName")
                or info.get("longName")
                or "Unknown Company"
            )
            sector = info.get("sector") or doc.get("sector") or "Unknown"
            rows.append((ticker, company_name, sector))
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        rows.sort(key=lambda item: item[0])
        dominant_sector = max(sector_counts.items(), key=lambda item: item[1])[0] if sector_counts else "Unknown"

        lines = [
            f"Universe {normalized_universe} Overview",
            f"- Total tickers: {len(rows)}",
            f"- Dominant sector: {dominant_sector}",
            "- Sector breakdown:",
        ]
        for sector, count in sorted(sector_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"  - {sector}: {count}")

        lines.append("- Constituents:")
        lines.extend(f"  - {ticker}: {company_name} ({sector})" for ticker, company_name, sector in rows)
        return _set_lookup_cache(cache_key, "\n".join(lines))

    except Exception as e:
        return f"Unable to summarize universe due to a database or query error: {type(e).__name__}: {str(e)}"


@tool
def get_stock_database_snapshot(tickers: list[str]) -> str:
    """Fetch a broad stock snapshot, using MongoDB first and yfinance as a labeled fallback when needed."""
    try:
        if not tickers:
            return "Unable to fetch stock database snapshot: no tickers were provided."

        cleaned_tickers = sorted({t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()})
        if not cleaned_tickers:
            return "Unable to fetch stock database snapshot: no valid tickers were provided."

        mongo_error = None
        try:
            docs = _find_documents_with_retry(
                {"ticker": {"$in": cleaned_tickers}},
                {
                    "ticker": 1,
                    "symbol": 1,
                    "shortName": 1,
                    "longName": 1,
                    "universes": 1,
                    "historical_prices.Date": 1,
                    "historical_prices.date": 1,
                    "info": 1,
                    "key_stats": 1,
                    "financials": 1,
                    "graph_relationships": 1,
                    "analysis_and_estimates": 1,
                },
            )
        except Exception as exc:
            docs = []
            mongo_error = exc
            logger.warning("MongoDB stock snapshot lookup failed; attempting yfinance fallback. Error: %s", exc)

        found = {str(doc.get('ticker', '')).upper(): doc for doc in docs}
        sections = []
        missing = []
        fallback_tickers = []

        for ticker in cleaned_tickers:
            doc = found.get(ticker)
            if not doc:
                fallback_doc = _fetch_yfinance_snapshot_doc(ticker)
                if fallback_doc is not None:
                    doc = fallback_doc
                    fallback_tickers.append(ticker)
                else:
                    if mongo_error is not None:
                        missing.append(
                            f"- {ticker}: MongoDB unavailable and yfinance fallback failed ({type(mongo_error).__name__})"
                        )
                    else:
                        missing.append(f"- {ticker}: ticker not found")
                    continue
            sections.append(_format_stock_record(doc))

        if not sections:
            if mongo_error is not None:
                return (
                    "Unable to fetch stock database snapshot due to a database or fallback error: "
                    f"{type(mongo_error).__name__}: {str(mongo_error)}"
                )
            return "Unable to fetch stock database snapshot: none of the requested tickers were found in MongoDB or via yfinance fallback."

        response = ["MongoDB Stock Snapshot", ""]
        if fallback_tickers:
            response = [
                "Stock Snapshot",
                "",
                "Source note: MongoDB primary lookup was supplemented by yfinance fallback for "
                + ", ".join(fallback_tickers)
                + ".",
                "",
            ]
        response.append("\n\n".join(sections))

        if missing:
            response.extend(["", "Unavailable tickers:"])
            response.extend(missing)

        return "\n".join(response)

    except Exception as e:
        return f"Unable to fetch stock database snapshot due to a database or query error: {type(e).__name__}: {str(e)}"


@tool
def get_stocks_by_sector(sector: str) -> str:
    """Fetch stocks from MongoDB whose sector matches the requested sector."""
    try:
        if not sector or not sector.strip():
            return "Unable to search by sector: no sector was provided. Use the list_available_sectors tool if you need the sector names available in the database."

        normalized_sector = sector.strip()
        cache_key = _lookup_cache_key("get_stocks_by_sector", normalized_sector)
        cached = _get_lookup_cache(cache_key)
        if cached:
            return cached

        collection = _get_collection()
        pattern = re.escape(normalized_sector)
        query = {
            "$or": [
                {"sector": {"$regex": pattern, "$options": "i"}},
                {"info.sector": {"$regex": pattern, "$options": "i"}},
            ]
        }
        projection = {
            "ticker": 1,
            "symbol": 1,
            "shortName": 1,
            "longName": 1,
            "info.shortName": 1,
            "info.longName": 1,
            "info.company_name": 1,
            "sector": 1,
            "info.sector": 1,
        }

        docs = list(collection.find(query, projection).sort("ticker", 1))
        if not docs:
            return f"No stocks matching the sector '{sector}' were found in the database."

        results = []
        seen = set()

        for doc in docs:
            info = doc.get("info", {}) if isinstance(doc.get("info"), dict) else {}
            ticker = doc.get("ticker") or doc.get("symbol") or "UNKNOWN"
            company_name = (
                doc.get("shortName")
                or doc.get("longName")
                or info.get("shortName")
                or info.get("longName")
                or info.get("company_name")
                or "Unknown Company"
            )

            key = (str(ticker).upper(), str(company_name))
            if key in seen:
                continue
            seen.add(key)
            results.append((str(ticker).upper(), str(company_name)))

        results.sort(key=lambda item: item[0])
        lines = [f"Here are the stocks in the {normalized_sector} sector found in the database:"]
        lines.extend(f"- {ticker}: {company_name}" for ticker, company_name in results)
        return _set_lookup_cache(cache_key, "\n".join(lines))

    except Exception as e:
        return f"Unable to search stocks by sector due to a database or query error: {type(e).__name__}: {str(e)}"


@tool
def analyze_institutional_network(tickers: list[str]) -> str:
    """Analyze institutional-holder network centrality for the requested tickers using MongoDB data."""
    try:
        if not tickers:
            return "Unable to analyze institutional network: no tickers were provided."

        cleaned_tickers = _normalize_tickers(tickers)
        if not cleaned_tickers:
            return "Unable to analyze institutional network: no valid tickers were provided."

        collection = _get_collection()
        docs = list(
            collection.find(
                {"ticker": {"$in": cleaned_tickers}},
                {"ticker": 1, "graph_relationships.institutional_holders": 1},
            )
        )

        if not docs:
            return "Unable to analyze institutional network: none of the requested tickers were found in MongoDB."

        graph = nx.Graph()
        stock_nodes = []
        missing = []

        for ticker in cleaned_tickers:
            doc = next((item for item in docs if str(item.get("ticker", "")).upper() == ticker), None)
            if not doc:
                missing.append(f"{ticker} (ticker not found)")
                continue

            holders = doc.get("graph_relationships", {}).get("institutional_holders", [])
            if not holders:
                missing.append(f"{ticker} (no institutional holder data)")
                continue

            stock_nodes.append(ticker)
            graph.add_node(ticker, bipartite=0)

            for holder in holders:
                holder_name = holder.get("Holder")
                pct_str = str(holder.get("pctHeld", "0")).replace("%", "").strip()
                try:
                    weight = float(pct_str)
                except ValueError:
                    weight = 0.0

                if holder_name and weight > 0:
                    graph.add_node(holder_name, bipartite=1)
                    graph.add_edge(ticker, holder_name, weight=weight)

        if not stock_nodes:
            details = ", ".join(missing) if missing else "no eligible tickers"
            return f"Unable to analyze institutional network: {details}."

        try:
            centrality = nx.eigenvector_centrality(graph, max_iter=2000, weight="weight")
            method = "Eigenvector Centrality"
        except Exception:
            centrality = nx.degree_centrality(graph)
            method = "Degree Centrality fallback"

        stock_centrality = {node: score for node, score in centrality.items() if node in stock_nodes}
        if not stock_centrality:
            return "Unable to analyze institutional network: graph centrality could not be computed for the requested tickers."

        c_series = pd.Series(stock_centrality)
        c_min, c_max = c_series.min(), c_series.max()
        if c_max > c_min:
            normalized = (c_series - c_min) / (c_max - c_min)
        else:
            normalized = pd.Series(0.0, index=c_series.index)

        lines = [
            "Institutional Network Risk Analysis",
            f"Method used: {method}",
            "Normalized structural risk scores:",
        ]
        for ticker, score in normalized.sort_values(ascending=False).items():
            lines.append(f"- {ticker}: {score:.4f}")

        if missing:
            lines.append("")
            lines.append("Unavailable tickers:")
            lines.extend(f"- {item}" for item in missing)

        return "\n".join(lines)

    except Exception as e:
        return f"Unable to analyze institutional network due to a database or graph error: {type(e).__name__}: {str(e)}"


@tool
def get_historical_prices(tickers: list[str], target_date: str) -> str:
    """Fetch historical closing prices on or immediately prior to a target date, with yfinance fallback when MongoDB is unavailable."""
    try:
        if not tickers:
            return "Unable to fetch historical prices: no tickers were provided."

        cleaned_tickers = _normalize_tickers(tickers)
        if not cleaned_tickers:
            return "Unable to fetch historical prices: no valid tickers were provided."

        target_dt = pd.to_datetime(target_date, format="%Y-%m-%d", errors="raise")
        mongo_error = None
        try:
            docs = _find_price_documents_with_retry(
                cleaned_tickers,
                start_date="1900-01-01",
                end_date=target_dt.strftime("%Y-%m-%d"),
            )
        except Exception as exc:
            docs = []
            mongo_error = exc
            logger.warning("MongoDB historical price lookup failed; attempting yfinance fallback. Error: %s", exc)
        if not docs:
            fallback_lines = []
            fallback_missing = []
            for ticker in cleaned_tickers:
                fallback_row = _fetch_yfinance_price_on_or_before(ticker, target_dt)
                if fallback_row is not None:
                    fallback_lines.append(
                        f"- {ticker}: close={fallback_row['close']:.2f} on {fallback_row['date']} "
                        f"(source: {fallback_row['source']})"
                    )
                else:
                    fallback_missing.append(ticker)

            if fallback_lines:
                response = [
                    f"Historical closing prices on or immediately before {target_date}:",
                    *fallback_lines,
                    "",
                    "Source note:",
                    f"- yfinance fallback used for: {', '.join(ticker for ticker in cleaned_tickers if ticker not in fallback_missing)}",
                ]
                if fallback_missing:
                    response.extend(["", "Missing or unavailable:"])
                    response.extend(f"- {ticker}" for ticker in fallback_missing)
                return "\n".join(response)

            if mongo_error is not None:
                return (
                    f"Unable to fetch historical prices for {target_date}. MongoDB lookup failed with "
                    f"{type(mongo_error).__name__}: {str(mongo_error)} and yfinance fallback did not recover the requested tickers."
                )
            return f"Unable to fetch historical prices: none of the requested tickers were found in MongoDB for {target_date}."

        found = {doc.get("ticker", "").upper(): doc for doc in docs}
        lines = []
        missing = []
        fallback_used = []

        for ticker in cleaned_tickers:
            doc = found.get(ticker)
            if not doc:
                fallback_row = _fetch_yfinance_price_on_or_before(ticker, target_dt)
                if fallback_row is not None:
                    fallback_used.append(ticker)
                    lines.append(
                        f"- {ticker}: close={fallback_row['close']:.2f} on {fallback_row['date']} "
                        f"(source: {fallback_row['source']})"
                    )
                elif mongo_error is not None:
                    missing.append(f"{ticker} (MongoDB unavailable and yfinance fallback failed)")
                else:
                    missing.append(f"{ticker} (ticker not found)")
                continue

            df = _extract_price_frame(doc)
            if df.empty:
                fallback_row = _fetch_yfinance_price_on_or_before(ticker, target_dt)
                if fallback_row is not None:
                    fallback_used.append(ticker)
                    lines.append(
                        f"- {ticker}: close={fallback_row['close']:.2f} on {fallback_row['date']} "
                        f"(source: {fallback_row['source']})"
                    )
                else:
                    missing.append(f"{ticker} (no historical price series)")
                continue

            row = _get_effective_price_on_or_before(df, target_dt)
            if row is None:
                fallback_row = _fetch_yfinance_price_on_or_before(ticker, target_dt)
                if fallback_row is not None:
                    fallback_used.append(ticker)
                    lines.append(
                        f"- {ticker}: close={fallback_row['close']:.2f} on {fallback_row['date']} "
                        f"(source: {fallback_row['source']})"
                    )
                else:
                    missing.append(f"{ticker} (no price on or before {target_date})")
                continue

            lines.append(f"- {ticker}: close={row['Close']:.2f} on {row['Date'].strftime('%Y-%m-%d')} (source: MongoDB)")

        if not lines:
            if mongo_error is not None:
                return (
                    f"Unable to fetch historical prices for {target_date}. MongoDB lookup failed with "
                    f"{type(mongo_error).__name__}: {str(mongo_error)} and yfinance fallback did not recover the requested tickers."
                )
            return (
                f"Unable to fetch historical prices for {target_date}. "
                f"No requested tickers had usable data on or before that date. "
                f"Missing details: {', '.join(missing)}"
            )

        response = [
            f"Historical closing prices on or immediately before {target_date}:",
            *lines,
        ]

        if fallback_used:
            response.extend(
                [
                    "",
                    "Source note:",
                    f"- yfinance fallback used for: {', '.join(fallback_used)}",
                ]
            )

        if missing:
            response.append("")
            response.append("Missing or unavailable:")
            response.extend(f"- {item}" for item in missing)

        return "\n".join(response)

    except Exception as e:
        return f"Unable to fetch historical prices due to an internal error: {type(e).__name__}: {str(e)}"


@tool
def run_historical_cvar_optimization(
    tickers: list[str],
    target_date: str,
    risk_tolerance: str = "moderate",
) -> str:
    """Run a 95% long-only CVaR optimization using the 90 trading days before a historical target date."""
    try:
        if not tickers or len(tickers) < 2:
            return "Unable to run historical CVaR optimization: please provide at least two valid tickers."

        cleaned_tickers = _normalize_tickers(tickers)
        if len(cleaned_tickers) < 2:
            return "Unable to run historical CVaR optimization: please provide at least two valid tickers."

        target_dt = pd.to_datetime(target_date, format="%Y-%m-%d", errors="raise")
        docs = _find_price_documents_with_retry(
            cleaned_tickers,
            start_date=(target_dt - pd.Timedelta(days=540)).strftime("%Y-%m-%d"),
            end_date=target_dt.strftime("%Y-%m-%d"),
        )

        found = {str(doc.get("ticker", "")).upper(): doc for doc in docs}
        price_series = {}
        effective_dates = {}
        missing = []
        fallback_used = []

        for ticker in cleaned_tickers:
            doc = found.get(ticker)
            if doc:
                df = _extract_price_frame(doc)
            else:
                df = _cached_yfinance_history_frame(
                    ticker,
                    (target_dt - pd.Timedelta(days=540)).strftime("%Y-%m-%d"),
                    (target_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                )
                if not df.empty:
                    fallback_used.append(ticker)

            if df.empty:
                missing.append(f"{ticker} (no historical price series)")
                continue

            eligible = df[df["Date"] <= target_dt].copy()
            if eligible.empty:
                missing.append(f"{ticker} (no data on or before {target_date})")
                continue

            effective_dates[ticker] = eligible["Date"].iloc[-1].strftime("%Y-%m-%d")
            trailing_window = eligible.tail(90).copy()

            if len(trailing_window) < 20:
                missing.append(f"{ticker} (insufficient history before {target_date})")
                continue

            series = trailing_window.set_index("Date")["Close"].rename(ticker)
            price_series[ticker] = series

        if len(price_series) < 2:
            return (
                f"Unable to run historical CVaR optimization for {target_date}: fewer than two tickers had enough "
                f"usable trailing history. Missing details: {', '.join(missing) if missing else 'none'}"
            )

        prices = pd.concat(price_series.values(), axis=1).sort_index()
        prices = prices.ffill().dropna(how="any")

        if len(prices) < 20 or prices.shape[1] < 2:
            return (
                f"Unable to run historical CVaR optimization for {target_date}: not enough overlapping historical "
                f"prices across the requested tickers."
            )

        log_returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="any")
        if log_returns.empty or len(log_returns) < 20:
            return f"Unable to run historical CVaR optimization for {target_date}: insufficient clean return history."

        asset_names = list(log_returns.columns)
        returns_matrix = log_returns.to_numpy()
        num_periods, num_assets = returns_matrix.shape
        beta = 0.95

        mean_daily_returns = log_returns.mean().to_numpy()
        mean_annual_returns = mean_daily_returns * 252.0

        profile = (risk_tolerance or "moderate").strip().lower()
        if profile not in {"conservative", "moderate", "aggressive"}:
            profile = "moderate"

        percentile_map = {
            "conservative": 25,
            "moderate": 50,
            "aggressive": 75,
        }
        target_annual_return = float(np.percentile(mean_annual_returns, percentile_map[profile]))
        target_daily_return = target_annual_return / 252.0

        weights = cp.Variable(num_assets)
        alpha = cp.Variable()
        tail_excess = cp.Variable(num_periods, nonneg=True)

        portfolio_returns = returns_matrix @ weights
        losses = -portfolio_returns

        cvar_95 = alpha + (1.0 / ((1.0 - beta) * num_periods)) * cp.sum(tail_excess)

        constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            tail_excess >= losses - alpha,
            mean_daily_returns @ weights >= target_daily_return,
        ]

        problem = cp.Problem(cp.Minimize(cvar_95), constraints)

        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except Exception:
            problem.solve(solver=cp.SCS, verbose=False)

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weights.value is None:
            return (
                f"Historical CVaR optimization could not find a stable solution for {target_date}. "
                f"Solver status: {problem.status}."
            )

        optimal_weights = np.maximum(np.asarray(weights.value).reshape(-1), 0.0)
        optimal_weights[optimal_weights < 0.01] = 0.0

        if float(optimal_weights.sum()) <= 0:
            return f"Historical CVaR optimization failed for {target_date}: all weights were negligible after cleanup."

        optimal_weights = optimal_weights / optimal_weights.sum()
        expected_annualized_return = float(mean_annual_returns @ optimal_weights)

        realized_portfolio_returns = returns_matrix @ optimal_weights
        portfolio_losses = -realized_portfolio_returns
        var_95 = float(np.quantile(portfolio_losses, beta))
        tail_losses = portfolio_losses[portfolio_losses >= var_95]
        expected_cvar_95 = float(tail_losses.mean()) if len(tail_losses) > 0 else var_95

        allocation_lines = []
        for ticker, weight in sorted(zip(asset_names, optimal_weights), key=lambda item: item[1], reverse=True):
            if weight > 0:
                allocation_lines.append(f"- {ticker}: {weight * 100:.2f}%")

        response = [
            "Historical CVaR Optimization Result",
            f"Target date requested: {target_date}",
            f"Effective price window end: {prices.index.max().strftime('%Y-%m-%d')}",
            f"Risk tolerance: {profile.capitalize()}",
            f"Universe: {', '.join(asset_names)}",
            f"Target annual return floor used in optimization: {target_annual_return * 100:.2f}%",
            "",
            "Optimal allocation weights:",
            *allocation_lines,
            "",
            f"Estimated/backtested annualized portfolio return: {expected_annualized_return * 100:.2f}%",
            f"Expected 95% CVaR (daily tail risk): {expected_cvar_95 * 100:.2f}%",
            "",
            "Historical pricing dates used at or before the target date:",
            *[f"- {ticker}: {effective_dates[ticker]}" for ticker in asset_names if ticker in effective_dates],
        ]

        if missing:
            response.extend(["", "Excluded or unavailable tickers:"])
            response.extend(f"- {item}" for item in missing)

        if fallback_used:
            response.extend(["", "Source note:"])
            response.append(f"- yfinance used and cached for: {', '.join(fallback_used)}")

        return "\n".join(response)

    except Exception as e:
        return f"Unable to run historical CVaR optimization due to an internal error: {type(e).__name__}: {str(e)}"


@tool
def run_full_governance_pipeline(
    tickers: list[str],
    target_date: str,
    risk_tolerance: str = "moderate",
    previous_weights: Optional[dict[str, float]] = None,
    config: RunnableConfig = None,
) -> str:
    """
    Run the full deterministic governance pipeline against local MongoDB only:
    historical prices, institutional network analysis, and historical CVaR optimization.
    This tool is advisory only, never executes trades, generates plots inline,
    and returns a lightweight JSON payload to avoid LLM context bloat.
    """
    try:
        if not tickers:
            return json.dumps(
                _build_lightweight_governance_payload(
                    status="error_no_tickers_provided",
                    message="Unable to run full governance pipeline: no tickers were provided.",
                    target_date=target_date,
                    valid_tickers=[],
                    dropped_tickers=[],
                )
            )

        cleaned_tickers = _normalize_tickers(tickers)
        if not cleaned_tickers:
            return json.dumps(
                _build_lightweight_governance_payload(
                    status="error_no_valid_tickers_provided",
                    message="Unable to run full governance pipeline: no valid tickers were provided.",
                    target_date=target_date,
                    valid_tickers=[],
                    dropped_tickers=[],
                )
            )

        try:
            target_dt = pd.to_datetime(target_date, format="%Y-%m-%d", errors="raise")
        except Exception:
            return json.dumps(
                _build_lightweight_governance_payload(
                    status="error_invalid_target_date",
                    message="Unable to run full governance pipeline: target_date must use the YYYY-MM-DD format.",
                    target_date=target_date,
                    valid_tickers=[],
                    dropped_tickers=[],
                )
            )

        docs = _find_documents_with_retry(
            {"ticker": {"$in": cleaned_tickers}},
            {
                "ticker": 1,
                "historical_prices.Date": 1,
                "historical_prices.date": 1,
                "historical_prices.Close": 1,
                "historical_prices.close": 1,
                "graph_relationships.institutional_holders": 1,
            },
        )
        docs_by_ticker = {str(doc.get("ticker", "")).upper(): doc for doc in docs}
        price_frames = _build_price_frames(docs_by_ticker)
        prepared = _prepare_portfolio_inputs(
            docs_by_ticker=docs_by_ticker,
            price_frames=price_frames,
            cleaned_tickers=cleaned_tickers,
            target_dt=target_dt,
            target_date=target_date,
        )

        valid_tickers = prepared["valid_tickers"]
        dropped_tickers = prepared["dropped_tickers"]
        data_sources = prepared.get("data_sources", {})
        network_payload = _build_network_analysis_payload(docs_by_ticker, valid_tickers)
        lightweight_systemic_risk = {
            "method": network_payload.get("method", "Unavailable"),
            "scores": network_payload.get("scores", {}),
        }

        if len(valid_tickers) < 2:
            generated_plots = _generate_inline_governance_plots(
                target_date=target_date,
                weights={},
                network_payload=network_payload,
                config=config,
            )
            return json.dumps(
                _build_lightweight_governance_payload(
                    status="error_fewer_than_two_valid_tickers_after_history_validation",
                    message=(
                        "Unable to complete optimization because fewer than two requested tickers had valid "
                        f"historical coverage through {target_date}."
                    ),
                    target_date=target_date,
                    valid_tickers=valid_tickers,
                    dropped_tickers=dropped_tickers,
                    data_sources=data_sources,
                    systemic_risk=lightweight_systemic_risk,
                    optimization={},
                    generated_plots=generated_plots,
                )
            )

        optimization_payload = _build_optimization_payload(
            overlapping_prices=prepared["overlapping_prices"],
            effective_dates=prepared["effective_dates"],
            target_date=target_date,
            risk_tolerance=risk_tolerance,
            network_scores=network_payload.get("scores", {}),
            previous_weights=previous_weights,
        )

        optimization_succeeded = optimization_payload.get("status") == "success"
        generated_plots = _generate_inline_governance_plots(
            target_date=target_date,
            weights=optimization_payload.get("weights", {}) if optimization_succeeded else {},
            network_payload=network_payload,
            config=config,
        )

        if dropped_tickers:
            status = (
                "partial_success_some_requested_tickers_were_dropped_due_to_missing_data"
                if optimization_succeeded
                else "error_optimization_failed_some_requested_tickers_were_dropped_due_to_missing_data"
            )
        else:
            status = "success" if optimization_succeeded else "error_optimization_failed"

        if optimization_succeeded:
            message = "Full historical governance pipeline completed successfully."
        else:
            message = optimization_payload.get("message", "Governance pipeline completed with errors.")

        if dropped_tickers:
            message += " Some requested tickers were dropped due to missing or insufficient price history."
        if any(source == "yfinance" for source in data_sources.values()):
            message += " yfinance was used and cached for tickers not available in MongoDB."

        lightweight_optimization = {}
        if optimization_succeeded:
            lightweight_optimization = _lightweight_optimization_payload(optimization_payload)
            instability_index = float(optimization_payload.get("instability_index", 0.0))
            lambda_t = float(optimization_payload.get("lambda_t", 0.0))
            weights = optimization_payload.get("weights", {})
            regime_type = "crisis" if instability_index > 0.5 else "calm"
            memory_manager.store_regime_pattern(
                target_date=target_date,
                regime_type=regime_type,
                instability_index=instability_index,
                lambda_t=lambda_t,
                weights=weights if isinstance(weights, dict) else {},
            )

        return json.dumps(
            _build_lightweight_governance_payload(
                status=status,
                message=message,
                target_date=target_date,
                valid_tickers=valid_tickers,
                dropped_tickers=dropped_tickers,
                data_sources=data_sources,
                systemic_risk=lightweight_systemic_risk,
                optimization=lightweight_optimization,
                generated_plots=generated_plots,
            )
        )

    except Exception as e:
        return json.dumps(
            _build_lightweight_governance_payload(
                status=f"error_internal_governance_pipeline_failure_{type(e).__name__.lower()}",
                message=(
                    f"Unable to run full governance pipeline due to an internal error: "
                    f"{type(e).__name__}: {str(e)}"
                ),
                target_date=target_date,
                valid_tickers=[],
                dropped_tickers=[],
            )
        )


@tool
def plot_us_economic_indicators(config: RunnableConfig = None) -> str:
    """
    Generate and plot US Unemployment Rate vs GDP per capita with recession bands.
    Use this when the user asks to see the unemployment vs GDP plot, the recession bands plot,
    or the usaUnemploymentAndGdp dataset.
    """
    import pandas as pd
    import numpy as np

    quarters = pd.date_range(start="2000-01-01", end="2024-12-31", freq="QE")
    data = []
    
    anchors = {
        2000: (4.0, 36300),
        2001: (5.4, 37100),
        2002: (5.8, 38000),
        2003: (6.0, 39400),
        2004: (5.5, 41700),
        2005: (5.1, 44100),
        2006: (4.6, 46200),
        2007: (4.6, 47900),
        2008: (5.8, 48300),
        2009: (9.3, 47000),
        2010: (9.6, 48300),
        2011: (8.9, 49700),
        2012: (8.1, 51400),
        2013: (7.4, 52700),
        2014: (6.2, 54900),
        2015: (5.3, 56700),
        2016: (4.9, 57800),
        2017: (4.4, 60000),
        2018: (3.9, 62800),
        2019: (3.7, 65000),
        2020: (8.1, 63000),
        2021: (5.4, 70200),
        2022: (3.6, 76300),
        2023: (3.6, 81600),
        2024: (4.1, 85000)
    }

    for dt in quarters:
        yr = dt.year
        q = (dt.month - 1) // 3 + 1
        base_unemp, base_gdp = anchors.get(yr, (5.0, 50000))
        if q == 1:
            unemp = base_unemp
            gdp = base_gdp
        elif q == 2:
            unemp = base_unemp * 0.98 + 0.1
            gdp = base_gdp * 1.008
        elif q == 3:
            unemp = base_unemp * 1.02 - 0.05
            gdp = base_gdp * 1.015
        else:
            unemp = base_unemp * 0.95 + 0.2
            gdp = base_gdp * 1.025
        
        if yr == 2001 and q == 3:
            unemp = 5.0
            gdp = 37000
        elif yr == 2008 and q == 4:
            unemp = 6.9
            gdp = 48000
        elif yr == 2009 and q == 4:
            unemp = 9.9
            gdp = 47200
        elif yr == 2020 and q == 2:
            unemp = 13.0
            gdp = 59000
            
        data.append({
            "date": dt.strftime("%Y-%m-%d"),
            "unemploymentRate": round(unemp, 2),
            "gdpPerCapita": round(gdp, 2)
        })

    recessions = [
      {
        "start": "2001-03-01",
        "end": "2001-11-01",
        "label": "Early 2000s",
      },
      {
        "start": "2007-12-01",
        "end": "2009-06-01",
        "label": "Great Recession",
      },
      { 
        "start": "2020-02-01", 
        "end": "2020-04-01", 
        "label": "COVID-19" 
      },
    ]

    spec = {
        "plot_type": "line",
        "title": "US unemployment rate comparison with GDP per capita",
        "x_label": "Date",
        "x_type": "time",
        "yAxis": [
          {
            "id": "unemployment-axis",
            "label": "Unemployment Rate",
            "position": "left",
            "value_format": "percent",
          },
          {
            "id": "gdp-axis",
            "label": "GDP per capita in US$",
            "position": "right",
            "value_format": "k",
          },
        ],
        "series": [
          {
            "name": "Unemployment rate",
            "color": "#af3838",
            "yAxisId": "unemployment-axis",
            "value_format": "percent",
            "data": [{"x": item["date"], "y": item["unemploymentRate"]} for item in data]
          },
          {
            "name": "GDP per capita",
            "color": "#4caf50",
            "yAxisId": "gdp-axis",
            "value_format": "k",
            "data": [{"x": item["date"], "y": item["gdpPerCapita"]} for item in data]
          }
        ],
        "recessions": recessions
    }

    import uuid
    from src.memory.mongodb_memory_layer import MongoMemoryManager
    
    plot_id = str(uuid.uuid4())
    stored = False
    
    try:
        mongo = MongoMemoryManager()
        stored = bool(mongo.store_plot(plot_id, spec, ttl_days=365))
    except Exception as e:
        logger.error("Failed to store plot in MongoDB: %s", e)

    session_id = (
        config.get("configurable", {}).get("thread_id", "default")
        if config
        else "default"
    )
    from src.agents.plot_store import register_plot
    register_plot(plot_id, spec, session_id)
    if not stored:
        logger.warning(
            "Registered macro comparison PlotSpec %s in process memory because persistent storage is unavailable",
            plot_id,
        )

    return "Chart ready: US unemployment rate comparison with GDP per capita"

