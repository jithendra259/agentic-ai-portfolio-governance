import json
import logging
import os
import re
import time
import warnings
from functools import lru_cache
from typing import Optional

import cvxpy as cp
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
from langchain_core.tools import tool
from pymongo import MongoClient
from pymongo.errors import AutoReconnect, NetworkTimeout, PyMongoError
from src.agents.generate_dynamic_plot import generate_financial_plot
from src.memory.mongodb_memory_layer import MongoMemoryManager

load_dotenv()


MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "Stock_data"
COLLECTION_NAME = "ticker"
logger = logging.getLogger(__name__)
memory_manager = MongoMemoryManager()
memory_manager.setup_indexes()
LOOKUP_CACHE_TTL_SECONDS = 300
_LOOKUP_CACHE = {}


@lru_cache(maxsize=1)
def _get_client():
    if not MONGO_URI:
        raise ValueError("MONGO_URI is not set in the environment.")

    return MongoClient(
        MONGO_URI,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=15000,
        connectTimeoutMS=20000,
        socketTimeoutMS=60000,
        maxPoolSize=20,
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


def _extract_price_frame(doc: dict) -> pd.DataFrame:
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

    df = df[[date_col, close_col]].rename(columns={date_col: "Date", close_col: "Close"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
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

    for ticker in cleaned_tickers:
        doc = docs_by_ticker.get(ticker)
        if not doc:
            _warn_drop_ticker(ticker, "ticker not found in MongoDB", dropped_tickers)
            continue

        df = price_frames.get(ticker, pd.DataFrame())
        if df.empty:
            _warn_drop_ticker(ticker, "no historical price series stored", dropped_tickers)
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


def _build_optimization_payload(
    overlapping_prices: pd.DataFrame,
    effective_dates: dict[str, str],
    target_date: str,
    risk_tolerance: str = "moderate",
    network_scores: Optional[dict[str, float]] = None,
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
    target_daily_return = target_annual_return / 252.0

    weights = cp.Variable(num_assets)
    alpha = cp.Variable()
    tail_excess = cp.Variable(num_periods, nonneg=True)

    portfolio_returns = returns_matrix @ weights
    losses = -portfolio_returns
    cvar_95 = alpha + (1.0 / ((1.0 - beta) * num_periods)) * cp.sum(tail_excess)
    lambda_t = float(lambda_max / (1.0 + np.exp(-k * (instability_index - i_thresh))))
    graph_penalty = lambda_t * (c_vector @ weights)

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= 0.15,
        tail_excess >= losses - alpha,
        mean_daily_returns @ weights >= target_daily_return,
    ]

    problem = cp.Problem(cp.Minimize(cvar_95 + graph_penalty), constraints)

    for solver in [cp.CLARABEL, cp.OSQP, cp.SCS]:
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                break
        except Exception:
            continue

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weights.value is None:
        return {
            "status": "error",
            "message": (
                f"Historical CVaR optimization could not find a stable solution for {target_date}. "
                f"Solver status: {problem.status}."
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

    return {
        "status": "success",
        "optimization_type": "graph_regularized_cvar",
        "risk_tolerance": profile,
        "weights": weights_map,
        "expected_annualized_return": round(expected_annualized_return, 6),
        "expected_cvar_95": round(expected_cvar_95, 6),
        "target_annual_return_floor": round(target_annual_return, 6),
        "instability_index": round(instability_index, 6),
        "lambda_t": round(lambda_t, 6),
        "graph_scores_used": {
            ticker: round(float(score), 6)
            for ticker, score in zip(asset_names, c_vector)
        },
        "objective_value": round(float(problem.value), 6) if problem.value is not None else None,
        "effective_window_end": overlapping_prices.index.max().strftime("%Y-%m-%d"),
        "historical_pricing_dates": {
            ticker: effective_dates[ticker]
            for ticker in asset_names
            if ticker in effective_dates
        },
        "correlation_matrix": correlation_matrix.to_dict(),
        "covariance_matrix": covariance_matrix.to_dict(),
    }


def _generate_inline_governance_plots(
    target_date: str,
    weights: dict[str, float],
    network_payload: dict,
) -> list[str]:
    generated_plots = []
    plot_requests = []
    risk_scores = network_payload.get("scores", {}) if isinstance(network_payload, dict) else {}
    holder_edges = network_payload.get("holder_edges", []) if isinstance(network_payload, dict) else []

    if weights:
        plot_requests.append(
            {
                "plot_type": "pie",
                "title": f"Optimal Allocation Weights as of {target_date}",
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
            plot_output = generate_financial_plot.invoke(request)
        except Exception as exc:
            logger.warning("Unable to generate %s plot inline: %s", request["plot_type"], exc)
            continue

        if isinstance(plot_output, str) and "![" in plot_output:
            generated_plots.append(plot_output)
            continue

        logger.warning(
            "Plot tool returned a non-markdown response for %s: %s",
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
) -> dict:
    return {
        "status": status,
        "message": message,
        "target_date": str(target_date or ""),
        "valid_tickers": valid_tickers or [],
        "dropped_tickers": dropped_tickers or [],
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
        f"Expected annualized portfolio return: {expected_annualized_return * 100:.2f}%",
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

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date")
    return [
        {"Date": row["Date"].strftime("%Y-%m-%d"), "Close": round(float(row["Close"]), 6)}
        for _, row in frame.iterrows()
    ]


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
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
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

        docs = _find_documents_with_retry(
            {"ticker": {"$in": cleaned_tickers}},
            {
                "ticker": 1,
                "historical_prices.Date": 1,
                "historical_prices.date": 1,
                "historical_prices.Close": 1,
                "historical_prices.close": 1,
            },
        )
        if not docs:
            return (
                "Unable to generate the historical price plot: none of the requested tickers were found "
                "in MongoDB."
            )

        docs_by_ticker = {str(doc.get("ticker", "")).upper(): doc for doc in docs}
        included = {}
        excluded = []

        for ticker in cleaned_tickers:
            doc = docs_by_ticker.get(ticker)
            if not doc:
                excluded.append(f"- {ticker}: ticker not found in MongoDB")
                continue

            df = _extract_price_frame(doc)
            if df.empty:
                excluded.append(f"- {ticker}: no stored historical price series")
                continue

            filtered = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()
            if filtered.empty:
                excluded.append(
                    f"- {ticker}: no historical prices between {start_date} and {end_date}"
                )
                continue

            included[ticker] = [
                {
                    "date": row["Date"].strftime("%Y-%m-%d"),
                    "close": round(float(row["Close"]), 6),
                }
                for _, row in filtered.iterrows()
            ]

        if not included:
            lines = [
                "Unable to generate the historical price plot because no requested tickers had usable data in the selected date range."
            ]
            if excluded:
                lines.extend(["", "Excluded tickers:"])
                lines.extend(excluded)
            return "\n".join(lines)

        plot_output = generate_financial_plot.invoke(
            {
                "data": {"price_history": included},
                "plot_type": "line",
                "title": f"Historical Price Comparison {start_date} to {end_date}",
            }
        )

        coverage_lines = [
            f"- {ticker}: {rows[0]['date']} to {rows[-1]['date']} ({len(rows)} observations)"
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
            plot_output,
        ]

        if excluded:
            response.extend(["", "Excluded tickers:"])
            response.extend(excluded)

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
        docs = _find_documents_with_retry(
            {"ticker": {"$in": cleaned_tickers}},
            {
                "ticker": 1, 
                "historical_prices.Date": 1, 
                "historical_prices.date": 1, 
                "historical_prices.Close": 1, 
                "historical_prices.close": 1
            },
        )
        if not docs:
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
        docs = _find_documents_with_retry(
            {"ticker": {"$in": cleaned_tickers}},
            {
                "ticker": 1, 
                "historical_prices.Date": 1, 
                "historical_prices.date": 1, 
                "historical_prices.Close": 1, 
                "historical_prices.close": 1
            },
        )
        if not docs:
            return f"Unable to run historical CVaR optimization: none of the requested tickers were found in MongoDB for {target_date}."

        price_series = {}
        effective_dates = {}
        missing = []

        for doc in docs:
            ticker = doc.get("ticker", "").upper()
            df = _extract_price_frame(doc)

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
            f"Expected annualized portfolio return: {expected_annualized_return * 100:.2f}%",
            f"Expected 95% CVaR (daily tail risk): {expected_cvar_95 * 100:.2f}%",
            "",
            "Historical pricing dates used at or before the target date:",
            *[f"- {ticker}: {effective_dates[ticker]}" for ticker in asset_names if ticker in effective_dates],
        ]

        if missing:
            response.extend(["", "Excluded or unavailable tickers:"])
            response.extend(f"- {item}" for item in missing)

        return "\n".join(response)

    except Exception as e:
        return f"Unable to run historical CVaR optimization due to an internal error: {type(e).__name__}: {str(e)}"


@tool
def run_full_governance_pipeline(
    tickers: list[str],
    target_date: str,
    risk_tolerance: str = "moderate",
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
        if not docs:
            return json.dumps(
                _build_lightweight_governance_payload(
                    status="error_no_requested_tickers_found_in_local_mongodb",
                    message=(
                        "Unable to run full governance pipeline: none of the requested tickers "
                        f"were found in local MongoDB for {target_date}."
                    ),
                    target_date=target_date,
                    valid_tickers=[],
                    dropped_tickers=[
                        {"ticker": ticker, "reason": "ticker not found in MongoDB"}
                        for ticker in cleaned_tickers
                    ],
                )
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
        )

        optimization_succeeded = optimization_payload.get("status") == "success"
        generated_plots = _generate_inline_governance_plots(
            target_date=target_date,
            weights=optimization_payload.get("weights", {}) if optimization_succeeded else {},
            network_payload=network_payload,
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
            message += " Some requested tickers were dropped due to missing or insufficient local history."

        lightweight_optimization = {}
        if optimization_succeeded:
            lightweight_optimization = {
                "weights": optimization_payload.get("weights", {}),
                "expected_annualized_return": optimization_payload.get("expected_annualized_return"),
                "expected_cvar_95": optimization_payload.get("expected_cvar_95"),
                "instability_index": optimization_payload.get("instability_index"),
                "lambda_t": optimization_payload.get("lambda_t"),
            }
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
