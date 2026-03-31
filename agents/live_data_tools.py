import os
import re
from functools import lru_cache

import cvxpy as cp
import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import tool
from pymongo import MongoClient

load_dotenv()


MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "Stock_data"
COLLECTION_NAME = "ticker"


@lru_cache(maxsize=1)
def _get_client():
    if not MONGO_URI:
        raise ValueError("MONGO_URI is not set in the environment.")

    return MongoClient(
        MONGO_URI,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        socketTimeoutMS=15000,
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


def _summarize_metrics(metrics: dict, max_items: int = 8) -> list[str]:
    if not isinstance(metrics, dict) or not metrics:
        return []

    lines = []
    for key, value in list(metrics.items())[:max_items]:
        lines.append(f"  - {key}: {value}")
    return lines


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

    return "\n".join(lines)


@tool
def list_available_sectors() -> str:
    """List distinct sectors available in the MongoDB stock database."""
    try:
        collection = _get_collection()
        docs = list(collection.find({}, {"sector": 1, "info.sector": 1}))

        sectors = set()
        for doc in docs:
            direct_sector = doc.get("sector")
            if isinstance(direct_sector, str) and direct_sector.strip():
                sectors.add(direct_sector.strip())

            info = doc.get("info", {}) if isinstance(doc.get("info"), dict) else {}
            info_sector = info.get("sector")
            if isinstance(info_sector, str) and info_sector.strip():
                sectors.add(info_sector.strip())

        if not sectors:
            return "No sectors were found in the MongoDB database."

        sector_list = sorted(sectors)
        lines = ["Here are the available sectors found in the database:"]
        lines.extend(f"- {sector}" for sector in sector_list)
        return "\n".join(lines)

    except Exception as e:
        return f"Unable to list available sectors due to a database or query error: {type(e).__name__}: {str(e)}"


@tool
def get_stocks_by_universe(universe: str) -> str:
    """Fetch stocks from MongoDB that belong to a requested universe such as U1 or U7."""
    try:
        if not universe or not universe.strip():
            return "Unable to search by universe: no universe was provided."

        normalized_universe = universe.strip().upper()
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
                    "universes": 1,
                },
            )
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
        return "\n".join(lines)

    except Exception as e:
        return f"Unable to search stocks by universe due to a database or query error: {type(e).__name__}: {str(e)}"


@tool
def get_universe_overview(universe: str) -> str:
    """Summarize a universe with its tickers, sector mix, and dominant sector from MongoDB."""
    try:
        if not universe or not universe.strip():
            return "Unable to summarize universe: no universe was provided."

        normalized_universe = universe.strip().upper()
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
            )
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
        return "\n".join(lines)

    except Exception as e:
        return f"Unable to summarize universe due to a database or query error: {type(e).__name__}: {str(e)}"


@tool
def get_stock_database_snapshot(tickers: list[str]) -> str:
    """Fetch a broad MongoDB-backed snapshot for one or more tickers, covering stored metadata and historical coverage."""
    try:
        if not tickers:
            return "Unable to fetch stock database snapshot: no tickers were provided."

        cleaned_tickers = sorted({t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()})
        if not cleaned_tickers:
            return "Unable to fetch stock database snapshot: no valid tickers were provided."

        collection = _get_collection()
        docs = list(
            collection.find(
                {"ticker": {"$in": cleaned_tickers}},
                {
                    "ticker": 1,
                    "symbol": 1,
                    "shortName": 1,
                    "longName": 1,
                    "universes": 1,
                    "historical_prices": 1,
                    "info": 1,
                    "key_stats": 1,
                    "financials": 1,
                    "graph_relationships": 1,
                    "analysis_and_estimates": 1,
                },
            )
        )

        if not docs:
            return "Unable to fetch stock database snapshot: none of the requested tickers were found in MongoDB."

        found = {str(doc.get('ticker', '')).upper(): doc for doc in docs}
        sections = []
        missing = []

        for ticker in cleaned_tickers:
            doc = found.get(ticker)
            if not doc:
                missing.append(f"- {ticker}: ticker not found")
                continue
            sections.append(_format_stock_record(doc))

        response = ["MongoDB Stock Snapshot", ""]
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

        collection = _get_collection()
        pattern = re.escape(sector.strip())
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

        docs = list(collection.find(query, projection))
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
        lines = [f"Here are the stocks in the {sector} sector found in the database:"]
        lines.extend(f"- {ticker}: {company_name}" for ticker, company_name in results)
        return "\n".join(lines)

    except Exception as e:
        return f"Unable to search stocks by sector due to a database or query error: {type(e).__name__}: {str(e)}"


@tool
def analyze_institutional_network(tickers: list[str]) -> str:
    """Analyze institutional-holder network centrality for the requested tickers using MongoDB data."""
    try:
        if not tickers:
            return "Unable to analyze institutional network: no tickers were provided."

        cleaned_tickers = sorted({t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()})
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
    """Fetch historical closing prices from MongoDB on or immediately prior to a target date."""
    try:
        if not tickers:
            return "Unable to fetch historical prices: no tickers were provided."

        cleaned_tickers = sorted({t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()})
        if not cleaned_tickers:
            return "Unable to fetch historical prices: no valid tickers were provided."

        target_dt = pd.to_datetime(target_date, format="%Y-%m-%d", errors="raise")
        collection = _get_collection()

        docs = list(collection.find({"ticker": {"$in": cleaned_tickers}}, {"ticker": 1, "historical_prices": 1}))
        if not docs:
            return f"Unable to fetch historical prices: none of the requested tickers were found in MongoDB for {target_date}."

        found = {doc.get("ticker", "").upper(): doc for doc in docs}
        lines = []
        missing = []

        for ticker in cleaned_tickers:
            doc = found.get(ticker)
            if not doc:
                missing.append(f"{ticker} (ticker not found)")
                continue

            df = _extract_price_frame(doc)
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
                f"Missing details: {', '.join(missing)}"
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

        cleaned_tickers = sorted({t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()})
        if len(cleaned_tickers) < 2:
            return "Unable to run historical CVaR optimization: please provide at least two valid tickers."

        target_dt = pd.to_datetime(target_date, format="%Y-%m-%d", errors="raise")
        collection = _get_collection()

        docs = list(collection.find({"ticker": {"$in": cleaned_tickers}}, {"ticker": 1, "historical_prices": 1}))
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
