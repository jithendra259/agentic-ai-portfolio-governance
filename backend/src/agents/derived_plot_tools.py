import uuid
from typing import Any

import pandas as pd
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.agents.plot_store import register_plot
from src.agents.price_series_tool import get_price_series_for_analysis, load_cached_analysis_dataset
from src.agents.live_data_tools import _extract_price_frame, _find_price_documents_with_retry, _get_collection, _normalize_tickers


OHLC_FIELDS = ("open", "high", "low", "close")
MAX_MISSING_HEATMAP_TICKERS = 30
MAX_MISSING_HEATMAP_DATES = 260
SUPPORTED_ANALYSIS_TASKS = {
    "missing_data_heatmap",
    "ohlc_correlation_heatmap",
    "returns_correlation_heatmap",
    "price_line",
    "returns_box_plot",
}


def _dataset_from_cache_or_query(
    ticker: str,
    start_date: str,
    end_date: str,
    analysis_cache_key: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if analysis_cache_key:
        cached = load_cached_analysis_dataset(analysis_cache_key)
        if cached:
            return cached, analysis_cache_key

    result = get_price_series_for_analysis.invoke(
        {
            "tickers": [ticker],
            "start_date": start_date,
            "end_date": end_date,
        }
    )
    if not isinstance(result, dict) or result.get("error"):
        return None, None

    cache_key = result.get("analysis_cache_key")
    return load_cached_analysis_dataset(cache_key), cache_key


def _ohlc_frame(dataset: dict[str, Any], ticker: str) -> pd.DataFrame:
    rows = (dataset.get("prices") or {}).get(ticker.upper()) or []
    if not rows:
        raise ValueError(f"No OHLC rows found for {ticker.upper()}.")

    frame = pd.DataFrame(rows)
    missing = [field for field in OHLC_FIELDS if field not in frame.columns]
    if missing:
        raise ValueError(f"Missing OHLC fields for {ticker.upper()}: {', '.join(missing)}.")

    frame = frame[list(OHLC_FIELDS)].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 5:
        raise ValueError(f"Not enough complete OHLC rows for {ticker.upper()} correlation.")
    return frame


def _sample_labels(labels: list[str], max_count: int) -> list[str]:
    if len(labels) <= max_count:
        return labels
    step = (len(labels) + max_count - 1) // max_count
    sampled = labels[::step]
    if labels[-1] not in sampled:
        sampled.append(labels[-1])
    return sampled[:max_count]


def _resolve_tickers_for_missing_heatmap(
    tickers: list[str] | None,
    sector: str | None,
    universe: str | None,
    analysis_cache_key: str | None,
) -> list[str]:
    explicit = _normalize_tickers(tickers or [])
    if explicit:
        return explicit[:MAX_MISSING_HEATMAP_TICKERS]

    if analysis_cache_key:
        cached = load_cached_analysis_dataset(analysis_cache_key)
        cached_tickers = _normalize_tickers((cached or {}).get("tickers_included", []))
        if cached_tickers:
            return cached_tickers[:MAX_MISSING_HEATMAP_TICKERS]

    collection = _get_collection()
    query: dict[str, Any]
    if sector and str(sector).strip():
        pattern = str(sector).strip()
        query = {
            "$or": [
                {"sector": {"$regex": pattern, "$options": "i"}},
                {"info.sector": {"$regex": pattern, "$options": "i"}},
            ]
        }
    elif universe and str(universe).strip():
        query = {"universes": str(universe).strip().upper()}
    else:
        query = {"historical_prices.0": {"$exists": True}}

    docs = list(
        collection.find(query, {"ticker": 1, "symbol": 1})
        .sort("ticker", 1)
        .limit(MAX_MISSING_HEATMAP_TICKERS)
    )
    return _normalize_tickers([doc.get("ticker") or doc.get("symbol") for doc in docs])[:MAX_MISSING_HEATMAP_TICKERS]


def _missing_heatmap_spec(
    tickers: list[str],
    start_date: str,
    end_date: str,
    title: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    docs = _find_price_documents_with_retry(
        tickers,
        start_date=start_date,
        end_date=end_date,
        keep_ohlcv=False,
    )
    docs_by_ticker = {str(doc.get("ticker", "")).upper(): doc for doc in docs}
    frames: dict[str, pd.DataFrame] = {}

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    for ticker in tickers:
        frame = _extract_price_frame(docs_by_ticker.get(ticker, {}))
        if frame.empty:
            continue
        frame = frame[(frame["Date"] >= start_dt) & (frame["Date"] <= end_dt)].copy()
        if not frame.empty:
            frames[ticker] = frame

    if not frames:
        raise ValueError("No historical price rows found for the selected missing-data heatmap scope.")

    all_dates = sorted(
        {
            row.strftime("%Y-%m-%d")
            for frame in frames.values()
            for row in pd.to_datetime(frame["Date"], errors="coerce").dropna()
        }
    )
    if not all_dates:
        raise ValueError("No valid dates found for the selected missing-data heatmap scope.")

    sampled_dates = _sample_labels(all_dates, MAX_MISSING_HEATMAP_DATES)
    date_sets = {
        ticker: {
            row.strftime("%Y-%m-%d")
            for row in pd.to_datetime(frame["Date"], errors="coerce").dropna()
        }
        for ticker, frame in frames.items()
    }

    matrix = {
        date: {
            ticker: (1 if date in date_sets.get(ticker, set()) else 0)
            for ticker in tickers
        }
        for date in sampled_dates
    }
    missing_counts = {
        ticker: int(sum(1 for date in all_dates if date not in date_sets.get(ticker, set())))
        for ticker in tickers
    }
    present_counts = {
        ticker: int(sum(1 for date in all_dates if date in date_sets.get(ticker, set())))
        for ticker in tickers
    }
    series_data = [
        [ticker_index, date_index, matrix[date][ticker]]
        for date_index, date in enumerate(sampled_dates)
        for ticker_index, ticker in enumerate(tickers)
    ]

    spec = {
        "plot_type": "heatmap",
        "title": title,
        "matrix": matrix,
        "metadata": {
            "heatmap_type": "missing",
            "value_meaning": {"1": "data present", "0": "missing"},
            "start_date": start_date,
            "end_date": end_date,
            "tickers": tickers,
            "total_dates": len(all_dates),
            "rendered_dates": len(sampled_dates),
            "missing_counts": missing_counts,
            "present_counts": present_counts,
        },
        "xAxis": [{"data": tickers}],
        "yAxis": [{"data": sampled_dates}],
        "series": [{"data": series_data}],
        "height": min(720, max(360, len(sampled_dates) * 12 + 120)),
        "hideLegend": False,
    }
    summary = {
        "tickers": tickers,
        "total_dates": len(all_dates),
        "rendered_dates": len(sampled_dates),
        "missing_counts": missing_counts,
        "present_counts": present_counts,
    }
    return spec, summary


def _register_spec(spec: dict[str, Any], config: RunnableConfig = None) -> str:
    plot_id = str(uuid.uuid4())
    session_id = (
        config.get("configurable", {}).get("thread_id", "default")
        if config
        else "default"
    )
    register_plot(plot_id, spec, session_id)
    return plot_id


def _returns_correlation_heatmap_spec(
    tickers: list[str],
    start_date: str,
    end_date: str,
    analysis_cache_key: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = load_cached_analysis_dataset(analysis_cache_key) if analysis_cache_key else None
    cache_key = analysis_cache_key
    if not dataset:
        result = get_price_series_for_analysis.invoke(
            {"tickers": tickers, "start_date": start_date, "end_date": end_date}
        )
        if not isinstance(result, dict) or result.get("error"):
            raise ValueError((result or {}).get("error") or "Unable to fetch price series.")
        cache_key = result.get("analysis_cache_key")
        dataset = load_cached_analysis_dataset(cache_key)
    if not dataset:
        raise ValueError("Unable to load cached returns dataset.")

    returns = dataset.get("returns") or {}
    dates = dataset.get("return_dates_by_ticker") or {}
    series = []
    for ticker in tickers:
        vals = returns.get(ticker, [])
        idx = dates.get(ticker, [])
        if vals and idx:
            series.append(pd.Series(vals, index=pd.to_datetime(idx), name=ticker))
    if len(series) < 2:
        raise ValueError("At least two tickers with returns are required for returns correlation.")

    frame = pd.concat(series, axis=1).sort_index().dropna()
    if frame.empty:
        raise ValueError("No overlapping returns were found for the selected tickers.")
    corr = frame.corr()
    matrix = {
        row: {col: round(float(corr.loc[row, col]), 6) for col in corr.columns}
        for row in corr.index
    }
    labels = [str(label) for label in corr.columns]
    spec = {
        "plot_type": "heatmap",
        "title": f"Returns Correlation Heatmap - {start_date} to {end_date}",
        "matrix": matrix,
        "metadata": {
            "heatmap_type": "correlation",
            "method": "pearson",
            "analysis_cache_key": cache_key,
            "observations": int(len(frame)),
            "tickers": labels,
        },
        "xAxis": [{"data": labels}],
        "yAxis": [{"data": labels}],
        "series": [{
            "data": [
                [col_index, row_index, float(corr.loc[row, col])]
                for row_index, row in enumerate(corr.index)
                for col_index, col in enumerate(corr.columns)
            ]
        }],
        "height": min(720, max(360, len(labels) * 42 + 120)),
    }
    return spec, {"analysis_cache_key": cache_key, "observations": int(len(frame)), "tickers": labels}


def _price_line_spec(
    tickers: list[str],
    start_date: str,
    end_date: str,
    analysis_cache_key: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = load_cached_analysis_dataset(analysis_cache_key) if analysis_cache_key else None
    cache_key = analysis_cache_key
    if not dataset:
        result = get_price_series_for_analysis.invoke(
            {"tickers": tickers, "start_date": start_date, "end_date": end_date}
        )
        if not isinstance(result, dict) or result.get("error"):
            raise ValueError((result or {}).get("error") or "Unable to fetch price series.")
        cache_key = result.get("analysis_cache_key")
        dataset = load_cached_analysis_dataset(cache_key)
    if not dataset:
        raise ValueError("Unable to load cached price dataset.")

    prices = dataset.get("prices") or {}
    series = []
    for ticker in tickers:
        rows = prices.get(ticker, [])
        points = [
            {"x": row.get("date"), "y": row.get("close")}
            for row in rows
            if row.get("date") and row.get("close") is not None
        ]
        if points:
            series.append({"name": ticker, "label": ticker, "data": points, "showMark": False})
    if not series:
        raise ValueError("No close-price series found for selected tickers.")

    total_points = sum(len(item["data"]) for item in series)
    spec = {
        "plot_type": "line",
        "title": f"Close Price Trend - {start_date} to {end_date}",
        "x_label": "Date",
        "x_type": "time",
        "y_label": "Close Price",
        "series": series,
        "grid": {"horizontal": True},
        "curve": "monotoneX",
        "skipAnimation": total_points > 500,
        "density": {"rendered_points": total_points},
    }
    return spec, {"analysis_cache_key": cache_key, "rendered_points": total_points, "tickers": tickers}


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty series.")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _returns_box_plot_spec(
    tickers: list[str],
    start_date: str,
    end_date: str,
    analysis_cache_key: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = load_cached_analysis_dataset(analysis_cache_key) if analysis_cache_key else None
    cache_key = analysis_cache_key
    if not dataset:
        result = get_price_series_for_analysis.invoke(
            {"tickers": tickers, "start_date": start_date, "end_date": end_date}
        )
        if not isinstance(result, dict) or result.get("error"):
            raise ValueError((result or {}).get("error") or "Unable to fetch price series.")
        cache_key = result.get("analysis_cache_key")
        dataset = load_cached_analysis_dataset(cache_key)
    if not dataset:
        raise ValueError("Unable to load cached returns dataset.")

    returns = dataset.get("returns") or {}
    boxes = []
    for ticker in tickers:
        values = sorted(float(value) * 100 for value in returns.get(ticker, []) if value is not None)
        if len(values) < 5:
            continue
        q1 = _percentile(values, 0.25)
        median = _percentile(values, 0.5)
        q3 = _percentile(values, 0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        inlier_values = [value for value in values if lower_fence <= value <= upper_fence]
        outliers = [round(value, 4) for value in values if value < lower_fence or value > upper_fence]
        boxes.append(
            {
                "label": ticker,
                "min": round(min(inlier_values), 4),
                "q1": round(q1, 4),
                "median": round(median, 4),
                "q3": round(q3, 4),
                "max": round(max(inlier_values), 4),
                "outliers": outliers[:80],
                "outlier_count": len(outliers),
                "sample_size": len(values),
            }
        )
    if not boxes:
        raise ValueError("No ticker had enough daily returns for a box plot.")

    spec = {
        "plot_type": "box",
        "title": f"Daily Returns Box Plot - {start_date} to {end_date}",
        "y_label": "Daily return (%)",
        "data": boxes,
        "metadata": {
            "analysis_task": "returns_box_plot",
            "analysis_cache_key": cache_key,
            "start_date": start_date,
            "end_date": end_date,
            "tickers": [box["label"] for box in boxes],
        },
        "height": 380,
    }
    return spec, {"analysis_cache_key": cache_key, "tickers": [box["label"] for box in boxes]}


@tool
def generate_ohlc_correlation_heatmap(
    ticker: str,
    start_date: str = "2005-01-01",
    end_date: str = "2025-12-31",
    analysis_cache_key: str | None = None,
    method: str = "pearson",
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """
    Compute Open/High/Low/Close correlations for one ticker and attach a heatmap.

    Use this for requests like "correlation between stock prices OHLC of AXP in
    heatmap", "OHLC correlation matrix", or "open high low close heat map".
    It computes the matrix from stored historical OHLC data and registers an
    interactive PlotSpec for the chat UI.
    """
    clean_ticker = str(ticker or "").strip().upper()
    if not clean_ticker:
        return {"status": "error", "error": "ticker is required."}

    try:
        dataset, cache_key = _dataset_from_cache_or_query(
            clean_ticker,
            str(start_date or "2005-01-01"),
            str(end_date or "2025-12-31"),
            analysis_cache_key,
        )
        if not dataset:
            return {"status": "error", "error": f"No cached or stored OHLC data found for {clean_ticker}."}

        frame = _ohlc_frame(dataset, clean_ticker)
        corr = frame.corr(method=method if method in {"pearson", "spearman", "kendall"} else "pearson")
        matrix = {
            row: {col: round(float(corr.loc[row, col]), 6) for col in OHLC_FIELDS}
            for row in OHLC_FIELDS
        }
        title = f"{clean_ticker} OHLC Correlation Heatmap"
        spec = {
            "plot_type": "heatmap",
            "title": title,
            "matrix": matrix,
            "metadata": {
                "heatmap_type": "correlation",
                "ticker": clean_ticker,
                "fields": list(OHLC_FIELDS),
                "method": method,
                "observations": int(len(frame)),
                "analysis_cache_key": cache_key,
            },
            "xAxis": [{"data": [field.title() for field in OHLC_FIELDS]}],
            "yAxis": [{"data": [field.title() for field in OHLC_FIELDS]}],
            "series": [
                {
                    "data": [
                        [col_index, row_index, float(corr.loc[row, col])]
                        for row_index, row in enumerate(OHLC_FIELDS)
                        for col_index, col in enumerate(OHLC_FIELDS)
                    ]
                }
            ],
            "height": 360,
        }

        plot_id = _register_spec(spec, config)
        return {
            "status": "success",
            "message": f"Generated {title}.",
            "plot_id": plot_id,
            "analysis_cache_key": cache_key,
            "observations": int(len(frame)),
            "correlation_matrix": matrix,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@tool
def generate_missing_data_heatmap(
    tickers: list[str] | None = None,
    sector: str | None = None,
    universe: str | None = None,
    start_date: str = "2005-01-01",
    end_date: str = "2025-12-31",
    analysis_cache_key: str | None = None,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """
    Build and attach a missing-data heatmap from stored historical price data.

    Use this when the user asks for a missing data heatmap, data availability
    heatmap, coverage heatmap, or gaps in historical stock data. The tool can
    derive its scope from explicit tickers, a sector such as Healthcare, a
    universe such as U1, or a previous analysis_cache_key.
    """
    try:
        resolved_tickers = _resolve_tickers_for_missing_heatmap(
            tickers=tickers,
            sector=sector,
            universe=universe,
            analysis_cache_key=analysis_cache_key,
        )
        if not resolved_tickers:
            return {
                "status": "error",
                "error": "No tickers could be resolved for the missing-data heatmap scope.",
            }

        scope = sector or universe or "Selected Tickers"
        title = f"Missing Data Heatmap - {scope}"
        spec, summary = _missing_heatmap_spec(
            resolved_tickers,
            str(start_date or "2005-01-01"),
            str(end_date or "2025-12-31"),
            title,
        )

        plot_id = _register_spec(spec, config)
        return {
            "status": "success",
            "message": f"Generated {title}.",
            "plot_id": plot_id,
            **summary,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@tool
def run_data_analysis_plot(
    analysis_task: str,
    tickers: list[str] | None = None,
    sector: str | None = None,
    universe: str | None = None,
    ticker: str | None = None,
    start_date: str = "2005-01-01",
    end_date: str = "2025-12-31",
    analysis_cache_key: str | None = None,
    method: str = "pearson",
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """
    Common safe analysis-to-plot tool for chatbot chart generation.

    The LLM should call this instead of asking the user for intermediate
    matrices/tables. It resolves data scope, runs approved pandas transforms,
    cleans/aligns data as needed, registers a PlotSpec, and returns metadata.

    Supported analysis_task values:
    - missing_data_heatmap
    - ohlc_correlation_heatmap
    - returns_correlation_heatmap
    - price_line
    - returns_box_plot
    """
    task = str(analysis_task or "").strip().lower()
    if task not in SUPPORTED_ANALYSIS_TASKS:
        return {
            "status": "error",
            "error": f"Unsupported analysis_task '{analysis_task}'. Supported: {', '.join(sorted(SUPPORTED_ANALYSIS_TASKS))}.",
        }

    try:
        if task == "missing_data_heatmap":
            resolved = _resolve_tickers_for_missing_heatmap(tickers, sector, universe, analysis_cache_key)
            if not resolved:
                raise ValueError("No tickers could be resolved from tickers, sector, universe, or analysis_cache_key.")
            scope = sector or universe or "Selected Tickers"
            spec, summary = _missing_heatmap_spec(resolved, start_date, end_date, f"Missing Data Heatmap - {scope}")
            plot_id = _register_spec(spec, config)
            return {"status": "success", "plot_id": plot_id, "analysis_task": task, **summary}

        if task == "ohlc_correlation_heatmap":
            clean_ticker = str(ticker or (tickers or [""])[0]).strip().upper()
            result = generate_ohlc_correlation_heatmap.func(
                ticker=clean_ticker,
                start_date=start_date,
                end_date=end_date,
                analysis_cache_key=analysis_cache_key,
                method=method,
                config=config,
            )
            if isinstance(result, dict):
                result["analysis_task"] = task
            return result

        resolved_tickers = _normalize_tickers(tickers or ([ticker] if ticker else []))
        if not resolved_tickers:
            resolved_tickers = _resolve_tickers_for_missing_heatmap(None, sector, universe, analysis_cache_key)
        if not resolved_tickers:
            raise ValueError("No tickers could be resolved for this analysis.")

        if task == "returns_correlation_heatmap":
            spec, summary = _returns_correlation_heatmap_spec(resolved_tickers, start_date, end_date, analysis_cache_key)
        elif task == "price_line":
            spec, summary = _price_line_spec(resolved_tickers, start_date, end_date, analysis_cache_key)
        elif task == "returns_box_plot":
            spec, summary = _returns_box_plot_spec(resolved_tickers, start_date, end_date, analysis_cache_key)
        else:
            raise ValueError(f"Unhandled analysis task: {task}")

        plot_id = _register_spec(spec, config)
        return {"status": "success", "plot_id": plot_id, "analysis_task": task, **summary}
    except Exception as exc:
        return {"status": "error", "analysis_task": task, "error": str(exc)}
