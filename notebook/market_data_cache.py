"""Incremental adjusted-close cache for rate-limit-safe market downloads."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


TickerFetcher = Callable[[str, str | pd.Timestamp, str | pd.Timestamp], pd.Series]


def _load_cache(path: Path) -> pd.DataFrame:
    sidecar = path.with_suffix(path.suffix + ".tmp")
    if not path.exists() and not sidecar.exists():
        return pd.DataFrame()
    frames = []
    for candidate in (path, sidecar):
        if not candidate.exists() or candidate.stat().st_size == 0:
            continue
        try:
            frames.append(pd.read_csv(candidate, index_col=0, parse_dates=True))
        except (OSError, ValueError, pd.errors.ParserError):
            continue
    if not frames:
        return pd.DataFrame()
    frame = frames[0]
    for newer in frames[1:]:
        frame = newer.combine_first(frame)
        for column in newer.columns:
            if column not in frame.columns:
                frame[column] = newer[column]
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def _has_required_coverage(
    series: pd.Series,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> bool:
    values = pd.Series(series).dropna()
    if values.empty:
        return False
    start_tolerance = start + pd.Timedelta(days=7)
    end_tolerance = end_exclusive - pd.Timedelta(days=7)
    return bool(
        values.index.min() <= start_tolerance
        and values.index.max() >= end_tolerance
    )


def _save_cache(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.sort_index().to_csv(temporary)
    try:
        temporary.replace(path)
    except PermissionError:
        # Some Windows/indexer combinations lock the main CSV. The sidecar is
        # intentionally retained and _load_cache treats it as the newer layer.
        return


def update_adjusted_close_cache(
    tickers: Iterable[str],
    start: str | pd.Timestamp,
    end_exclusive: str | pd.Timestamp,
    cache_path: str | Path,
    fetch_ticker: TickerFetcher,
    max_attempts: int = 3,
    pause_seconds: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch missing tickers individually and persist each success immediately."""
    target = list(dict.fromkeys(str(ticker).upper().strip() for ticker in tickers))
    start = pd.Timestamp(start)
    end_exclusive = pd.Timestamp(end_exclusive)
    path = Path(cache_path)
    cache = _load_cache(path)
    audit_rows: list[dict[str, object]] = []

    for ticker in target:
        if ticker in cache.columns and _has_required_coverage(
            cache[ticker], start, end_exclusive
        ):
            audit_rows.append(
                {"ticker": ticker, "status": "cached", "attempts": 0}
            )
            continue

        downloaded = pd.Series(dtype=float)
        attempts = 0
        error = ""
        for attempts in range(1, max(1, int(max_attempts)) + 1):
            try:
                candidate = pd.Series(
                    fetch_ticker(ticker, start, end_exclusive),
                    dtype=float,
                ).dropna()
                candidate.index = pd.to_datetime(candidate.index)
                candidate = candidate[~candidate.index.duplicated(keep="last")]
                if _has_required_coverage(candidate, start, end_exclusive):
                    downloaded = candidate.sort_index()
                    break
                error = "insufficient_date_coverage"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            if pause_seconds > 0:
                time.sleep(float(pause_seconds))

        if not downloaded.empty:
            cache = cache.drop(columns=[ticker], errors="ignore").join(
                downloaded.rename(ticker), how="outer"
            )
            _save_cache(cache, path)
            audit_rows.append(
                {"ticker": ticker, "status": "downloaded", "attempts": attempts}
            )
        else:
            audit_rows.append(
                {
                    "ticker": ticker,
                    "status": "failed",
                    "attempts": attempts,
                    "error": error,
                }
            )

    available = [ticker for ticker in target if ticker in cache.columns]
    return cache.reindex(columns=available).sort_index(), pd.DataFrame(audit_rows)
