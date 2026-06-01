"""
Technical Analysis Service
==========================
Computes 12 standard technical indicators from OHLCV data and detects
trading signals (crossovers, breakouts, support/resistance, trends).

All computations are vectorized via pandas / numpy for performance on
10+ years of daily data (~2 500+ observations per ticker).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept a DataFrame with columns [Date, Open, High, Low, Close, Volume]
    and return a copy with all indicator columns appended.
    """
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ── Simple Moving Averages ────────────────────────────────────────────
    for window in (20, 50, 100, 200):
        df[f"SMA{window}"] = close.rolling(window=window, min_periods=1).mean()

    # ── Exponential Moving Averages ───────────────────────────────────────
    for span in (12, 26):
        df[f"EMA{span}"] = close.ewm(span=span, adjust=False).mean()

    # ── RSI (14) ──────────────────────────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── MACD ──────────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # ── Bollinger Bands (20, 2) ───────────────────────────────────────────
    sma20 = close.rolling(20, min_periods=1).mean()
    std20 = close.rolling(20, min_periods=1).std()
    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Middle"] = sma20
    df["BB_Lower"] = sma20 - 2 * std20
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

    # ── ATR (14) ──────────────────────────────────────────────────────────
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # ── ADX (14) ──────────────────────────────────────────────────────────
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # Zero out when the other is larger
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    atr14 = df["ATR"].replace(0, np.nan)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr14)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr14)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df["ADX"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    df["Plus_DI"] = plus_di
    df["Minus_DI"] = minus_di

    # ── Stochastic Oscillator (k=14, d=3) ─────────────────────────────────
    low14 = low.rolling(14, min_periods=1).min()
    high14 = high.rolling(14, min_periods=1).max()
    df["Stoch_K"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(3, min_periods=1).mean()

    # ── OBV ───────────────────────────────────────────────────────────────
    obv_sign = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    df["OBV"] = (volume * obv_sign).cumsum()

    # ── VWAP ──────────────────────────────────────────────────────────────
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    df["VWAP"] = cum_tp_vol / cum_vol

    return df


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

def detect_golden_death_cross(df: pd.DataFrame) -> list[dict]:
    """
    Detect Golden Cross (SMA50 crosses above SMA200) and
    Death Cross (SMA50 crosses below SMA200).
    """
    signals: list[dict] = []
    if "SMA50" not in df.columns or "SMA200" not in df.columns:
        return signals

    sma50 = df["SMA50"].values
    sma200 = df["SMA200"].values
    dates = df["Date"].values

    for i in range(1, len(df)):
        if pd.isna(sma50[i]) or pd.isna(sma200[i]) or pd.isna(sma50[i - 1]) or pd.isna(sma200[i - 1]):
            continue
        prev_diff = sma50[i - 1] - sma200[i - 1]
        curr_diff = sma50[i] - sma200[i]
        if prev_diff <= 0 < curr_diff:
            signals.append({
                "date": str(pd.Timestamp(dates[i]).date()),
                "type": "golden",
                "sma50": round(float(sma50[i]), 2),
                "sma200": round(float(sma200[i]), 2),
            })
        elif prev_diff >= 0 > curr_diff:
            signals.append({
                "date": str(pd.Timestamp(dates[i]).date()),
                "type": "death",
                "sma50": round(float(sma50[i]), 2),
                "sma200": round(float(sma200[i]), 2),
            })
    return signals


def detect_rsi_signals(df: pd.DataFrame) -> dict:
    """Detect RSI overbought/oversold zones and buy/sell signals."""
    result: dict[str, Any] = {
        "overbought_zones": [],
        "oversold_zones": [],
        "buy_signals": [],
        "sell_signals": [],
    }
    if "RSI" not in df.columns:
        return result

    rsi = df["RSI"].values
    dates = df["Date"].values

    in_overbought = False
    in_oversold = False
    ob_start = None
    os_start = None

    for i in range(len(df)):
        if pd.isna(rsi[i]):
            continue
        date_str = str(pd.Timestamp(dates[i]).date())

        # Overbought zones
        if rsi[i] > 70 and not in_overbought:
            in_overbought = True
            ob_start = date_str
        elif rsi[i] <= 70 and in_overbought:
            in_overbought = False
            result["overbought_zones"].append({"start": ob_start, "end": date_str})
            result["sell_signals"].append({"date": date_str, "rsi": round(float(rsi[i]), 2), "reason": "RSI dropped below 70"})

        # Oversold zones
        if rsi[i] < 30 and not in_oversold:
            in_oversold = True
            os_start = date_str
        elif rsi[i] >= 30 and in_oversold:
            in_oversold = False
            result["oversold_zones"].append({"start": os_start, "end": date_str})
            result["buy_signals"].append({"date": date_str, "rsi": round(float(rsi[i]), 2), "reason": "RSI rose above 30"})

    return result


def detect_macd_signals(df: pd.DataFrame) -> dict:
    """Detect MACD bullish/bearish crossovers and trend shifts."""
    result: dict[str, Any] = {
        "bullish_crossovers": [],
        "bearish_crossovers": [],
        "trend_shifts": [],
    }
    if "MACD" not in df.columns or "MACD_Signal" not in df.columns:
        return result

    macd = df["MACD"].values
    signal = df["MACD_Signal"].values
    hist = df["MACD_Hist"].values
    dates = df["Date"].values

    for i in range(1, len(df)):
        if any(pd.isna(v) for v in [macd[i], signal[i], macd[i - 1], signal[i - 1]]):
            continue
        date_str = str(pd.Timestamp(dates[i]).date())

        prev_diff = macd[i - 1] - signal[i - 1]
        curr_diff = macd[i] - signal[i]

        if prev_diff <= 0 < curr_diff:
            result["bullish_crossovers"].append({
                "date": date_str,
                "macd": round(float(macd[i]), 4),
                "signal": round(float(signal[i]), 4),
            })
        elif prev_diff >= 0 > curr_diff:
            result["bearish_crossovers"].append({
                "date": date_str,
                "macd": round(float(macd[i]), 4),
                "signal": round(float(signal[i]), 4),
            })

        # Trend shift: histogram sign change
        if not pd.isna(hist[i]) and not pd.isna(hist[i - 1]):
            if hist[i - 1] < 0 <= hist[i]:
                result["trend_shifts"].append({"date": date_str, "direction": "bullish"})
            elif hist[i - 1] > 0 >= hist[i]:
                result["trend_shifts"].append({"date": date_str, "direction": "bearish"})

    return result


def detect_bollinger_signals(df: pd.DataFrame) -> dict:
    """Detect Bollinger Band breakouts, squeezes, and volatility expansions."""
    result: dict[str, Any] = {
        "upper_breakouts": [],
        "lower_breakouts": [],
        "squeezes": [],
        "expansions": [],
    }
    if "BB_Upper" not in df.columns:
        return result

    close = df["Close"].values
    upper = df["BB_Upper"].values
    lower = df["BB_Lower"].values
    width = df["BB_Width"].values
    dates = df["Date"].values

    # Median bandwidth for squeeze/expansion detection
    valid_widths = width[~np.isnan(width)]
    if len(valid_widths) == 0:
        return result
    median_width = float(np.median(valid_widths))

    in_squeeze = False
    squeeze_start = None

    for i in range(1, len(df)):
        if any(pd.isna(v) for v in [close[i], upper[i], lower[i], width[i]]):
            continue
        date_str = str(pd.Timestamp(dates[i]).date())

        # Upper breakout
        if close[i] > upper[i] and close[i - 1] <= upper[i - 1] if not pd.isna(close[i - 1]) else False:
            result["upper_breakouts"].append({"date": date_str, "close": round(float(close[i]), 2), "upper": round(float(upper[i]), 2)})

        # Lower breakout
        if close[i] < lower[i] and close[i - 1] >= lower[i - 1] if not pd.isna(close[i - 1]) else False:
            result["lower_breakouts"].append({"date": date_str, "close": round(float(close[i]), 2), "lower": round(float(lower[i]), 2)})

        # Squeeze detection (bandwidth < 50% of median)
        if width[i] < median_width * 0.5:
            if not in_squeeze:
                in_squeeze = True
                squeeze_start = date_str
        else:
            if in_squeeze:
                in_squeeze = False
                result["squeezes"].append({"start": squeeze_start, "end": date_str})
                result["expansions"].append({"date": date_str, "width": round(float(width[i]), 4)})

    return result


def detect_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Detect support and resistance levels using swing highs/lows (fractal method).
    Pivot points are calculated from the most recent complete trading day.
    """
    result: dict[str, Any] = {
        "support_levels": [],
        "resistance_levels": [],
        "pivot_points": [],
    }

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    dates = df["Date"].values

    if len(df) < window * 2 + 1:
        return result

    # Swing highs / lows (fractal method)
    swing_highs: list[dict] = []
    swing_lows: list[dict] = []

    half = window // 2
    for i in range(half, len(df) - half):
        # Swing high: current high is the max in the window
        window_highs = high[i - half: i + half + 1]
        if high[i] == np.max(window_highs) and not pd.isna(high[i]):
            swing_highs.append({"date": str(pd.Timestamp(dates[i]).date()), "price": round(float(high[i]), 2)})

        # Swing low: current low is the min in the window
        window_lows = low[i - half: i + half + 1]
        if low[i] == np.min(window_lows) and not pd.isna(low[i]):
            swing_lows.append({"date": str(pd.Timestamp(dates[i]).date()), "price": round(float(low[i]), 2)})

    # Cluster swing highs/lows to find major levels (within 1.5% of each other)
    def _cluster_levels(levels: list[dict], pct_threshold: float = 0.015) -> list[dict]:
        if not levels:
            return []
        sorted_levels = sorted(levels, key=lambda x: x["price"])
        clusters: list[list[dict]] = [[sorted_levels[0]]]

        for lvl in sorted_levels[1:]:
            cluster_avg = np.mean([l["price"] for l in clusters[-1]])
            if abs(lvl["price"] - cluster_avg) / cluster_avg < pct_threshold:
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])

        major: list[dict] = []
        for cluster in clusters:
            if len(cluster) >= 2:  # At least 2 touches
                avg_price = round(float(np.mean([l["price"] for l in cluster])), 2)
                touches = len(cluster)
                first_date = cluster[0]["date"]
                last_date = cluster[-1]["date"]
                major.append({"price": avg_price, "touches": touches, "first_date": first_date, "last_date": last_date})
        return major

    result["resistance_levels"] = _cluster_levels(swing_highs)
    result["support_levels"] = _cluster_levels(swing_lows)

    # Pivot points from last complete day
    if len(df) >= 2:
        last = df.iloc[-1]
        pp = round(float((last["High"] + last["Low"] + last["Close"]) / 3), 2)
        r1 = round(float(2 * pp - last["Low"]), 2)
        s1 = round(float(2 * pp - last["High"]), 2)
        r2 = round(float(pp + (last["High"] - last["Low"])), 2)
        s2 = round(float(pp - (last["High"] - last["Low"])), 2)
        result["pivot_points"] = [
            {"level": "S2", "price": s2},
            {"level": "S1", "price": s1},
            {"level": "PP", "price": pp},
            {"level": "R1", "price": r1},
            {"level": "R2", "price": r2},
        ]

    return result


def detect_trend(df: pd.DataFrame) -> dict:
    """
    Detect the current trend direction using SMA20/SMA50 slope, ADX strength,
    and price position relative to moving averages.
    """
    result: dict[str, Any] = {
        "trend": "sideways",
        "strength": "weak",
        "adx_value": None,
        "reversal_points": [],
        "summary": "",
    }

    if len(df) < 50:
        result["summary"] = "Insufficient data for trend analysis (need at least 50 observations)."
        return result

    # Recent window (last 20 days)
    recent = df.tail(20)
    close = recent["Close"].values

    # Trend direction from SMA slopes
    sma20_recent = recent["SMA20"].dropna()
    sma50_recent = recent["SMA50"].dropna() if "SMA50" in recent.columns else pd.Series(dtype=float)

    if len(sma20_recent) >= 5:
        sma20_slope = (sma20_recent.iloc[-1] - sma20_recent.iloc[-5]) / sma20_recent.iloc[-5] * 100
    else:
        sma20_slope = 0.0

    # ADX for trend strength
    adx_val = None
    if "ADX" in df.columns and not df["ADX"].dropna().empty:
        adx_val = round(float(df["ADX"].dropna().iloc[-1]), 2)
        result["adx_value"] = adx_val

    # Price vs MAs
    last_close = float(close[-1]) if len(close) > 0 else 0
    above_sma20 = last_close > float(sma20_recent.iloc[-1]) if len(sma20_recent) > 0 else None
    above_sma50 = last_close > float(sma50_recent.iloc[-1]) if len(sma50_recent) > 0 else None

    # Determine trend
    if sma20_slope > 0.5 and above_sma20 and above_sma50:
        result["trend"] = "bullish"
    elif sma20_slope < -0.5 and not above_sma20 and not above_sma50:
        result["trend"] = "bearish"
    else:
        result["trend"] = "sideways"

    # Strength from ADX
    if adx_val is not None:
        if adx_val > 40:
            result["strength"] = "very strong"
        elif adx_val > 25:
            result["strength"] = "strong"
        elif adx_val > 20:
            result["strength"] = "moderate"
        else:
            result["strength"] = "weak"

    # Detect reversal points (SMA20 slope sign changes)
    if "SMA20" in df.columns:
        sma20_full = df["SMA20"].values
        dates = df["Date"].values
        for i in range(5, len(df)):
            if pd.isna(sma20_full[i]) or pd.isna(sma20_full[i - 5]):
                continue
            curr_slope = sma20_full[i] - sma20_full[i - 5]
            prev_slope = sma20_full[i - 1] - sma20_full[i - 6] if i >= 6 and not pd.isna(sma20_full[i - 6]) else None
            if prev_slope is not None:
                if prev_slope < 0 <= curr_slope:
                    result["reversal_points"].append({
                        "date": str(pd.Timestamp(dates[i]).date()),
                        "type": "bullish_reversal",
                        "price": round(float(df["Close"].values[i]), 2),
                    })
                elif prev_slope > 0 >= curr_slope:
                    result["reversal_points"].append({
                        "date": str(pd.Timestamp(dates[i]).date()),
                        "type": "bearish_reversal",
                        "price": round(float(df["Close"].values[i]), 2),
                    })

    # Keep only recent reversal points (last 10)
    result["reversal_points"] = result["reversal_points"][-10:]

    result["summary"] = (
        f"Current trend: {result['trend'].upper()} "
        f"(strength: {result['strength']}, ADX: {adx_val or 'N/A'}). "
        f"Price {'above' if above_sma20 else 'below'} SMA20, "
        f"{'above' if above_sma50 else 'below'} SMA50. "
        f"SMA20 slope: {sma20_slope:+.2f}%."
    )
    return result


# ---------------------------------------------------------------------------
# Convenience: run everything at once
# ---------------------------------------------------------------------------

def run_full_analysis(df: pd.DataFrame) -> dict:
    """
    Run all indicators + all signal detectors and return a single dict.
    The DataFrame must have columns: Date, Open, High, Low, Close, Volume.
    """
    df_ind = compute_all_indicators(df)

    return {
        "indicators_df": df_ind,
        "cross_signals": detect_golden_death_cross(df_ind),
        "rsi_signals": detect_rsi_signals(df_ind),
        "macd_signals": detect_macd_signals(df_ind),
        "bollinger_signals": detect_bollinger_signals(df_ind),
        "support_resistance": detect_support_resistance(df_ind),
        "trend": detect_trend(df_ind),
    }
