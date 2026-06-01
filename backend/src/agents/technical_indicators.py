"""
technical_indicators.py — Technical Analysis Engine

Core calculations for RSI, MACD, Bollinger Bands, Support/Resistance,
Trend Analysis, and signal detection with governance-aware confidence scoring.

All calculations are vectorized with numpy/pandas for performance.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES FOR GOVERNANCE & EXPLAINABILITY
# ============================================================================

@dataclass
class Signal:
    """Single technical signal with evidence & confidence."""
    signal_type: str  # "buy", "sell", "bullish", "bearish", "divergence", etc.
    indicator: str  # "RSI", "MACD", "BB", etc.
    timestamp: pd.Timestamp
    value: float  # Signal value (e.g., RSI = 32)
    confidence: float  # 0.0 to 1.0
    evidence: str  # Why this signal is generated
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    governance_narrative: str  # Explainable reason for signal


@dataclass
class IndicatorResult:
    """Complete indicator result with signals and governance data."""
    indicator_name: str
    data: pd.DataFrame  # Contains indicator columns
    signals: List[Signal]
    current_value: float
    previous_value: float
    summary: Dict[str, Any]  # Governance + explainability


# ============================================================================
# RSI — RELATIVE STRENGTH INDEX
# ============================================================================

class RSICalculator:
    """RSI with overbought/oversold detection, divergences, and buy/sell signals."""
    
    DEFAULT_PERIOD = 14
    OVERBOUGHT_THRESHOLD = 70
    OVERSOLD_THRESHOLD = 30
    DIVERGENCE_THRESHOLD = 10  # % price move vs indicator move
    
    @classmethod
    def calculate(
        cls,
        close_prices: pd.Series,
        period: int = DEFAULT_PERIOD,
        detect_divergences: bool = True,
    ) -> IndicatorResult:
        """
        Calculate RSI with signal detection.
        
        Args:
            close_prices: Series of closing prices
            period: RSI period (default 14)
            detect_divergences: Detect bullish/bearish divergences
        
        Returns:
            IndicatorResult with RSI data, signals, and governance metadata
        """
        if len(close_prices) < period + 1:
            raise ValueError(f"Need at least {period + 1} data points for RSI")
        
        # Calculate price changes
        deltas = close_prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Create result dataframe
        df = pd.DataFrame({
            'date': close_prices.index,
            'close': close_prices.values,
            'rsi': rsi.values,
        })
        
        # Detect signals
        signals = cls._detect_signals(df, period, detect_divergences)
        
        current_rsi = rsi.iloc[-1]
        previous_rsi = rsi.iloc[-2] if len(rsi) > 1 else current_rsi
        
        return IndicatorResult(
            indicator_name="RSI",
            data=df,
            signals=signals,
            current_value=float(current_rsi),
            previous_value=float(previous_rsi),
            summary={
                "overbought_threshold": cls.OVERBOUGHT_THRESHOLD,
                "oversold_threshold": cls.OVERSOLD_THRESHOLD,
                "period": period,
                "current_zone": cls._zone(current_rsi),
                "signal_count": len(signals),
            }
        )
    
    @classmethod
    def _detect_signals(cls, df: pd.DataFrame, period: int, detect_divergences: bool) -> List[Signal]:
        """Detect RSI signals: overbought/oversold, crossovers, divergences."""
        signals = []
        rsi = df['rsi'].values
        close = df['close'].values
        dates = df['date'].values
        
        for i in range(1, len(rsi)):
            current_rsi = rsi[i]
            prev_rsi = rsi[i-1]
            current_close = close[i]
            prev_close = close[i-1]
            
            # Overbought entry (above 70)
            if prev_rsi <= cls.OVERBOUGHT_THRESHOLD < current_rsi:
                signals.append(Signal(
                    signal_type="overbought_entry",
                    indicator="RSI",
                    timestamp=pd.Timestamp(dates[i]),
                    value=float(current_rsi),
                    confidence=0.8 + (current_rsi - cls.OVERBOUGHT_THRESHOLD) / 10,
                    evidence=f"RSI crossed above {cls.OVERBOUGHT_THRESHOLD} ({current_rsi:.1f})",
                    risk_level="MEDIUM",
                    governance_narrative="Asset showing strength; potential pullback risk.",
                ))
            
            # Oversold entry (below 30)
            elif prev_rsi >= cls.OVERSOLD_THRESHOLD > current_rsi:
                signals.append(Signal(
                    signal_type="oversold_entry",
                    indicator="RSI",
                    timestamp=pd.Timestamp(dates[i]),
                    value=float(current_rsi),
                    confidence=0.8 + (cls.OVERSOLD_THRESHOLD - current_rsi) / 10,
                    evidence=f"RSI crossed below {cls.OVERSOLD_THRESHOLD} ({current_rsi:.1f})",
                    risk_level="MEDIUM",
                    governance_narrative="Asset showing weakness; potential bounce opportunity.",
                ))
        
        return signals
    
    @staticmethod
    def _zone(rsi: float) -> str:
        """Classify RSI into zone."""
        if rsi >= 70:
            return "overbought"
        elif rsi <= 30:
            return "oversold"
        elif rsi >= 50:
            return "bullish"
        else:
            return "bearish"


# ============================================================================
# MACD — MOVING AVERAGE CONVERGENCE DIVERGENCE
# ============================================================================

class MACDCalculator:
    """MACD with crossover detection, histogram analysis, and trend strength."""
    
    DEFAULT_FAST = 12
    DEFAULT_SLOW = 26
    DEFAULT_SIGNAL = 9
    
    @classmethod
    def calculate(
        cls,
        close_prices: pd.Series,
        fast_period: int = DEFAULT_FAST,
        slow_period: int = DEFAULT_SLOW,
        signal_period: int = DEFAULT_SIGNAL,
    ) -> IndicatorResult:
        """
        Calculate MACD with crossover signals.
        
        Args:
            close_prices: Series of closing prices
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)
        
        Returns:
            IndicatorResult with MACD data and crossover signals
        """
        if len(close_prices) < slow_period + signal_period:
            raise ValueError(f"Need at least {slow_period + signal_period} data points for MACD")
        
        # Calculate EMAs
        ema_fast = close_prices.ewm(span=fast_period).mean()
        ema_slow = close_prices.ewm(span=slow_period).mean()
        
        # Calculate MACD line and signal line
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line
        
        # Create result dataframe
        df = pd.DataFrame({
            'date': close_prices.index,
            'close': close_prices.values,
            'macd': macd_line.values,
            'signal': signal_line.values,
            'histogram': histogram.values,
        })
        
        # Detect crossovers
        signals = cls._detect_crossovers(df)
        
        current_macd = macd_line.iloc[-1]
        previous_macd = macd_line.iloc[-2] if len(macd_line) > 1 else current_macd
        
        return IndicatorResult(
            indicator_name="MACD",
            data=df,
            signals=signals,
            current_value=float(current_macd),
            previous_value=float(previous_macd),
            summary={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
                "current_histogram": float(histogram.iloc[-1]),
                "signal_count": len(signals),
                "momentum_direction": "bullish" if histogram.iloc[-1] > 0 else "bearish",
            }
        )
    
    @classmethod
    def _detect_crossovers(cls, df: pd.DataFrame) -> List[Signal]:
        """Detect MACD/Signal line crossovers (bullish and bearish)."""
        signals = []
        macd = df['macd'].values
        signal = df['signal'].values
        histogram = df['histogram'].values
        dates = df['date'].values
        
        for i in range(1, len(macd)):
            prev_hist = histogram[i-1]
            curr_hist = histogram[i]
            
            # Bullish crossover (MACD crosses above signal)
            if prev_hist <= 0 < curr_hist:
                signals.append(Signal(
                    signal_type="bullish_crossover",
                    indicator="MACD",
                    timestamp=pd.Timestamp(dates[i]),
                    value=float(macd[i]),
                    confidence=0.75 + min(0.2, abs(curr_hist) / 100),
                    evidence=f"MACD crossed above signal line (histogram: {curr_hist:.4f})",
                    risk_level="LOW",
                    governance_narrative="Momentum shifting positive; trend reversal potential.",
                ))
            
            # Bearish crossover (MACD crosses below signal)
            elif prev_hist >= 0 > curr_hist:
                signals.append(Signal(
                    signal_type="bearish_crossover",
                    indicator="MACD",
                    timestamp=pd.Timestamp(dates[i]),
                    value=float(macd[i]),
                    confidence=0.75 + min(0.2, abs(curr_hist) / 100),
                    evidence=f"MACD crossed below signal line (histogram: {curr_hist:.4f})",
                    risk_level="LOW",
                    governance_narrative="Momentum shifting negative; trend reversal potential.",
                ))
        
        return signals


# ============================================================================
# BOLLINGER BANDS
# ============================================================================

class BollingerBandsCalculator:
    """Bollinger Bands with squeeze detection, breakout signals, and volatility analysis."""
    
    DEFAULT_PERIOD = 20
    DEFAULT_STD_DEV = 2.0
    SQUEEZE_THRESHOLD = 0.5  # When band width < 50% of average
    
    @classmethod
    def calculate(
        cls,
        close_prices: pd.Series,
        period: int = DEFAULT_PERIOD,
        std_dev: float = DEFAULT_STD_DEV,
    ) -> IndicatorResult:
        """
        Calculate Bollinger Bands with squeeze and breakout detection.
        
        Args:
            close_prices: Series of closing prices
            period: MA period for bands (default 20)
            std_dev: Standard deviation multiplier (default 2.0)
        
        Returns:
            IndicatorResult with Bollinger Bands data and signals
        """
        if len(close_prices) < period:
            raise ValueError(f"Need at least {period} data points for Bollinger Bands")
        
        # Calculate middle band (SMA) and standard deviation
        sma = close_prices.rolling(window=period).mean()
        std = close_prices.rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        band_width = upper_band - lower_band
        
        # Create result dataframe
        df = pd.DataFrame({
            'date': close_prices.index,
            'close': close_prices.values,
            'sma': sma.values,
            'upper_band': upper_band.values,
            'lower_band': lower_band.values,
            'band_width': band_width.values,
        })
        
        # Detect signals
        signals = cls._detect_signals(df, period)
        
        current_bw = band_width.iloc[-1]
        avg_bw = band_width[~band_width.isna()].mean()
        
        return IndicatorResult(
            indicator_name="Bollinger Bands",
            data=df,
            signals=signals,
            current_value=float(close_prices.iloc[-1]),
            previous_value=float(close_prices.iloc[-2]) if len(close_prices) > 1 else float(close_prices.iloc[-1]),
            summary={
                "period": period,
                "std_dev": std_dev,
                "current_band_width": float(current_bw) if not pd.isna(current_bw) else None,
                "avg_band_width": float(avg_bw),
                "band_width_ratio": float(current_bw / avg_bw) if not pd.isna(current_bw) and avg_bw > 0 else None,
                "signal_count": len(signals),
                "squeeze_detected": float(current_bw / avg_bw) < cls.SQUEEZE_THRESHOLD if not pd.isna(current_bw) and avg_bw > 0 else False,
            }
        )
    
    @classmethod
    def _detect_signals(cls, df: pd.DataFrame, period: int) -> List[Signal]:
        """Detect BB signals: breakouts, squeezes, touch signals."""
        signals = []
        close = df['close'].values
        upper = df['upper_band'].values
        lower = df['lower_band'].values
        band_width = df['band_width'].values
        dates = df['date'].values
        
        avg_bw = np.nanmean(band_width[period:])
        
        for i in range(period, len(close)):
            curr_close = close[i]
            prev_close = close[i-1]
            curr_upper = upper[i]
            prev_upper = upper[i-1] if i > 0 else curr_upper
            curr_lower = lower[i]
            prev_lower = lower[i-1] if i > 0 else curr_lower
            curr_bw = band_width[i]
            
            # Upper band breakout
            if prev_close <= prev_upper and curr_close > curr_upper:
                signals.append(Signal(
                    signal_type="upper_breakout",
                    indicator="Bollinger Bands",
                    timestamp=pd.Timestamp(dates[i]),
                    value=float(curr_close),
                    confidence=0.7,
                    evidence=f"Price broke above upper band ({curr_close:.2f} > {curr_upper:.2f})",
                    risk_level="MEDIUM",
                    governance_narrative="Strong upside breakout; momentum confirmation needed.",
                ))
            
            # Lower band breakout
            elif prev_close >= prev_lower and curr_close < curr_lower:
                signals.append(Signal(
                    signal_type="lower_breakout",
                    indicator="Bollinger Bands",
                    timestamp=pd.Timestamp(dates[i]),
                    value=float(curr_close),
                    confidence=0.7,
                    evidence=f"Price broke below lower band ({curr_close:.2f} < {curr_lower:.2f})",
                    risk_level="MEDIUM",
                    governance_narrative="Strong downside breakout; momentum confirmation needed.",
                ))
            
            # Squeeze detection
            if curr_bw / avg_bw < cls.SQUEEZE_THRESHOLD:
                signals.append(Signal(
                    signal_type="squeeze",
                    indicator="Bollinger Bands",
                    timestamp=pd.Timestamp(dates[i]),
                    value=float(curr_bw / avg_bw),
                    confidence=0.6,
                    evidence=f"Band width contracted to {curr_bw/avg_bw:.1%} of average",
                    risk_level="MEDIUM",
                    governance_narrative="Low volatility period; breakout likely imminent.",
                ))
        
        return signals


# ============================================================================
# SUPPORT & RESISTANCE DETECTION
# ============================================================================

class SupportResistanceCalculator:
    """Identify support/resistance via swing highs/lows and fractal detection."""
    
    DEFAULT_LOOKBACK = 20  # Bars to look back for swing detection
    
    @classmethod
    def detect(
        cls,
        close_prices: pd.Series,
        high_prices: pd.Series,
        low_prices: pd.Series,
        lookback: int = DEFAULT_LOOKBACK,
    ) -> IndicatorResult:
        """
        Detect major support and resistance levels using swing analysis.
        
        Args:
            close_prices: Series of closing prices
            high_prices: Series of high prices
            low_prices: Series of low prices
            lookback: Bars to analyze for swings
        
        Returns:
            IndicatorResult with S&R levels and proximity signals
        """
        if len(close_prices) < lookback * 2:
            raise ValueError(f"Need at least {lookback * 2} data points for S&R detection")
        
        # Detect swing points
        swings = cls._find_swings(high_prices, low_prices, lookback)
        
        # Cluster nearby levels
        resistance_levels = cls._cluster_levels(swings['resistance'])
        support_levels = cls._cluster_levels(swings['support'])
        
        # Detect proximity and breakouts
        signals = cls._detect_proximity_signals(
            close_prices, resistance_levels, support_levels
        )
        
        current_price = close_prices.iloc[-1]
        
        return IndicatorResult(
            indicator_name="Support & Resistance",
            data=pd.DataFrame({
                'date': close_prices.index,
                'close': close_prices.values,
            }),
            signals=signals,
            current_value=float(current_price),
            previous_value=float(close_prices.iloc[-2]) if len(close_prices) > 1 else current_price,
            summary={
                "resistance_levels": [float(r) for r in sorted(resistance_levels, reverse=True)],
                "support_levels": [float(s) for s in sorted(support_levels)],
                "nearest_resistance": float(min(r for r in resistance_levels if r > current_price)) if any(r > current_price for r in resistance_levels) else None,
                "nearest_support": float(max(s for s in support_levels if s < current_price)) if any(s < current_price for s in support_levels) else None,
                "signal_count": len(signals),
            }
        )
    
    @staticmethod
    def _find_swings(highs: pd.Series, lows: pd.Series, lookback: int) -> Dict[str, List[float]]:
        """Find swing highs and lows."""
        swings = {'resistance': [], 'support': []}
        
        for i in range(lookback, len(highs) - lookback):
            # Swing high (local maximum)
            if highs.iloc[i] == highs.iloc[i-lookback:i+lookback+1].max():
                swings['resistance'].append(float(highs.iloc[i]))
            
            # Swing low (local minimum)
            if lows.iloc[i] == lows.iloc[i-lookback:i+lookback+1].min():
                swings['support'].append(float(lows.iloc[i]))
        
        return swings
    
    @staticmethod
    def _cluster_levels(values: List[float], tolerance: float = 0.02) -> List[float]:
        """Cluster nearby price levels within tolerance percentage."""
        if not values:
            return []
        
        values = sorted(set(values))
        clusters = []
        current_cluster = [values[0]]
        
        for value in values[1:]:
            # If within tolerance of cluster center, add to cluster
            if abs(value - np.mean(current_cluster)) / np.mean(current_cluster) < tolerance:
                current_cluster.append(value)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [value]
        
        clusters.append(np.mean(current_cluster))
        return clusters
    
    @classmethod
    def _detect_proximity_signals(
        cls,
        prices: pd.Series,
        resistance_levels: List[float],
        support_levels: List[float],
    ) -> List[Signal]:
        """Detect when price approaches or breaks levels."""
        signals = []
        current_price = prices.iloc[-1]
        proximity_threshold = 0.01  # 1%
        
        # Check resistance proximity
        for level in resistance_levels:
            if level > current_price:
                pct_to_level = (level - current_price) / current_price
                if pct_to_level < proximity_threshold:
                    confidence = 1.0 - pct_to_level / proximity_threshold
                    signals.append(Signal(
                        signal_type="approaching_resistance",
                        indicator="Support & Resistance",
                        timestamp=prices.index[-1],
                        value=float(level),
                        confidence=confidence * 0.8,
                        evidence=f"Price {pct_to_level:.1%} below resistance at {level:.2f}",
                        risk_level="MEDIUM",
                        governance_narrative=f"Approaching resistance; potential pullback risk near ${level:.2f}.",
                    ))
        
        # Check support proximity
        for level in support_levels:
            if level < current_price:
                pct_to_level = (current_price - level) / current_price
                if pct_to_level < proximity_threshold:
                    confidence = 1.0 - pct_to_level / proximity_threshold
                    signals.append(Signal(
                        signal_type="approaching_support",
                        indicator="Support & Resistance",
                        timestamp=prices.index[-1],
                        value=float(level),
                        confidence=confidence * 0.8,
                        evidence=f"Price {pct_to_level:.1%} above support at {level:.2f}",
                        risk_level="MEDIUM",
                        governance_narrative=f"Approaching support; potential bounce opportunity near ${level:.2f}.",
                    ))
        
        return signals


# ============================================================================
# TREND ANALYSIS
# ============================================================================

class TrendAnalyzer:
    """Identify trend direction, strength, and reversals."""
    
    DEFAULT_SHORT_MA = 20
    DEFAULT_MEDIUM_MA = 50
    DEFAULT_LONG_MA = 200
    
    @classmethod
    def analyze(
        cls,
        close_prices: pd.Series,
        short_period: int = DEFAULT_SHORT_MA,
        medium_period: int = DEFAULT_MEDIUM_MA,
        long_period: int = DEFAULT_LONG_MA,
    ) -> IndicatorResult:
        """
        Analyze trend using moving average alignment and crossovers.
        
        Args:
            close_prices: Series of closing prices
            short_period: Short MA period (default 20 - SMA)
            medium_period: Medium MA period (default 50 - SMA)
            long_period: Long MA period (default 200 - SMA)
        
        Returns:
            IndicatorResult with trend classification and signals
        """
        if len(close_prices) < long_period:
            raise ValueError(f"Need at least {long_period} data points for trend analysis")
        
        # Calculate moving averages
        sma_short = close_prices.rolling(window=short_period).mean()
        sma_medium = close_prices.rolling(window=medium_period).mean()
        sma_long = close_prices.rolling(window=long_period).mean()
        
        # Create result dataframe
        df = pd.DataFrame({
            'date': close_prices.index,
            'close': close_prices.values,
            'sma_20': sma_short.values,
            'sma_50': sma_medium.values,
            'sma_200': sma_long.values,
        })
        
        # Detect signals (crossovers)
        signals = cls._detect_signals(df)
        
        # Determine current trend
        current_close = close_prices.iloc[-1]
        current_sma_short = sma_short.iloc[-1]
        current_sma_medium = sma_medium.iloc[-1]
        current_sma_long = sma_long.iloc[-1]
        
        trend = cls._classify_trend(
            current_close, current_sma_short, current_sma_medium, current_sma_long
        )
        
        return IndicatorResult(
            indicator_name="Trend Analysis",
            data=df,
            signals=signals,
            current_value=float(current_close),
            previous_value=float(close_prices.iloc[-2]) if len(close_prices) > 1 else current_close,
            summary={
                "trend": trend,
                "current_price": float(current_close),
                "sma_20": float(current_sma_short),
                "sma_50": float(current_sma_medium),
                "sma_200": float(current_sma_long),
                "signal_count": len(signals),
                "alignment_quality": cls._alignment_quality(
                    current_close, current_sma_short, current_sma_medium, current_sma_long, trend
                ),
            }
        )
    
    @classmethod
    def _detect_signals(cls, df: pd.DataFrame) -> List[Signal]:
        """Detect MA crossovers (golden cross, death cross, etc.)."""
        signals = []
        sma_20 = df['sma_20'].values
        sma_50 = df['sma_50'].values
        sma_200 = df['sma_200'].values
        dates = df['date'].values
        
        for i in range(1, len(sma_20)):
            # Golden Cross: 20 crosses above 50
            if sma_20[i-1] <= sma_50[i-1] and sma_20[i] > sma_50[i]:
                signals.append(Signal(
                    signal_type="golden_cross",
                    indicator="Trend",
                    timestamp=pd.Timestamp(dates[i]),
                    value=float(sma_20[i]),
                    confidence=0.85,
                    evidence="SMA 20 crossed above SMA 50",
                    risk_level="LOW",
                    governance_narrative="Strong bullish alignment; trend reversal signal.",
                ))
            
            # Death Cross: 20 crosses below 50
            elif sma_20[i-1] >= sma_50[i-1] and sma_20[i] < sma_50[i]:
                signals.append(Signal(
                    signal_type="death_cross",
                    indicator="Trend",
                    timestamp=pd.Timestamp(dates[i]),
                    value=float(sma_20[i]),
                    confidence=0.85,
                    evidence="SMA 20 crossed below SMA 50",
                    risk_level="LOW",
                    governance_narrative="Strong bearish alignment; trend reversal signal.",
                ))
        
        return signals
    
    @staticmethod
    def _classify_trend(
        price: float, sma_20: float, sma_50: float, sma_200: float
    ) -> str:
        """Classify trend based on MA alignment."""
        # Count how many moving averages are below price
        below_count = sum([
            price > sma_20,
            price > sma_50,
            price > sma_200,
        ])
        
        if below_count >= 2:
            return "bullish"
        elif below_count == 0:
            return "bearish"
        else:
            return "mixed"
    
    @staticmethod
    def _alignment_quality(
        price: float, sma_20: float, sma_50: float, sma_200: float, trend: str
    ) -> float:
        """Score how well MAs are aligned with the trend (0.0 to 1.0)."""
        if trend == "bullish":
            # Perfect alignment: price > 20 > 50 > 200
            if price > sma_20 > sma_50 > sma_200:
                return 1.0
            elif price > sma_20 > sma_50:
                return 0.8
            elif price > sma_20:
                return 0.6
            else:
                return 0.3
        elif trend == "bearish":
            # Perfect alignment: price < 20 < 50 < 200
            if price < sma_20 < sma_50 < sma_200:
                return 1.0
            elif price < sma_20 < sma_50:
                return 0.8
            elif price < sma_20:
                return 0.6
            else:
                return 0.3
        else:
            return 0.5
