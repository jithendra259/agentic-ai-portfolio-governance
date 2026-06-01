"""
technical_report_generator.py — AI Technical Analysis Report Generator

Produces governance-compliant technical analysis reports with:
- Indicator evidence (which signals triggered, their strength)
- Confidence scoring (0-100%)
- Governance narrative (why this signal matters for portfolio)
- Risk assessment (volatility, drawdown potential)
- Explainability (citable sources, calculation methods)
"""

import logging
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple, Any

import pandas as pd

from src.agents.technical_indicators import (
    RSICalculator, MACDCalculator, BollingerBandsCalculator,
    SupportResistanceCalculator, TrendAnalyzer,
    IndicatorResult, Signal
)

logger = logging.getLogger(__name__)


class TechnicalReportGenerator:
    """Generate comprehensive technical analysis reports with governance integration."""
    
    def __init__(self, ticker: str, time_period: str = "1y"):
        """
        Initialize report generator.
        
        Args:
            ticker: Stock ticker symbol
            time_period: Data period ("1m", "3m", "6m", "1y", "5y", etc.)
        """
        self.ticker = ticker
        self.time_period = time_period
        self.generated_at = datetime.utcnow().isoformat()
    
    def generate_full_report(
        self,
        price_data: pd.DataFrame,
        include_sections: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete technical analysis report.
        
        Args:
            price_data: DataFrame with columns: date, close, high, low, volume
            include_sections: Report sections to include
        
        Returns:
            Dictionary containing report data, suitable for JSON serialization
        """
        if include_sections is None:
            include_sections = [
                "market_summary",
                "technical_indicators",
                "trend_analysis",
                "momentum_analysis",
                "volatility_analysis",
                "support_resistance",
                "signals_bullish",
                "signals_bearish",
                "risk_assessment",
                "recommendations",
                "governance_metadata",
            ]
        
        # Ensure price data is sorted
        price_data = price_data.sort_values('date')
        close_prices = pd.Series(price_data['close'].values, index=price_data['date'])
        
        # Calculate all indicators
        try:
            rsi_result = RSICalculator.calculate(close_prices)
            macd_result = MACDCalculator.calculate(close_prices)
            bb_result = BollingerBandsCalculator.calculate(close_prices)
            trend_result = TrendAnalyzer.analyze(close_prices)
            
            high_prices = pd.Series(price_data['high'].values, index=price_data['date'])
            low_prices = pd.Series(price_data['low'].values, index=price_data['date'])
            sr_result = SupportResistanceCalculator.detect(close_prices, high_prices, low_prices)
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise
        
        # Build report sections
        report = {
            "metadata": {
                "ticker": self.ticker,
                "generated_at": self.generated_at,
                "time_period": self.time_period,
                "data_points": len(price_data),
                "date_range": {
                    "start": price_data['date'].min().isoformat() if hasattr(price_data['date'].min(), 'isoformat') else str(price_data['date'].min()),
                    "end": price_data['date'].max().isoformat() if hasattr(price_data['date'].max(), 'isoformat') else str(price_data['date'].max()),
                }
            }
        }
        
        if "market_summary" in include_sections:
            report["market_summary"] = self._build_market_summary(price_data)
        
        if "technical_indicators" in include_sections:
            report["technical_indicators"] = {
                "rsi": self._indicator_to_dict(rsi_result),
                "macd": self._indicator_to_dict(macd_result),
                "bollinger_bands": self._indicator_to_dict(bb_result),
                "moving_averages": self._ma_summary(trend_result),
            }
        
        if "trend_analysis" in include_sections:
            report["trend_analysis"] = self._build_trend_analysis(trend_result)
        
        if "momentum_analysis" in include_sections:
            report["momentum_analysis"] = self._build_momentum_analysis(rsi_result, macd_result)
        
        if "volatility_analysis" in include_sections:
            report["volatility_analysis"] = self._build_volatility_analysis(price_data, bb_result)
        
        if "support_resistance" in include_sections:
            report["support_resistance"] = self._build_sr_analysis(sr_result)
        
        # Aggregate all signals
        all_signals = (rsi_result.signals + macd_result.signals + 
                      bb_result.signals + sr_result.signals + trend_result.signals)
        
        if "signals_bullish" in include_sections:
            bullish_signals = [s for s in all_signals if s.signal_type in 
                              ["oversold_entry", "bullish_crossover", "golden_cross", 
                               "lower_breakout", "approaching_support"]]
            report["signals_bullish"] = self._signals_to_dicts(bullish_signals)
        
        if "signals_bearish" in include_sections:
            bearish_signals = [s for s in all_signals if s.signal_type in 
                              ["overbought_entry", "bearish_crossover", "death_cross",
                               "upper_breakout", "approaching_resistance"]]
            report["signals_bearish"] = self._signals_to_dicts(bearish_signals)
        
        if "risk_assessment" in include_sections:
            report["risk_assessment"] = self._build_risk_assessment(price_data, trend_result)
        
        if "recommendations" in include_sections:
            report["recommendations"] = self._build_recommendations(all_signals, trend_result)
        
        if "governance_metadata" in include_sections:
            report["governance_metadata"] = self._build_governance_metadata(all_signals)
        
        return report
    
    # ========================================================================
    # REPORT SECTION BUILDERS
    # ========================================================================
    
    @staticmethod
    def _build_market_summary(price_data: pd.DataFrame) -> Dict[str, Any]:
        """Build market summary section."""
        current_price = price_data['close'].iloc[-1]
        prev_price = price_data['close'].iloc[-2] if len(price_data) > 1 else current_price
        open_price = price_data['close'].iloc[0]
        high = price_data['high'].max()
        low = price_data['low'].min()
        
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
        
        period_change = current_price - open_price
        period_change_pct = (period_change / open_price) * 100 if open_price != 0 else 0
        
        return {
            "current_price": float(current_price),
            "previous_close": float(prev_price),
            "price_change": float(change),
            "price_change_pct": float(change_pct),
            "period_change": float(period_change),
            "period_change_pct": float(period_change_pct),
            "52week_high": float(high),
            "52week_low": float(low),
            "price_range": float(high - low),
        }
    
    @staticmethod
    def _build_trend_analysis(trend_result: IndicatorResult) -> Dict[str, Any]:
        """Build trend analysis section."""
        summary = trend_result.summary
        return {
            "current_trend": summary["trend"],
            "trend_strength": summary["alignment_quality"],
            "moving_averages": {
                "sma_20": float(summary["sma_20"]),
                "sma_50": float(summary["sma_50"]),
                "sma_200": float(summary["sma_200"]),
            },
            "interpretation": TechnicalReportGenerator._interpret_trend(summary),
            "recent_signals": [
                {
                    "type": s.signal_type,
                    "date": s.timestamp.isoformat() if hasattr(s.timestamp, 'isoformat') else str(s.timestamp),
                    "evidence": s.evidence,
                    "governance_narrative": s.governance_narrative,
                } for s in trend_result.signals
            ]
        }
    
    @staticmethod
    def _build_momentum_analysis(rsi_result: IndicatorResult, macd_result: IndicatorResult) -> Dict[str, Any]:
        """Build momentum analysis section."""
        return {
            "rsi": {
                "current_value": rsi_result.current_value,
                "zone": rsi_result.summary["current_zone"],
                "interpretation": TechnicalReportGenerator._interpret_rsi(rsi_result.current_value),
            },
            "macd": {
                "macd_line": float(macd_result.data['macd'].iloc[-1]),
                "signal_line": float(macd_result.data['signal'].iloc[-1]),
                "histogram": macd_result.summary["current_histogram"],
                "momentum_direction": macd_result.summary["momentum_direction"],
            },
            "recent_signals": [
                {
                    "indicator": s.indicator,
                    "type": s.signal_type,
                    "value": float(s.value),
                    "confidence": float(s.confidence),
                    "evidence": s.evidence,
                } for s in rsi_result.signals + macd_result.signals
            ]
        }
    
    @staticmethod
    def _build_volatility_analysis(price_data: pd.DataFrame, bb_result: IndicatorResult) -> Dict[str, Any]:
        """Build volatility analysis section."""
        returns = price_data['close'].pct_change()
        volatility = returns.std() * 100  # As percentage
        
        summary = bb_result.summary
        band_width_ratio = summary.get("band_width_ratio", 1.0)
        
        return {
            "realized_volatility_pct": float(volatility),
            "volatility_interpretation": TechnicalReportGenerator._interpret_volatility(volatility),
            "bollinger_bands": {
                "band_width": float(summary["current_band_width"]) if summary["current_band_width"] else 0,
                "band_width_ratio": float(band_width_ratio) if band_width_ratio else 1.0,
                "squeeze_detected": summary["squeeze_detected"],
            },
            "interpretation": "High volatility period" if band_width_ratio > 1.2 else "Low volatility squeeze" if band_width_ratio < 0.5 else "Normal volatility",
        }
    
    @staticmethod
    def _build_sr_analysis(sr_result: IndicatorResult) -> Dict[str, Any]:
        """Build support/resistance analysis."""
        summary = sr_result.summary
        return {
            "resistance_levels": summary["resistance_levels"],
            "support_levels": summary["support_levels"],
            "nearest_resistance": summary["nearest_resistance"],
            "nearest_support": summary["nearest_support"],
            "proximity_signals": [
                {
                    "type": s.signal_type,
                    "level": float(s.value),
                    "governance_narrative": s.governance_narrative,
                } for s in sr_result.signals
            ]
        }
    
    @staticmethod
    def _build_risk_assessment(price_data: pd.DataFrame, trend_result: IndicatorResult) -> Dict[str, Any]:
        """Build risk assessment section."""
        # Calculate max drawdown
        cumulative = (1 + price_data['close'].pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Volatility-based risk
        volatility = price_data['close'].pct_change().std() * 100
        
        trend = trend_result.summary["trend"]
        
        return {
            "max_drawdown_pct": float(max_drawdown),
            "current_volatility_pct": float(volatility),
            "trend_alignment": trend_result.summary["alignment_quality"],
            "risk_level": TechnicalReportGenerator._assess_risk_level(trend, volatility, max_drawdown),
            "risk_factors": TechnicalReportGenerator._identify_risk_factors(trend, volatility),
        }
    
    @staticmethod
    def _build_recommendations(all_signals: List[Signal], trend_result: IndicatorResult) -> Dict[str, Any]:
        """Build actionable recommendations."""
        bullish_signals = [s for s in all_signals if s.signal_type in 
                          ["oversold_entry", "bullish_crossover", "golden_cross", "lower_breakout"]]
        bearish_signals = [s for s in all_signals if s.signal_type in 
                          ["overbought_entry", "bearish_crossover", "death_cross", "upper_breakout"]]
        
        trend = trend_result.summary["trend"]
        
        recommendation = "NEUTRAL"
        if len(bullish_signals) > len(bearish_signals) and trend in ["bullish", "mixed"]:
            recommendation = "BUY"
        elif len(bearish_signals) > len(bullish_signals) and trend in ["bearish", "mixed"]:
            recommendation = "SELL"
        elif trend == "bullish":
            recommendation = "BUY"
        elif trend == "bearish":
            recommendation = "SELL"
        
        return {
            "primary_recommendation": recommendation,
            "confidence": float(
                min(1.0, (len(bullish_signals) if recommendation == "BUY" else len(bearish_signals)) / 3.0)
            ),
            "rationale": TechnicalReportGenerator._rationale_for_recommendation(
                recommendation, bullish_signals, bearish_signals, trend
            ),
            "key_levels": TechnicalReportGenerator._key_levels_for_action(
                recommendation, trend_result
            ),
        }
    
    @staticmethod
    def _build_governance_metadata(all_signals: List[Signal]) -> Dict[str, Any]:
        """Build governance and explainability metadata."""
        signal_confidence_avg = sum(s.confidence for s in all_signals) / len(all_signals) if all_signals else 0.5
        
        signals_by_indicator = {}
        for signal in all_signals:
            if signal.indicator not in signals_by_indicator:
                signals_by_indicator[signal.indicator] = []
            signals_by_indicator[signal.indicator].append({
                "type": signal.signal_type,
                "confidence": float(signal.confidence),
                "evidence": signal.evidence,
                "governance_narrative": signal.governance_narrative,
            })
        
        return {
            "total_signals": len(all_signals),
            "average_signal_confidence": float(signal_confidence_avg),
            "signals_by_indicator": signals_by_indicator,
            "calculation_methods": {
                "rsi": "14-period RSI with overbought (>70) and oversold (<30) zones",
                "macd": "12/26/9 MACD with signal line crossover detection",
                "bollinger_bands": "20-period SMA with 2 std dev bands, squeeze detection",
                "support_resistance": "Swing high/low detection with level clustering",
                "trend": "SMA 20/50/200 alignment and golden/death cross detection",
            },
            "governance_framework": {
                "all_signals_evidence_based": True,
                "confidence_scored": True,
                "explainability_required": True,
                "auditability": "Full calculation history available for review",
            },
        }
    
    # ========================================================================
    # INTERPRETATION & ANALYSIS HELPERS
    # ========================================================================
    
    @staticmethod
    def _interpret_trend(summary: Dict) -> str:
        """Interpret trend summary."""
        trend = summary["trend"]
        strength = summary["alignment_quality"]
        
        if trend == "bullish":
            if strength > 0.9:
                return "Strong uptrend with excellent MA alignment (20 > 50 > 200)"
            elif strength > 0.6:
                return "Bullish trend with good MA alignment"
            else:
                return "Weakening bullish trend"
        elif trend == "bearish":
            if strength > 0.9:
                return "Strong downtrend with excellent MA alignment (20 < 50 < 200)"
            elif strength > 0.6:
                return "Bearish trend with good MA alignment"
            else:
                return "Weakening bearish trend"
        else:
            return "Mixed signals; no clear trend direction"
    
    @staticmethod
    def _interpret_rsi(rsi_value: float) -> str:
        """Interpret RSI value."""
        if rsi_value >= 70:
            return "Overbought (>70); pullback risk"
        elif rsi_value >= 50:
            return "Bullish momentum"
        elif rsi_value > 30:
            return "Neutral momentum"
        else:
            return "Oversold (<30); bounce opportunity"
    
    @staticmethod
    def _interpret_volatility(volatility: float) -> str:
        """Interpret volatility level."""
        if volatility > 30:
            return "High volatility"
        elif volatility > 15:
            return "Normal volatility"
        else:
            return "Low volatility (potential squeeze)"
    
    @staticmethod
    def _assess_risk_level(trend: str, volatility: float, max_drawdown: float) -> str:
        """Assess overall risk level."""
        risk_score = 0
        
        if trend == "bearish":
            risk_score += 2
        elif trend == "bullish":
            risk_score -= 1
        
        if volatility > 25:
            risk_score += 2
        elif volatility < 10:
            risk_score -= 1
        
        if max_drawdown < -20:
            risk_score += 2
        
        if risk_score >= 3:
            return "HIGH"
        elif risk_score >= 1:
            return "MEDIUM"
        else:
            return "LOW"
    
    @staticmethod
    def _identify_risk_factors(trend: str, volatility: float) -> List[str]:
        """Identify specific risk factors."""
        factors = []
        
        if trend == "bearish":
            factors.append("Bearish trend increases downside risk")
        
        if volatility > 25:
            factors.append("High volatility increases whipsaw risk")
        
        if volatility < 10:
            factors.append("Low volatility may indicate complacency before breakout")
        
        return factors
    
    @staticmethod
    def _rationale_for_recommendation(
        recommendation: str,
        bullish_signals: List[Signal],
        bearish_signals: List[Signal],
        trend: str,
    ) -> str:
        """Build rationale for recommendation."""
        if recommendation == "BUY":
            return f"Bullish signals ({len(bullish_signals)}) outnumber bearish ({len(bearish_signals)}) with {trend} trend"
        elif recommendation == "SELL":
            return f"Bearish signals ({len(bearish_signals)}) outnumber bullish ({len(bullish_signals)}) with {trend} trend"
        else:
            return "Mixed signals with uncertain direction; recommend caution"
    
    @staticmethod
    def _key_levels_for_action(recommendation: str, trend_result: IndicatorResult) -> Dict[str, float]:
        """Identify key price levels for decision-making."""
        summary = trend_result.summary
        
        if recommendation == "BUY":
            return {
                "entry": float(summary["current_price"]),
                "stop_loss": float(summary["current_price"] * 0.95),
                "target_1": float(summary["current_price"] * 1.05),
                "target_2": float(summary["current_price"] * 1.10),
            }
        elif recommendation == "SELL":
            return {
                "entry": float(summary["current_price"]),
                "stop_loss": float(summary["current_price"] * 1.05),
                "target_1": float(summary["current_price"] * 0.95),
                "target_2": float(summary["current_price"] * 0.90),
            }
        else:
            return {}
    
    @staticmethod
    def _ma_summary(trend_result: IndicatorResult) -> Dict[str, Any]:
        """Summarize moving averages."""
        summary = trend_result.summary
        return {
            "sma_20": float(summary["sma_20"]),
            "sma_50": float(summary["sma_50"]),
            "sma_200": float(summary["sma_200"]),
            "current_price": float(summary["current_price"]),
        }
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    @staticmethod
    def _indicator_to_dict(result: IndicatorResult) -> Dict[str, Any]:
        """Convert IndicatorResult to dictionary."""
        return {
            "current_value": result.current_value,
            "previous_value": result.previous_value,
            "summary": result.summary,
            "signals": [TechnicalReportGenerator._signal_to_dict(s) for s in result.signals],
        }
    
    @staticmethod
    def _signal_to_dict(signal: Signal) -> Dict[str, Any]:
        """Convert Signal to dictionary."""
        return {
            "type": signal.signal_type,
            "indicator": signal.indicator,
            "timestamp": signal.timestamp.isoformat() if hasattr(signal.timestamp, 'isoformat') else str(signal.timestamp),
            "value": float(signal.value),
            "confidence": float(signal.confidence),
            "evidence": signal.evidence,
            "risk_level": signal.risk_level,
            "governance_narrative": signal.governance_narrative,
        }
    
    @staticmethod
    def _signals_to_dicts(signals: List[Signal]) -> List[Dict[str, Any]]:
        """Convert list of signals to dictionaries."""
        return [TechnicalReportGenerator._signal_to_dict(s) for s in signals]
