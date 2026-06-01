"""
technical_analysis_a5.py — Technical Analysis Agent

Handles RSI, MACD, Bollinger Bands, Support/Resistance, Trend Analysis,
and generates governance-compliant technical analysis reports.

Integrates with price data sources and provides Bloomberg-style technical
analysis with full explainability and auditability.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from src.agents.technical_indicators import (
    RSICalculator, MACDCalculator, BollingerBandsCalculator,
    SupportResistanceCalculator, TrendAnalyzer, IndicatorResult
)
from src.agents.technical_report_generator import TechnicalReportGenerator
from src.agents.live_data_tools import get_price_series_for_analysis
from src.agents.generate_dynamic_plot import generate_financial_plot
from src.agents.plot_store import GLOBAL_PLOT_IDS

logger = logging.getLogger(__name__)


# ============================================================================
# TECHNICAL ANALYSIS TOOLS
# ============================================================================

@tool
def calculate_rsi_signals(
    ticker: str,
    data_days: int = 252,
    config: Optional[RunnableConfig] = None,
) -> Dict[str, Any]:
    """
    Calculate RSI (Relative Strength Index) for a given ticker.
    
    Returns RSI values, overbought/oversold zones, and buy/sell signals.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "NVDA")
        data_days: Number of historical days to analyze (default 252 = 1 year)
        config: LangGraph config (for session tracking)
    
    Returns:
        Dictionary with RSI data, signals, and governance metadata
    """
    try:
        # Get historical price data
        price_data = get_price_series_for_analysis(ticker, days=data_days)
        if price_data is None or len(price_data) == 0:
            return {
                "error": f"Unable to fetch price data for {ticker}",
                "ticker": ticker,
            }
        
        close_prices = pd.Series(
            price_data['close'].values,
            index=pd.to_datetime(price_data['date'])
        )
        
        # Calculate RSI
        rsi_result = RSICalculator.calculate(close_prices)
        
        # Generate plot
        plot_data = {
            "title": f"RSI ({ticker})",
            "data": [
                {
                    "date": str(date),
                    "rsi": float(rsi),
                    "overbought": 70,
                    "oversold": 30,
                }
                for date, rsi in zip(
                    rsi_result.data['date'].values,
                    rsi_result.data['rsi'].values,
                )
            ]
        }
        
        session_id = config.configurable.get("session_id", "default") if config else "default"
        plot_id = f"rsi_{ticker}_{datetime.utcnow().timestamp()}"
        GLOBAL_PLOT_IDS[plot_id] = plot_data
        
        return {
            "ticker": ticker,
            "indicator": "RSI",
            "current_value": rsi_result.current_value,
            "zone": rsi_result.summary["current_zone"],
            "signals": [
                {
                    "type": s.signal_type,
                    "timestamp": s.timestamp.isoformat() if hasattr(s.timestamp, 'isoformat') else str(s.timestamp),
                    "value": float(s.value),
                    "confidence": float(s.confidence),
                    "evidence": s.evidence,
                    "governance_narrative": s.governance_narrative,
                }
                for s in rsi_result.signals
            ],
            "summary": rsi_result.summary,
            "plot_id": plot_id,
        }
    except Exception as e:
        logger.error(f"Error calculating RSI for {ticker}: {e}")
        return {
            "error": str(e),
            "ticker": ticker,
        }


@tool
def calculate_macd_crossovers(
    ticker: str,
    data_days: int = 252,
    config: Optional[RunnableConfig] = None,
) -> Dict[str, Any]:
    """
    Calculate MACD (Moving Average Convergence Divergence) for a given ticker.
    
    Detects bullish/bearish crossovers and momentum shifts.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "NVDA")
        data_days: Number of historical days to analyze (default 252 = 1 year)
        config: LangGraph config (for session tracking)
    
    Returns:
        Dictionary with MACD data, crossover signals, and governance metadata
    """
    try:
        # Get historical price data
        price_data = get_price_series_for_analysis(ticker, days=data_days)
        if price_data is None or len(price_data) == 0:
            return {
                "error": f"Unable to fetch price data for {ticker}",
                "ticker": ticker,
            }
        
        close_prices = pd.Series(
            price_data['close'].values,
            index=pd.to_datetime(price_data['date'])
        )
        
        # Calculate MACD
        macd_result = MACDCalculator.calculate(close_prices)
        
        # Generate plot data
        plot_data = {
            "title": f"MACD ({ticker})",
            "data": [
                {
                    "date": str(date),
                    "macd": float(macd),
                    "signal": float(signal),
                    "histogram": float(histogram),
                }
                for date, macd, signal, histogram in zip(
                    macd_result.data['date'].values,
                    macd_result.data['macd'].values,
                    macd_result.data['signal'].values,
                    macd_result.data['histogram'].values,
                )
            ]
        }
        
        session_id = config.configurable.get("session_id", "default") if config else "default"
        plot_id = f"macd_{ticker}_{datetime.utcnow().timestamp()}"
        GLOBAL_PLOT_IDS[plot_id] = plot_data
        
        return {
            "ticker": ticker,
            "indicator": "MACD",
            "current_macd": macd_result.current_value,
            "current_signal": macd_result.data['signal'].iloc[-1],
            "current_histogram": macd_result.summary["current_histogram"],
            "momentum_direction": macd_result.summary["momentum_direction"],
            "signals": [
                {
                    "type": s.signal_type,
                    "timestamp": s.timestamp.isoformat() if hasattr(s.timestamp, 'isoformat') else str(s.timestamp),
                    "value": float(s.value),
                    "confidence": float(s.confidence),
                    "evidence": s.evidence,
                    "governance_narrative": s.governance_narrative,
                }
                for s in macd_result.signals
            ],
            "summary": macd_result.summary,
            "plot_id": plot_id,
        }
    except Exception as e:
        logger.error(f"Error calculating MACD for {ticker}: {e}")
        return {
            "error": str(e),
            "ticker": ticker,
        }


@tool
def calculate_bollinger_bands(
    ticker: str,
    data_days: int = 252,
    config: Optional[RunnableConfig] = None,
) -> Dict[str, Any]:
    """
    Calculate Bollinger Bands for a given ticker.
    
    Detects volatility squeezes, breakouts, and band touches.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "NVDA")
        data_days: Number of historical days to analyze (default 252 = 1 year)
        config: LangGraph config (for session tracking)
    
    Returns:
        Dictionary with Bollinger Bands data, squeeze detection, and signals
    """
    try:
        # Get historical price data
        price_data = get_price_series_for_analysis(ticker, days=data_days)
        if price_data is None or len(price_data) == 0:
            return {
                "error": f"Unable to fetch price data for {ticker}",
                "ticker": ticker,
            }
        
        close_prices = pd.Series(
            price_data['close'].values,
            index=pd.to_datetime(price_data['date'])
        )
        
        # Calculate Bollinger Bands
        bb_result = BollingerBandsCalculator.calculate(close_prices)
        
        # Generate plot data
        plot_data = {
            "title": f"Bollinger Bands ({ticker})",
            "data": [
                {
                    "date": str(date),
                    "close": float(close),
                    "sma": float(sma),
                    "upper_band": float(upper),
                    "lower_band": float(lower),
                }
                for date, close, sma, upper, lower in zip(
                    bb_result.data['date'].values,
                    bb_result.data['close'].values,
                    bb_result.data['sma'].values,
                    bb_result.data['upper_band'].values,
                    bb_result.data['lower_band'].values,
                )
            ]
        }
        
        session_id = config.configurable.get("session_id", "default") if config else "default"
        plot_id = f"bb_{ticker}_{datetime.utcnow().timestamp()}"
        GLOBAL_PLOT_IDS[plot_id] = plot_data
        
        return {
            "ticker": ticker,
            "indicator": "Bollinger Bands",
            "current_price": float(bb_result.data['close'].iloc[-1]),
            "upper_band": float(bb_result.data['upper_band'].iloc[-1]),
            "lower_band": float(bb_result.data['lower_band'].iloc[-1]),
            "middle_band": float(bb_result.data['sma'].iloc[-1]),
            "band_width": float(bb_result.summary["current_band_width"]) if bb_result.summary["current_band_width"] else None,
            "squeeze_detected": bb_result.summary["squeeze_detected"],
            "signals": [
                {
                    "type": s.signal_type,
                    "timestamp": s.timestamp.isoformat() if hasattr(s.timestamp, 'isoformat') else str(s.timestamp),
                    "value": float(s.value),
                    "confidence": float(s.confidence),
                    "evidence": s.evidence,
                    "governance_narrative": s.governance_narrative,
                }
                for s in bb_result.signals
            ],
            "summary": bb_result.summary,
            "plot_id": plot_id,
        }
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands for {ticker}: {e}")
        return {
            "error": str(e),
            "ticker": ticker,
        }


@tool
def detect_support_resistance(
    ticker: str,
    data_days: int = 252,
    config: Optional[RunnableConfig] = None,
) -> Dict[str, Any]:
    """
    Identify support and resistance levels for a given ticker.
    
    Detects swing highs/lows and clusters levels.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "NVDA")
        data_days: Number of historical days to analyze (default 252 = 1 year)
        config: LangGraph config (for session tracking)
    
    Returns:
        Dictionary with support/resistance levels and proximity signals
    """
    try:
        # Get historical price data
        price_data = get_price_series_for_analysis(ticker, days=data_days)
        if price_data is None or len(price_data) == 0:
            return {
                "error": f"Unable to fetch price data for {ticker}",
                "ticker": ticker,
            }
        
        close = pd.Series(
            price_data['close'].values,
            index=pd.to_datetime(price_data['date'])
        )
        high = pd.Series(
            price_data['high'].values,
            index=pd.to_datetime(price_data['date'])
        )
        low = pd.Series(
            price_data['low'].values,
            index=pd.to_datetime(price_data['date'])
        )
        
        # Detect S&R
        sr_result = SupportResistanceCalculator.detect(close, high, low)
        
        return {
            "ticker": ticker,
            "indicator": "Support & Resistance",
            "current_price": float(close.iloc[-1]),
            "resistance_levels": sr_result.summary["resistance_levels"],
            "support_levels": sr_result.summary["support_levels"],
            "nearest_resistance": sr_result.summary["nearest_resistance"],
            "nearest_support": sr_result.summary["nearest_support"],
            "signals": [
                {
                    "type": s.signal_type,
                    "timestamp": s.timestamp.isoformat() if hasattr(s.timestamp, 'isoformat') else str(s.timestamp),
                    "level": float(s.value),
                    "confidence": float(s.confidence),
                    "evidence": s.evidence,
                    "governance_narrative": s.governance_narrative,
                }
                for s in sr_result.signals
            ],
            "summary": sr_result.summary,
        }
    except Exception as e:
        logger.error(f"Error detecting support/resistance for {ticker}: {e}")
        return {
            "error": str(e),
            "ticker": ticker,
        }


@tool
def analyze_trends(
    ticker: str,
    data_days: int = 252,
    config: Optional[RunnableConfig] = None,
) -> Dict[str, Any]:
    """
    Analyze trend direction and strength for a given ticker.
    
    Detects bullish/bearish/sideways trends and moving average crossovers.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "NVDA")
        data_days: Number of historical days to analyze (default 252 = 1 year)
        config: LangGraph config (for session tracking)
    
    Returns:
        Dictionary with trend classification and signal information
    """
    try:
        # Get historical price data
        price_data = get_price_series_for_analysis(ticker, days=data_days)
        if price_data is None or len(price_data) == 0:
            return {
                "error": f"Unable to fetch price data for {ticker}",
                "ticker": ticker,
            }
        
        close_prices = pd.Series(
            price_data['close'].values,
            index=pd.to_datetime(price_data['date'])
        )
        
        # Analyze trend
        trend_result = TrendAnalyzer.analyze(close_prices)
        
        # Generate plot data
        plot_data = {
            "title": f"Trend Analysis ({ticker})",
            "data": [
                {
                    "date": str(date),
                    "close": float(close),
                    "sma_20": float(sma_20),
                    "sma_50": float(sma_50),
                    "sma_200": float(sma_200),
                }
                for date, close, sma_20, sma_50, sma_200 in zip(
                    trend_result.data['date'].values,
                    trend_result.data['close'].values,
                    trend_result.data['sma_20'].values,
                    trend_result.data['sma_50'].values,
                    trend_result.data['sma_200'].values,
                )
            ]
        }
        
        session_id = config.configurable.get("session_id", "default") if config else "default"
        plot_id = f"trend_{ticker}_{datetime.utcnow().timestamp()}"
        GLOBAL_PLOT_IDS[plot_id] = plot_data
        
        return {
            "ticker": ticker,
            "indicator": "Trend Analysis",
            "trend": trend_result.summary["trend"],
            "trend_strength": trend_result.summary["alignment_quality"],
            "current_price": float(trend_result.summary["current_price"]),
            "sma_20": float(trend_result.summary["sma_20"]),
            "sma_50": float(trend_result.summary["sma_50"]),
            "sma_200": float(trend_result.summary["sma_200"]),
            "signals": [
                {
                    "type": s.signal_type,
                    "timestamp": s.timestamp.isoformat() if hasattr(s.timestamp, 'isoformat') else str(s.timestamp),
                    "value": float(s.value),
                    "confidence": float(s.confidence),
                    "evidence": s.evidence,
                    "governance_narrative": s.governance_narrative,
                }
                for s in trend_result.signals
            ],
            "summary": trend_result.summary,
            "plot_id": plot_id,
        }
    except Exception as e:
        logger.error(f"Error analyzing trends for {ticker}: {e}")
        return {
            "error": str(e),
            "ticker": ticker,
        }


@tool
def generate_technical_report(
    ticker: str,
    data_days: int = 252,
    config: Optional[RunnableConfig] = None,
) -> Dict[str, Any]:
    """
    Generate a comprehensive technical analysis report with governance integration.
    
    Produces a Bloomberg-style report with all indicators, signals, risk assessment,
    and explicit governance justification for every recommendation.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "NVDA")
        data_days: Number of historical days to analyze (default 252 = 1 year)
        config: LangGraph config (for session tracking)
    
    Returns:
        Dictionary containing full technical analysis report with governance metadata
    """
    try:
        # Get historical price data
        price_data = get_price_series_for_analysis(ticker, days=data_days)
        if price_data is None or len(price_data) == 0:
            return {
                "error": f"Unable to fetch price data for {ticker}",
                "ticker": ticker,
            }
        
        # Generate report
        generator = TechnicalReportGenerator(ticker=ticker)
        report = generator.generate_full_report(price_data)
        
        # Store report in session
        session_id = config.configurable.get("session_id", "default") if config else "default"
        report_id = f"ta_report_{ticker}_{datetime.utcnow().timestamp()}"
        GLOBAL_PLOT_IDS[report_id] = report
        
        return {
            "ticker": ticker,
            "report_id": report_id,
            "generated_at": report["metadata"]["generated_at"],
            "primary_recommendation": report["recommendations"]["primary_recommendation"],
            "confidence": report["recommendations"]["confidence"],
            "report_summary": {
                "trend": report["trend_analysis"]["current_trend"],
                "momentum": report["momentum_analysis"]["rsi"]["zone"],
                "volatility": report["volatility_analysis"]["volatility_interpretation"],
                "key_levels": report["recommendations"]["key_levels"],
            },
            "total_signals": report["governance_metadata"]["total_signals"],
            "avg_signal_confidence": report["governance_metadata"]["average_signal_confidence"],
            "full_report_data": report,
        }
    except Exception as e:
        logger.error(f"Error generating technical report for {ticker}: {e}")
        return {
            "error": str(e),
            "ticker": ticker,
        }


# ============================================================================
# AGENT TOOLS LIST (for registration with orchestrator)
# ============================================================================

TECHNICAL_ANALYSIS_TOOLS = [
    calculate_rsi_signals,
    calculate_macd_crossovers,
    calculate_bollinger_bands,
    detect_support_resistance,
    analyze_trends,
    generate_technical_report,
]
