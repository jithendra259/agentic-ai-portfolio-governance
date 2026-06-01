"""
Phase 2 Stub Agents

Placeholder implementations for agents being completed in Phase 2.
These stubs return valid structures to allow system testing.
Will be replaced with full implementations.
"""

from typing import Any, Dict

from src.agents.agent_base import (
    AgentConfig,
    AgentOutput,
    AgentType,
    SpecializedAgent,
    TaskDefinition,
)


class TechnicalAnalysisAgent(SpecializedAgent):
    """
    Technical Analysis Agent - STUB FOR PHASE 2
    
    Will calculate technical indicators when fully implemented.
    Currently returns valid structure for testing.
    """
    
    async def execute(
        self,
        task: TaskDefinition,
        context: Dict[str, Any],
    ) -> AgentOutput:
        """Execute technical analysis (stub)."""
        output = self._create_output(task.inputs.get("request_id", ""))
        
        output.mark_success({
            "indicators": {
                "sma_20": 150.25,
                "sma_50": 148.75,
                "sma_200": 145.50,
                "rsi": 65.4,
                "macd": {"macd": 2.5, "signal": 2.3, "histogram": 0.2},
            },
            "signals": [
                {
                    "type": "GOLDEN_CROSS",
                    "strength": 0.85,
                    "evidence": "SMA 20 crossed above SMA 50"
                }
            ],
        })
        
        output.add_evidence("SMA 20 above SMA 50 (bullish)")
        output.add_evidence("RSI above 60 (momentum)")
        output.confidence = 0.75
        
        return output


class RegimeDetectionAgent(SpecializedAgent):
    """
    Regime Detection Agent - STUB FOR PHASE 2
    
    Will detect market regimes when fully implemented.
    Currently returns valid structure for testing.
    """
    
    async def execute(
        self,
        task: TaskDefinition,
        context: Dict[str, Any],
    ) -> AgentOutput:
        """Detect market regime (stub)."""
        output = self._create_output(task.inputs.get("request_id", ""))
        
        output.mark_success({
            "regime": "BULL_MARKET",
            "confidence": 0.82,
            "primary_evidence": [
                "SMA alignment bullish",
                "Price above 200-day MA",
                "Positive momentum",
            ],
            "secondary_evidence": [
                "Volume increasing",
                "Trend strength high",
            ],
            "regime_duration_days": 45,
            "regime_strength": "STRONG",
        })
        
        output.add_evidence("Market in strong bull trend for 45 days")
        output.confidence = 0.82
        
        return output


class InstabilityAnalysisAgent(SpecializedAgent):
    """
    Instability Analysis Agent - STUB FOR PHASE 2
    
    Will compute instability metrics when fully implemented.
    Currently returns valid structure for testing.
    """
    
    async def execute(
        self,
        task: TaskDefinition,
        context: Dict[str, Any],
    ) -> AgentOutput:
        """Analyze instability (stub)."""
        output = self._create_output(task.inputs.get("request_id", ""))
        
        output.mark_success({
            "composite_instability_score": 0.35,
            "level": "LOW",
            "components": {
                "volatility_drift": {
                    "score": 0.3,
                    "direction": "STABLE",
                    "magnitude": 0.02,
                },
                "correlation_drift": {
                    "score": 0.4,
                    "changed_pairs": 2,
                },
                "covariance_drift": {
                    "score": 0.35,
                    "frobenius_norm": 0.18,
                },
            },
            "risk_warning": None,
        })
        
        output.add_evidence("Low instability environment (0.35/1.0)")
        output.confidence = 0.70
        
        return output


class PortfolioOptimizationAgent(SpecializedAgent):
    """
    Portfolio Optimization Agent - STUB FOR PHASE 2
    
    Will optimize allocations when fully implemented.
    Currently returns valid structure for testing.
    """
    
    async def execute(
        self,
        task: TaskDefinition,
        context: Dict[str, Any],
    ) -> AgentOutput:
        """Optimize portfolio (stub)."""
        output = self._create_output(task.inputs.get("request_id", ""))
        
        output.mark_success({
            "current_allocation": {
                "AAPL": 0.25,
                "MSFT": 0.20,
                "GOOGL": 0.15,
                "CASH": 0.40,
            },
            "recommended_allocation": {
                "AAPL": 0.30,
                "MSFT": 0.25,
                "GOOGL": 0.15,
                "CASH": 0.30,
            },
            "changes": [
                {
                    "ticker": "AAPL",
                    "current_weight": 0.25,
                    "recommended_weight": 0.30,
                    "change_amount": 0.05,
                    "rationale": "Bullish technical setup with strong momentum",
                },
            ],
            "expected_return": 0.12,
            "expected_volatility": 0.18,
            "sharpe_ratio": 0.67,
            "rebalance_needed": True,
            "rebalance_urgency": "MEDIUM",
        })
        
        output.add_evidence("Optimization completed with 5% allocation increase to AAPL")
        output.confidence = 0.68
        
        return output


class GovernanceAgent(SpecializedAgent):
    """
    Governance Agent - STUB FOR PHASE 2
    
    Will validate governance rules when fully implemented.
    Currently returns valid structure for testing.
    """
    
    async def execute(
        self,
        task: TaskDefinition,
        context: Dict[str, Any],
    ) -> AgentOutput:
        """Validate governance rules (stub)."""
        output = self._create_output(task.inputs.get("request_id", ""))
        
        output.mark_success({
            "status": "APPROVED",
            "validation_results": [
                {
                    "rule": "Risk Level <= Policy Max",
                    "status": "PASS",
                    "message": "Portfolio risk 0.18 < policy max 0.25",
                },
                {
                    "rule": "Allocation % <= Policy Limit",
                    "status": "PASS",
                    "message": "Max allocation 30% < policy limit 35%",
                },
                {
                    "rule": "Concentration Rules",
                    "status": "PASS",
                    "message": "No single holding > 50%",
                },
            ],
            "compliance_score": 0.95,
            "required_approvals": [],
            "escalation_level": "NONE",
        })
        
        output.add_evidence("All governance rules passed")
        output.confidence = 0.98
        
        return output


class ExplainabilityAgent(SpecializedAgent):
    """
    Explainability Agent - STUB FOR PHASE 2
    
    Will generate explanations when fully implemented.
    Currently returns valid structure for testing.
    """
    
    async def execute(
        self,
        task: TaskDefinition,
        context: Dict[str, Any],
    ) -> AgentOutput:
        """Generate explanations (stub)."""
        output = self._create_output(task.inputs.get("request_id", ""))
        
        output.mark_success({
            "executive_summary": (
                "AAPL shows strong technical bullish setup with golden cross and "
                "positive momentum. Recommend increasing allocation by 5% in a bull market "
                "regime with low instability."
            ),
            "technical_explanation": {
                "key_indicators": [
                    {
                        "name": "SMA 20/50 Golden Cross",
                        "value": "150.25 > 148.75",
                        "contribution_pct": 35.0,
                        "interpretation": "Strong bullish signal",
                    },
                    {
                        "name": "RSI (14)",
                        "value": 65.4,
                        "contribution_pct": 25.0,
                        "interpretation": "Strong momentum, approaching overbought",
                    },
                ],
                "regime_context": "Bull market with strong trend",
                "instability_impact": "Low instability favors growth positioning",
            },
            "beginner_explanation": (
                "The stock shows signs that experienced traders look for: "
                "the short-term trend (20-day) has crossed above the medium-term trend (50-day), "
                "which historically precedes periods of higher returns. The momentum indicator "
                "is strong, suggesting continued upward movement. The broader market is in a "
                "bull trend with relatively calm conditions, making this a good time to increase "
                "growth positions."
            ),
            "confidence_breakdown": {
                "data_quality": 0.95,
                "indicator_agreement": 0.85,
                "regime_clarity": 0.82,
                "governance_certainty": 0.98,
                "overall": 0.80,
            },
        })
        
        output.add_evidence("Comprehensive explanation generated across technical and beginner levels")
        output.confidence = 0.85
        
        return output
