"""
Response Agent

Assembles final user-facing response from verified outputs.
Constructs structured response with all required sections.
"""

import logging
from typing import Any, Dict

from src.agents.agent_base import (
    AgentConfig,
    AgentOutput,
    AgentType,
    SpecializedAgent,
    TaskDefinition,
)

logger = logging.getLogger(__name__)


class ResponseAgent(SpecializedAgent):
    """
    Response Agent: Constructs final response.
    
    Responsibilities:
    - Assemble verified outputs into user response
    - Provide executive summary
    - Include all technical findings
    - Highlight governance compliance
    - Present recommendation with confidence
    - List identified risks
    - Provide audit trail reference
    
    Input:
    - All verified agent outputs
    - Verification report
    - Plan from planner
    
    Output:
    - Complete user response with structure:
      1. Summary
      2. Key Findings
      3. Technical Analysis
      4. Regime Analysis
      5. Instability Analysis
      6. Governance Review
      7. Recommendation
      8. Risks
      9. Confidence Score
      10. Explanation
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize response agent."""
        super().__init__(config)
    
    async def execute(
        self,
        task: TaskDefinition,
        context: Dict[str, Any],
    ) -> AgentOutput:
        """
        Construct final response.
        
        Args:
            task: Task definition with all context
            context: Execution context with verified outputs
        
        Returns:
            AgentOutput with complete response
        """
        output = self._create_output(task.inputs.get("request_id", ""))
        
        try:
            # Validate inputs
            if not self._validate_inputs(task.inputs, ["request_id"]):
                output.mark_failed("Missing required input: request_id", "INVALID_INPUT")
                return output
            
            request_id = task.inputs["request_id"]
            
            self._log_info(f"Constructing response for request {request_id}")
            
            # Build response from context
            response = self._build_response(context, task.inputs)
            
            output.mark_success(response)
            output.add_evidence("Response assembled from verified outputs")
            output.confidence = 0.95  # Response construction is high confidence
            
        except Exception as e:
            self._log_error(f"Response construction failed: {str(e)}")
            output.mark_failed(f"Response error: {str(e)}", "RESPONSE_ERROR")
        
        return output
    
    def _build_response(
        self,
        context: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build complete response.
        
        Returns:
            Structured response dictionary
        """
        from uuid import uuid4
        from datetime import datetime
        
        # Extract outputs from context
        plan_output = context.get("planner", {})
        data_output = context.get("data", {})
        ta_output = context.get("technical_analysis", {})
        regime_output = context.get("regime_detection", {})
        instability_output = context.get("instability_analysis", {})
        portfolio_output = context.get("portfolio_optimization", {})
        governance_output = context.get("governance", {})
        explainability_output = context.get("explainability", {})
        verification_output = context.get("verification", {})
        
        # Build response structure
        response = {
            "response_id": str(uuid4()),
            "request_id": inputs.get("request_id", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "user_summary": self._build_summary(
                plan_output,
                ta_output,
                regime_output,
                governance_output,
            ),
            "findings": {
                "technical_analysis": ta_output.get("data", {}),
                "regime_analysis": regime_output.get("data", {}),
                "instability_analysis": instability_output.get("data", {}),
                "portfolio_recommendation": portfolio_output.get("data", {}),
            },
            "governance_review": {
                "status": governance_output.get("data", {}).get("status", "UNKNOWN"),
                "details": governance_output.get("data", {}).get("details", ""),
            },
            "recommendation": {
                "action": self._extract_recommendation(portfolio_output, governance_output),
                "confidence": self._calculate_confidence(context),
                "key_levels": self._extract_key_levels(ta_output, portfolio_output),
            },
            "risks": self._extract_risks(ta_output, instability_output),
            "confidence_score": self._calculate_confidence(context),
            "explanation": explainability_output.get("data", {}).get("explanation", ""),
            "audit_trail": {
                "plan_id": plan_output.get("data", {}).get("plan_id", ""),
                "task_results": self._build_task_summary(context),
                "verification_status": verification_output.get("data", {}).get("overall_status", "UNKNOWN"),
            },
        }
        
        return response
    
    def _build_summary(
        self,
        plan: Dict[str, Any],
        ta: Dict[str, Any],
        regime: Dict[str, Any],
        governance: Dict[str, Any],
    ) -> str:
        """Build executive summary."""
        parts = []
        
        # Intent
        if plan and plan.get("data"):
            intent = plan["data"].get("user_intent", "")
            if intent:
                parts.append(f"Analysis Type: {intent}")
        
        # Regime
        if regime and regime.get("data"):
            regime_label = regime["data"].get("regime", "Unknown")
            parts.append(f"Market Regime: {regime_label}")
        
        # Governance
        if governance and governance.get("data"):
            status = governance["data"].get("status", "UNKNOWN")
            parts.append(f"Governance Status: {status}")
        
        # Technical summary
        if ta and ta.get("data"):
            signals = ta["data"].get("signals", [])
            if signals:
                parts.append(f"Signals Detected: {len(signals)}")
        
        return " | ".join(parts) if parts else "Analysis complete"
    
    def _extract_recommendation(
        self,
        portfolio: Dict[str, Any],
        governance: Dict[str, Any],
    ) -> str:
        """Extract recommendation from portfolio and governance outputs."""
        # Check governance first
        if governance.get("data", {}).get("status") != "APPROVED":
            return "HOLD"  # Default to HOLD if not approved
        
        # Get recommendation from portfolio agent
        return portfolio.get("data", {}).get("recommendation", {}).get("action", "NEUTRAL")
    
    def _extract_key_levels(
        self,
        ta: Dict[str, Any],
        portfolio: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract key price levels."""
        levels = {}
        
        # From technical analysis
        if ta and ta.get("data"):
            ta_data = ta["data"]
            if "support_levels" in ta_data:
                levels["support"] = ta_data["support_levels"]
            if "resistance_levels" in ta_data:
                levels["resistance"] = ta_data["resistance_levels"]
        
        # From portfolio agent
        if portfolio and portfolio.get("data"):
            portfolio_data = portfolio["data"]
            if "key_levels" in portfolio_data:
                levels.update(portfolio_data["key_levels"])
        
        return levels
    
    def _extract_risks(
        self,
        ta: Dict[str, Any],
        instability: Dict[str, Any],
    ) -> list:
        """Extract identified risks."""
        risks = []
        
        # From technical analysis
        if ta and ta.get("data"):
            if "risks" in ta["data"]:
                risks.extend(ta["data"]["risks"])
        
        # From instability analysis
        if instability and instability.get("data"):
            if "risk_warning" in instability["data"]:
                risks.append(instability["data"]["risk_warning"])
        
        return risks
    
    def _calculate_confidence(self, context: Dict[str, Any]) -> float:
        """
        Calculate overall confidence from all agent confidences.
        
        Returns:
            Weighted average confidence (0.0 to 1.0)
        """
        confidences = []
        
        for agent_name, output in context.items():
            if isinstance(output, dict) and "confidence" in output:
                confidences.append(output["confidence"])
        
        if not confidences:
            return 0.75  # Default if no confidences provided
        
        return sum(confidences) / len(confidences)
    
    def _build_task_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build summary of task execution."""
        summary = {}
        
        for agent_name, output in context.items():
            if isinstance(output, dict) and "status" in output:
                summary[agent_name] = {
                    "status": output.get("status", "UNKNOWN"),
                    "confidence": output.get("confidence", 0),
                    "duration_ms": output.get("duration_ms", 0),
                }
        
        return summary
