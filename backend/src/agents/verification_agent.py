"""
Verification Agent

Critical gatekeeper that prevents hallucinations.
Validates all outputs before allowing recommendation.
Blocks unsupported claims.
"""

import logging
from typing import Any, Dict, List

from src.agents.agent_base import (
    AgentConfig,
    AgentOutput,
    AgentType,
    SpecializedAgent,
    TaskDefinition,
)
from src.agents.verification_framework import (
    VerificationFramework,
    VerificationCheck,
    VerificationCheckType,
    VerificationLevel,
)

logger = logging.getLogger(__name__)


class VerificationAgent(SpecializedAgent):
    """
    Verification Agent: Prevents hallucinations by validating all outputs.
    
    Critical Component - Acts as gatekeeper for recommendations.
    
    Responsibilities:
    - Verify all agent outputs
    - Check evidence availability
    - Validate calculations
    - Confirm governance approval
    - Detect contradictions
    - Block unsupported recommendations
    
    Anti-Hallucination Protocol:
    1. Every claim must have evidence
    2. Every calculation must be verified
    3. Every recommendation must be governance-approved
    4. Confidence must be calibrated
    5. If evidence missing → block
    6. If calculation incomplete → block
    7. If governance rejected → block
    8. If contradictions detected → flag for review
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize verification agent."""
        super().__init__(config)
        self.framework = VerificationFramework(min_confidence=0.5)
    
    async def execute(
        self,
        task: TaskDefinition,
        context: Dict[str, Any],
    ) -> AgentOutput:
        """
        Verify all outputs.
        
        Args:
            task: Task definition (unused, this agent uses context)
            context: Execution context with all outputs to verify
        
        Returns:
            AgentOutput with verification report
        """
        output = self._create_output(task.inputs.get("request_id", ""))
        
        try:
            self._log_info("Starting verification of all outputs")
            
            # Perform comprehensive verification
            verification_report = self._verify_all_outputs(context)
            
            # Compile results
            result = {
                "verification_status": verification_report.overall_status.value,
                "checks_passed": verification_report.checks_passed,
                "checks_failed": verification_report.checks_failed,
                "checks_warned": verification_report.checks_warned,
                "blocking_issues": verification_report.blocking_issues,
                "warnings": verification_report.warnings,
                "details": verification_report.to_dict(),
            }
            
            output.mark_success(result)
            
            # Set confidence based on verification
            if verification_report.checks_failed > 0:
                output.confidence = 0.3
                output.add_evidence(
                    f"Verification failed: {verification_report.checks_failed} checks did not pass"
                )
            elif verification_report.checks_warned > 0:
                output.confidence = 0.6
                output.add_evidence(
                    f"Verification passed with {verification_report.checks_warned} warnings"
                )
            else:
                output.confidence = 0.95
                output.add_evidence("All verification checks passed")
            
        except Exception as e:
            self._log_error(f"Verification failed: {str(e)}")
            output.mark_failed(f"Verification error: {str(e)}", "VERIFICATION_ERROR")
        
        return output
    
    def _verify_all_outputs(self, context: Dict[str, Any]):
        """
        Verify all agent outputs.
        
        Returns:
            VerificationReport
        """
        from src.agents.verification_framework import VerificationReport
        
        report = VerificationReport(request_id=context.get("request_id", ""))
        
        # Verify each agent output
        for agent_name, agent_output in context.items():
            if agent_name in ["request_id", "plan_id"]:
                continue  # Skip non-agent items
            
            if isinstance(agent_output, dict):
                # Verify this output
                self._verify_agent_output(agent_output, agent_name, report)
        
        report.finalize()
        
        return report
    
    def _verify_agent_output(
        self,
        output_dict: Dict[str, Any],
        agent_name: str,
        report,
    ) -> None:
        """Verify a single agent output."""
        
        self._log_info(f"Verifying output from {agent_name}")
        
        # Check execution status
        status = output_dict.get("status")
        if status != "success":
            result = self.framework._perform_check(
                self._create_mock_output(output_dict),
                VerificationCheck(
                    check_type=VerificationCheckType.EXECUTION_SUCCESS,
                    description=f"{agent_name} execution status",
                    severity=VerificationLevel.ERROR,
                )
            )
            report.add_check(result)
            return  # Don't verify further if execution failed
        
        # Check evidence
        evidence = output_dict.get("evidence", [])
        result = self.framework._perform_check(
            self._create_mock_output(output_dict),
            VerificationCheck(
                check_type=VerificationCheckType.EVIDENCE_REQUIRED,
                description=f"{agent_name} evidence",
                severity=VerificationLevel.ERROR,
            )
        )
        report.add_check(result)
        
        # Check confidence
        confidence = output_dict.get("confidence", 0)
        result = self.framework._perform_check(
            self._create_mock_output(output_dict),
            VerificationCheck(
                check_type=VerificationCheckType.CONFIDENCE_THRESHOLD,
                description=f"{agent_name} confidence threshold",
                severity=VerificationLevel.WARNING,
                min_confidence=0.5,
            )
        )
        report.add_check(result)
        
        # Check completeness
        data = output_dict.get("data", {})
        is_complete = bool(data) if isinstance(data, dict) else data is not None
        result = self.framework._perform_check(
            self._create_mock_output(output_dict),
            VerificationCheck(
                check_type=VerificationCheckType.COMPLETENESS,
                description=f"{agent_name} output completeness",
                severity=VerificationLevel.ERROR,
            )
        )
        report.add_check(result)
        
        # Agent-specific checks
        if "governance" in agent_name.lower():
            self._verify_governance_output(output_dict, report)
        
        if "data" in agent_name.lower():
            self._verify_data_output(output_dict, report)
        
        if "technical" in agent_name.lower():
            self._verify_technical_output(output_dict, report)
    
    def _verify_governance_output(self, output_dict: Dict[str, Any], report) -> None:
        """Verify governance-specific requirements."""
        
        status = output_dict.get("data", {}).get("status")
        
        from src.agents.verification_framework import VerificationResult, VerificationCheckType
        
        result = VerificationResult(
            check_type=VerificationCheckType.GOVERNANCE_APPROVED,
            passed=status == "APPROVED",
            message=f"Governance status: {status}",
            severity=VerificationLevel.CRITICAL,
        )
        report.add_check(result)
    
    def _verify_data_output(self, output_dict: Dict[str, Any], report) -> None:
        """Verify data retrieval requirements."""
        
        from src.agents.verification_framework import VerificationResult, VerificationCheckType
        
        data = output_dict.get("data", {})
        sources = output_dict.get("sources", [])
        
        result = VerificationResult(
            check_type=VerificationCheckType.DATA_RETRIEVED,
            passed=bool(data) and len(sources) > 0,
            message=f"Data retrieved with {len(sources)} source(s)",
            evidence=sources,
            severity=VerificationLevel.ERROR,
        )
        report.add_check(result)
    
    def _verify_technical_output(self, output_dict: Dict[str, Any], report) -> None:
        """Verify technical analysis requirements."""
        
        from src.agents.verification_framework import VerificationResult, VerificationCheckType
        
        data = output_dict.get("data", {})
        calculations = output_dict.get("calculations", {})
        
        result = VerificationResult(
            check_type=VerificationCheckType.CALCULATION_COMPLETE,
            passed=bool(calculations) and len(calculations) > 0,
            message=f"Calculations performed: {len(calculations)}",
            evidence=list(calculations.keys()),
            severity=VerificationLevel.ERROR,
        )
        report.add_check(result)
    
    def _create_mock_output(self, output_dict: Dict[str, Any]):
        """Create a mock AgentOutput for verification."""
        from src.agents.agent_base import AgentOutput, ExecutionStatus, VerificationStatus, AgentType
        
        status_map = {
            "success": ExecutionStatus.SUCCESS,
            "failed": ExecutionStatus.FAILED,
            "blocked": ExecutionStatus.BLOCKED,
        }
        
        return AgentOutput(
            agent_type=AgentType.VERIFICATION,
            agent_name="MockAgent",
            request_id="",
            status=status_map.get(output_dict.get("status", "failed"), ExecutionStatus.FAILED),
            data=output_dict.get("data", {}),
            error=output_dict.get("error"),
            confidence=output_dict.get("confidence", 0),
            evidence=output_dict.get("evidence", []),
            sources=output_dict.get("sources", []),
        )
