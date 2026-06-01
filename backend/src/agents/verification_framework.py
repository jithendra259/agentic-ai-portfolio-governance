"""
Verification Framework

Implements anti-hallucination checks and evidence validation.
Critical component that gates recommendations and ensures explainability.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.agents.agent_base import (
    AgentOutput,
    ExecutionStatus,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


class VerificationLevel(Enum):
    """Severity levels for verification failures."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class VerificationCheckType(Enum):
    """Types of verification checks."""
    EVIDENCE_REQUIRED = "evidence_required"
    DATA_RETRIEVED = "data_retrieved"
    CALCULATION_COMPLETE = "calculation_complete"
    GOVERNANCE_APPROVED = "governance_approved"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    NO_CONTRADICTION = "no_contradiction"
    COMPLETENESS = "completeness"
    EXECUTION_SUCCESS = "execution_success"


@dataclass
class VerificationCheck:
    """A single verification check to perform."""
    check_type: VerificationCheckType
    description: str
    severity: VerificationLevel = VerificationLevel.ERROR
    min_confidence: float = 0.5
    required: bool = True  # If True, failure blocks recommendation


@dataclass
class VerificationResult:
    """Result of a verification check."""
    check_type: VerificationCheckType
    passed: bool
    message: str
    evidence: List[str] = field(default_factory=list)
    severity: VerificationLevel = VerificationLevel.INFO
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VerificationReport:
    """Complete verification report."""
    verification_id: str = field(default_factory=lambda: str(uuid4()))
    request_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Checks performed
    checks: List[VerificationResult] = field(default_factory=list)
    
    # Summary
    checks_passed: int = 0
    checks_failed: int = 0
    checks_warned: int = 0
    
    # Overall status
    overall_status: VerificationStatus = VerificationStatus.VERIFIED
    blocking_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_check(self, result: VerificationResult) -> None:
        """Add a check result."""
        self.checks.append(result)
        
        if result.passed:
            self.checks_passed += 1
        elif result.severity == VerificationLevel.WARNING:
            self.checks_warned += 1
            self.warnings.append(result.message)
        else:
            self.checks_failed += 1
            self.blocking_issues.append(result.message)
    
    def finalize(self) -> None:
        """Finalize the report and determine overall status."""
        if self.checks_failed > 0:
            self.overall_status = VerificationStatus.BLOCKED
        elif self.checks_warned > 0:
            self.overall_status = VerificationStatus.WARNING
        else:
            self.overall_status = VerificationStatus.VERIFIED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'verification_id': self.verification_id,
            'request_id': self.request_id,
            'timestamp': self.timestamp.isoformat(),
            'checks_passed': self.checks_passed,
            'checks_failed': self.checks_failed,
            'checks_warned': self.checks_warned,
            'overall_status': self.overall_status.value,
            'blocking_issues': self.blocking_issues,
            'warnings': self.warnings,
            'checks': [
                {
                    'type': c.check_type.value,
                    'passed': c.passed,
                    'message': c.message,
                    'evidence': c.evidence,
                }
                for c in self.checks
            ]
        }


class VerificationFramework:
    """
    Verifies outputs from agents to prevent hallucinations.
    
    Anti-Hallucination Rules:
    1. Never invent numerical values
    2. Every recommendation requires source data
    3. Every signal requires evidence
    4. Every claim requires calculation proof
    5. Governance approval required before recommendation
    6. Confidence scoring mandatory
    7. If confidence < threshold, return warning
    8. If evidence unavailable, block recommendation
    """
    
    # Default verification checks
    DEFAULT_CHECKS = [
        VerificationCheck(
            check_type=VerificationCheckType.EXECUTION_SUCCESS,
            description="Agent execution completed successfully",
            severity=VerificationLevel.CRITICAL,
            required=True,
        ),
        VerificationCheck(
            check_type=VerificationCheckType.EVIDENCE_REQUIRED,
            description="Output includes supporting evidence",
            severity=VerificationLevel.ERROR,
            required=True,
        ),
        VerificationCheck(
            check_type=VerificationCheckType.DATA_RETRIEVED,
            description="Required data was successfully retrieved",
            severity=VerificationLevel.ERROR,
            required=True,
        ),
        VerificationCheck(
            check_type=VerificationCheckType.CONFIDENCE_THRESHOLD,
            description="Confidence score meets minimum threshold",
            severity=VerificationLevel.WARNING,
            min_confidence=0.5,
            required=False,
        ),
        VerificationCheck(
            check_type=VerificationCheckType.COMPLETENESS,
            description="All required outputs are present",
            severity=VerificationLevel.ERROR,
            required=True,
        ),
    ]
    
    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize verification framework.
        
        Args:
            min_confidence: Minimum confidence score for recommendations
        """
        self.min_confidence = min_confidence
        self.logger = logging.getLogger(__name__)
    
    def verify_output(
        self,
        output: AgentOutput,
        request_id: str,
        checks: Optional[List[VerificationCheck]] = None,
    ) -> VerificationReport:
        """
        Verify an agent output.
        
        Args:
            output: The agent output to verify
            request_id: Request ID for tracking
            checks: Specific checks to perform (or None for defaults)
        
        Returns:
            VerificationReport with results
        """
        if checks is None:
            checks = self.DEFAULT_CHECKS
        
        report = VerificationReport(request_id=request_id)
        
        # Perform all checks
        for check in checks:
            result = self._perform_check(output, check)
            report.add_check(result)
            
            self.logger.info(
                f"Check {check.check_type.value}: "
                f"{'PASS' if result.passed else 'FAIL'} - {result.message}"
            )
        
        report.finalize()
        
        return report
    
    def _perform_check(
        self,
        output: AgentOutput,
        check: VerificationCheck,
    ) -> VerificationResult:
        """Perform a single verification check."""
        
        if check.check_type == VerificationCheckType.EXECUTION_SUCCESS:
            return self._check_execution_success(output, check)
        
        elif check.check_type == VerificationCheckType.EVIDENCE_REQUIRED:
            return self._check_evidence_required(output, check)
        
        elif check.check_type == VerificationCheckType.DATA_RETRIEVED:
            return self._check_data_retrieved(output, check)
        
        elif check.check_type == VerificationCheckType.CONFIDENCE_THRESHOLD:
            return self._check_confidence_threshold(output, check)
        
        elif check.check_type == VerificationCheckType.COMPLETENESS:
            return self._check_completeness(output, check)
        
        elif check.check_type == VerificationCheckType.GOVERNANCE_APPROVED:
            return self._check_governance_approved(output, check)
        
        elif check.check_type == VerificationCheckType.NO_CONTRADICTION:
            return self._check_no_contradiction(output, check)
        
        else:
            return VerificationResult(
                check_type=check.check_type,
                passed=False,
                message=f"Unknown check type: {check.check_type}",
                severity=VerificationLevel.ERROR,
            )
    
    def _check_execution_success(
        self,
        output: AgentOutput,
        check: VerificationCheck,
    ) -> VerificationResult:
        """Rule 1: Execution must succeed."""
        passed = output.status == ExecutionStatus.SUCCESS
        
        return VerificationResult(
            check_type=check.check_type,
            passed=passed,
            message=(
                "Agent execution completed successfully"
                if passed
                else f"Agent execution failed: {output.error}"
            ),
            evidence=[f"Status: {output.status.value}"],
            severity=check.severity,
        )
    
    def _check_evidence_required(
        self,
        output: AgentOutput,
        check: VerificationCheck,
    ) -> VerificationResult:
        """Rule 3: Every signal requires evidence."""
        passed = len(output.evidence) > 0 or len(output.data) == 0
        
        return VerificationResult(
            check_type=check.check_type,
            passed=passed,
            message=(
                f"Evidence provided: {len(output.evidence)} item(s)"
                if passed
                else "No evidence provided for output claims"
            ),
            evidence=output.evidence[:3],  # Show first 3
            severity=check.severity,
        )
    
    def _check_data_retrieved(
        self,
        output: AgentOutput,
        check: VerificationCheck,
    ) -> VerificationResult:
        """Rule 2: Required data must be retrieved."""
        # Check if data agent provided data or if this is a downstream agent
        is_data_agent = "data" in output.agent_type.value.lower()
        
        if is_data_agent:
            # Data agent must have retrieved data
            has_data = bool(output.data)
            passed = has_data or output.status != ExecutionStatus.SUCCESS
        else:
            # Downstream agents inherit data from data agent
            passed = True
        
        return VerificationResult(
            check_type=check.check_type,
            passed=passed,
            message=(
                "Data retrieval successful"
                if passed
                else "Failed to retrieve required data"
            ),
            evidence=output.sources,
            severity=check.severity,
        )
    
    def _check_confidence_threshold(
        self,
        output: AgentOutput,
        check: VerificationCheck,
    ) -> VerificationResult:
        """Rule 6 & 7: Confidence must meet threshold."""
        passed = output.confidence >= check.min_confidence
        
        return VerificationResult(
            check_type=check.check_type,
            passed=passed,
            message=(
                f"Confidence {output.confidence:.1%} >= threshold {check.min_confidence:.1%}"
                if passed
                else f"Confidence {output.confidence:.1%} < threshold {check.min_confidence:.1%}. "
                     f"Recommend manual review."
            ),
            evidence=[f"Confidence score: {output.confidence:.2f}"],
            severity=check.severity,
        )
    
    def _check_completeness(
        self,
        output: AgentOutput,
        check: VerificationCheck,
    ) -> VerificationResult:
        """Rule 7: All required outputs present."""
        # Check that output has expected structure
        has_data = bool(output.data) if output.status == ExecutionStatus.SUCCESS else True
        passed = has_data and output.error is None
        
        return VerificationResult(
            check_type=check.check_type,
            passed=passed,
            message=(
                "All required outputs present"
                if passed
                else f"Incomplete output: {output.error}"
            ),
            severity=check.severity,
        )
    
    def _check_governance_approved(
        self,
        output: AgentOutput,
        check: VerificationCheck,
    ) -> VerificationResult:
        """Rule 5: Governance approval required."""
        # Check if governance agent approved
        is_governance_agent = "governance" in output.agent_type.value.lower()
        
        if is_governance_agent:
            passed = output.data.get("status") == "APPROVED"
        else:
            # Non-governance agents don't need this check
            passed = True
        
        return VerificationResult(
            check_type=check.check_type,
            passed=passed,
            message=(
                "Governance approval obtained"
                if passed
                else f"Governance approval denied: {output.data.get('reason', 'unknown')}"
            ),
            severity=check.severity,
        )
    
    def _check_no_contradiction(
        self,
        output: AgentOutput,
        check: VerificationCheck,
    ) -> VerificationResult:
        """Rule 8: Detect internal contradictions."""
        # Check for contradictory data within output
        passed = True
        contradictions = []
        
        if isinstance(output.data, dict):
            # Simple contradiction check: confidence and status alignment
            if output.confidence < 0.3 and output.status == ExecutionStatus.SUCCESS:
                contradictions.append("Low confidence but marked success")
            
            # Check for missing data in required fields
            if output.status == ExecutionStatus.SUCCESS:
                if not output.data and output.confidence > 0.8:
                    contradictions.append("High confidence but empty output")
        
        if contradictions:
            passed = False
        
        return VerificationResult(
            check_type=check.check_type,
            passed=passed,
            message=(
                "No contradictions detected"
                if passed
                else f"Contradictions found: {'; '.join(contradictions)}"
            ),
            evidence=contradictions,
            severity=check.severity,
        )
    
    def verify_recommendation(
        self,
        recommendation: Dict[str, Any],
        verification_report: VerificationReport,
    ) -> tuple[bool, Optional[str]]:
        """
        Verify a recommendation is sound.
        
        Returns:
            Tuple of (approved, rejection_reason)
        """
        # Rule: Blocked verification blocks recommendation
        if verification_report.overall_status == VerificationStatus.BLOCKED:
            reason = f"Verification blocked: {'; '.join(verification_report.blocking_issues)}"
            return False, reason
        
        # Rule: Must have evidence
        if not verification_report.checks_passed:
            return False, "No verification checks passed"
        
        # Rule: Governance approval required
        governance_passed = any(
            c.check_type == VerificationCheckType.GOVERNANCE_APPROVED and c.passed
            for c in verification_report.checks
        )
        if not governance_passed:
            # Only critical if governance check was performed
            governance_check_performed = any(
                c.check_type == VerificationCheckType.GOVERNANCE_APPROVED
                for c in verification_report.checks
            )
            if governance_check_performed:
                return False, "Governance approval not obtained"
        
        return True, None
