"""
Memory & Audit System

Provides comprehensive auditability and replay capability for all decisions.
Stores all plans, task results, verifications, and final decisions for compliance
and transparency.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """A single entry in the audit log."""
    entry_id: str
    request_id: str
    timestamp: datetime
    component: str  # Which agent or component
    action: str  # What happened
    status: str  # SUCCESS, FAILED, BLOCKED, WARNING
    data: Dict[str, Any]  # Relevant data
    confidence: float = 1.0  # Confidence in the action


class AuditLog:
    """
    In-memory audit log for current session.
    
    For persistence, integrate with MongoDB memory layer.
    """
    
    def __init__(self, request_id: str):
        """Initialize audit log for a request."""
        self.request_id = request_id
        self.entries: List[AuditEntry] = []
        self.created_at = datetime.utcnow()
    
    def log(
        self,
        component: str,
        action: str,
        status: str,
        data: Dict[str, Any],
        confidence: float = 1.0,
    ) -> None:
        """
        Log an action.
        
        Args:
            component: Agent or system component
            action: Action description
            status: SUCCESS, FAILED, BLOCKED, WARNING
            data: Relevant data
            confidence: Confidence in the action (0-1)
        """
        entry = AuditEntry(
            entry_id=str(uuid4()),
            request_id=self.request_id,
            timestamp=datetime.utcnow(),
            component=component,
            action=action,
            status=status,
            data=data,
            confidence=confidence,
        )
        self.entries.append(entry)
        
        logger.info(
            f"[AUDIT] {component} - {action}: {status} "
            f"(confidence: {confidence:.2f})"
        )
    
    def get_entries(self, component: Optional[str] = None) -> List[AuditEntry]:
        """Get audit entries, optionally filtered by component."""
        if component:
            return [e for e in self.entries if e.component == component]
        return self.entries
    
    def get_decision_trail(self) -> List[Dict[str, Any]]:
        """Get a summary of the decision trail."""
        return [
            {
                'timestamp': e.timestamp.isoformat(),
                'component': e.component,
                'action': e.action,
                'status': e.status,
                'confidence': e.confidence,
            }
            for e in self.entries
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'created_at': self.created_at.isoformat(),
            'entry_count': len(self.entries),
            'entries': [
                {
                    'entry_id': e.entry_id,
                    'timestamp': e.timestamp.isoformat(),
                    'component': e.component,
                    'action': e.action,
                    'status': e.status,
                    'confidence': e.confidence,
                    'data': e.data,
                }
                for e in self.entries
            ],
        }


class RequestMemory:
    """
    Memory for a single request execution.
    
    Stores:
    - User query and intent
    - Execution plan
    - Task results
    - Verification results
    - Final response
    - Audit trail
    """
    
    def __init__(self, request_id: str, user_query: str):
        """Initialize request memory."""
        self.request_id = request_id
        self.user_query = user_query
        self.created_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        
        # Execution data
        self.plan: Optional[Dict[str, Any]] = None
        self.task_results: Dict[str, Dict[str, Any]] = {}
        self.verification_report: Optional[Dict[str, Any]] = None
        self.final_response: Optional[Dict[str, Any]] = None
        
        # Audit trail
        self.audit_log = AuditLog(request_id)
    
    def store_plan(self, plan_dict: Dict[str, Any]) -> None:
        """Store the execution plan."""
        self.plan = plan_dict
        self.audit_log.log(
            component="Planner",
            action="Created execution plan",
            status="SUCCESS",
            data={"task_count": len(plan_dict.get("tasks", []))},
        )
    
    def store_task_result(self, task_id: str, result_dict: Dict[str, Any]) -> None:
        """Store a task result."""
        self.task_results[task_id] = result_dict
        
        status = result_dict.get("agent_output", {}).get("status", "UNKNOWN")
        self.audit_log.log(
            component=result_dict.get("agent_output", {}).get("agent_name", "Unknown"),
            action=f"Task {task_id} executed",
            status=status,
            data={
                "task_id": task_id,
                "duration_ms": result_dict.get("duration_ms", 0),
            },
            confidence=result_dict.get("agent_output", {}).get("confidence", 1.0),
        )
    
    def store_verification(self, verification_dict: Dict[str, Any]) -> None:
        """Store the verification report."""
        self.verification_report = verification_dict
        
        status = verification_dict.get("overall_status", "UNKNOWN")
        self.audit_log.log(
            component="VerificationAgent",
            action="Verification complete",
            status=status,
            data={
                "checks_passed": verification_dict.get("checks_passed", 0),
                "checks_failed": verification_dict.get("checks_failed", 0),
            },
        )
    
    def store_response(self, response_dict: Dict[str, Any]) -> None:
        """Store the final response."""
        self.final_response = response_dict
        self.completed_at = datetime.utcnow()
        
        self.audit_log.log(
            component="ResponseAgent",
            action="Final response generated",
            status="SUCCESS",
            data={
                "recommendation": response_dict.get("recommendation", {}).get("action"),
                "confidence": response_dict.get("confidence_score", 0),
            },
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the request execution."""
        duration = (
            (self.completed_at - self.created_at).total_seconds()
            if self.completed_at
            else None
        )
        
        return {
            'request_id': self.request_id,
            'user_query': self.user_query,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': duration,
            'tasks_completed': len(self.task_results),
            'has_verification': self.verification_report is not None,
            'has_response': self.final_response is not None,
            'recommendation': (
                self.final_response.get('recommendation', {}).get('action')
                if self.final_response else None
            ),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'request_id': self.request_id,
            'user_query': self.user_query,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'plan': self.plan,
            'task_results': self.task_results,
            'verification_report': self.verification_report,
            'final_response': self.final_response,
            'audit_trail': self.audit_log.to_dict(),
        }


class MemoryManager:
    """
    Manages request memory and audit logs.
    
    Should be integrated with MongoDB for persistence.
    Current implementation is in-memory; add MongoDB integration for production.
    """
    
    def __init__(self):
        """Initialize memory manager."""
        self.requests: Dict[str, RequestMemory] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_request(self, user_query: str) -> str:
        """
        Create a new request in memory.
        
        Returns:
            Request ID
        """
        request_id = str(uuid4())
        self.requests[request_id] = RequestMemory(request_id, user_query)
        
        self.logger.info(f"Created request {request_id}")
        
        return request_id
    
    def get_request(self, request_id: str) -> Optional[RequestMemory]:
        """Get a request from memory."""
        return self.requests.get(request_id)
    
    def store_plan(self, request_id: str, plan_dict: Dict[str, Any]) -> None:
        """Store an execution plan."""
        req = self.get_request(request_id)
        if req:
            req.store_plan(plan_dict)
    
    def store_task_result(self, request_id: str, task_id: str, result_dict: Dict[str, Any]) -> None:
        """Store a task result."""
        req = self.get_request(request_id)
        if req:
            req.store_task_result(task_id, result_dict)
    
    def store_verification(self, request_id: str, verification_dict: Dict[str, Any]) -> None:
        """Store a verification report."""
        req = self.get_request(request_id)
        if req:
            req.store_verification(verification_dict)
    
    def store_response(self, request_id: str, response_dict: Dict[str, Any]) -> None:
        """Store a final response."""
        req = self.get_request(request_id)
        if req:
            req.store_response(response_dict)
    
    def get_audit_log(self, request_id: str) -> Optional[AuditLog]:
        """Get the audit log for a request."""
        req = self.get_request(request_id)
        return req.audit_log if req else None
    
    def get_summary(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of request execution."""
        req = self.get_request(request_id)
        return req.get_summary() if req else None
    
    def get_full_record(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the complete record for a request."""
        req = self.get_request(request_id)
        return req.to_dict() if req else None
    
    def cleanup_old_requests(self, max_age_hours: int = 24) -> int:
        """
        Remove requests older than max_age_hours.
        
        Returns:
            Number of requests removed
        """
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        to_remove = [
            req_id for req_id, req in self.requests.items()
            if req.created_at < cutoff
        ]
        
        for req_id in to_remove:
            del self.requests[req_id]
        
        if to_remove:
            self.logger.info(f"Cleaned up {len(to_remove)} old requests")
        
        return len(to_remove)
