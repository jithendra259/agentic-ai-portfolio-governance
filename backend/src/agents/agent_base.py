"""
Agent Base Classes and Protocol Definitions

Provides the foundation for all specialized agents in the multi-agent system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4
import logging
import json

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Enumeration of all agent types in the system."""
    PLANNER = "planner"
    DATA = "data"
    TECHNICAL_ANALYSIS = "technical_analysis"
    REGIME_DETECTION = "regime_detection"
    INSTABILITY_ANALYSIS = "instability_analysis"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    GOVERNANCE = "governance"
    EXPLAINABILITY = "explainability"
    VERIFICATION = "verification"
    RESPONSE = "response"


class ExecutionStatus(Enum):
    """Execution status of an agent or task."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class VerificationStatus(Enum):
    """Verification status for output validation."""
    VERIFIED = "verified"
    BLOCKED = "blocked"
    WARNING = "warning"


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_type: AgentType
    name: str
    timeout_seconds: int = 30
    retry_count: int = 3
    log_level: str = "INFO"
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    """Standard output format from any agent."""
    agent_type: AgentType
    agent_name: str
    request_id: str
    output_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Execution metadata
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0
    
    # Main output
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_code: Optional[str] = None
    
    # Verification metadata
    verification_status: VerificationStatus = VerificationStatus.VERIFIED
    confidence: float = 1.0  # 0.0 to 1.0
    
    # Auditability
    evidence: List[str] = field(default_factory=list)
    calculations: Dict[str, str] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling datetime serialization."""
        d = asdict(self)
        d['agent_type'] = self.agent_type.value
        d['status'] = self.status.value
        d['verification_status'] = self.verification_status.value
        d['started_at'] = self.started_at.isoformat()
        d['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return d
    
    def mark_success(self, data: Dict[str, Any]) -> None:
        """Mark execution as successful."""
        self.status = ExecutionStatus.SUCCESS
        self.data = data
        self.completed_at = datetime.utcnow()
        self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000
    
    def mark_failed(self, error: str, error_code: str = "UNKNOWN") -> None:
        """Mark execution as failed."""
        self.status = ExecutionStatus.FAILED
        self.error = error
        self.error_code = error_code
        self.completed_at = datetime.utcnow()
        self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000
    
    def mark_blocked(self, reason: str) -> None:
        """Mark execution as blocked."""
        self.status = ExecutionStatus.BLOCKED
        self.error = reason
        self.completed_at = datetime.utcnow()
        self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000
    
    def add_evidence(self, evidence: str) -> None:
        """Add evidence to the output."""
        self.evidence.append(evidence)
    
    def add_calculation(self, name: str, description: str) -> None:
        """Record a calculation performed."""
        self.calculations[name] = description
    
    def add_source(self, source: str) -> None:
        """Add a data source."""
        self.sources.append(source)


@dataclass
class TaskDefinition:
    """Definition of a task to be executed by an agent."""
    task_id: str
    agent_type: AgentType
    agent_name: str
    description: str
    
    # Inputs
    inputs: Dict[str, Any] = field(default_factory=dict)
    
    # Execution control
    priority: int = 5  # 1=highest, 10=lowest
    timeout_seconds: int = 30
    retry_count: int = 1
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['agent_type'] = self.agent_type.value
        d['created_at'] = self.created_at.isoformat()
        return d


@dataclass
class TaskResult:
    """Result of executing a task."""
    task_id: str
    agent_output: AgentOutput
    task_definition: TaskDefinition
    
    # Execution details
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0
    
    # Status
    success: bool = False
    error: Optional[str] = None
    
    def mark_complete(self) -> None:
        """Mark task as complete."""
        self.completed_at = datetime.utcnow()
        self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000
        self.success = self.agent_output.status == ExecutionStatus.SUCCESS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_id': self.task_id,
            'agent_type': self.agent_output.agent_type.value,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_ms': self.duration_ms,
            'success': self.success,
            'error': self.error,
            'agent_output': self.agent_output.to_dict(),
        }


class Agent(ABC):
    """
    Abstract base class for all agents.
    
    All agents inherit from this class and implement the execute() method.
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize agent with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self.logger.setLevel(config.log_level)
    
    @abstractmethod
    async def execute(self, task: TaskDefinition, context: Dict[str, Any]) -> AgentOutput:
        """
        Execute the agent's task.
        
        Args:
            task: The task definition to execute
            context: Shared context including results from dependent tasks
        
        Returns:
            AgentOutput with results and metadata
        """
        pass
    
    def _create_output(self, request_id: str) -> AgentOutput:
        """Create a new output object for this agent."""
        return AgentOutput(
            agent_type=self.config.agent_type,
            agent_name=self.config.name,
            request_id=request_id,
        )
    
    def _log_info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(f"[{self.config.name}] {message}", extra=kwargs)
    
    def _log_error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(f"[{self.config.name}] {message}", extra=kwargs)
    
    def _log_warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(f"[{self.config.name}] {message}", extra=kwargs)


class SpecializedAgent(Agent):
    """
    Base class for specialized agents with common patterns.
    
    Provides standard error handling, validation, and output formatting.
    """
    
    def _validate_inputs(self, inputs: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        Validate that required inputs are present.
        
        Returns:
            True if all required inputs are present, False otherwise.
        """
        missing = [key for key in required_keys if key not in inputs or inputs[key] is None]
        if missing:
            self._log_error(f"Missing required inputs: {missing}")
            return False
        return True
    
    def _check_input_type(self, value: Any, expected_type: type, name: str) -> bool:
        """
        Validate input type.
        
        Returns:
            True if type matches, False otherwise.
        """
        if not isinstance(value, expected_type):
            self._log_error(
                f"Invalid type for {name}: expected {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
            return False
        return True
    
    def _safe_calculate(self, func, *args, **kwargs) -> tuple[bool, Any, Optional[str]]:
        """
        Safely execute a calculation, catching exceptions.
        
        Returns:
            Tuple of (success, result, error_message)
        """
        try:
            result = func(*args, **kwargs)
            return True, result, None
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self._log_error(f"Calculation failed: {error_msg}")
            return False, None, error_msg
    
    def _format_evidence(self, *items: str) -> None:
        """
        Format and log evidence items.
        
        Should be called before marking output as success.
        """
        for item in items:
            if item:
                self._log_info(f"Evidence: {item}")


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class HallucInationDetectedError(Exception):
    """Raised when hallucination is detected during verification."""
    pass


class GovernanceViolationError(Exception):
    """Raised when a governance rule is violated."""
    pass
