"""
Multi-Agent Manager

Orchestrates the complete multi-agent workflow.
Central coordinator for all agents and execution.
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from uuid import uuid4

from src.agents.agent_base import AgentType, AgentConfig
from src.agents.task_execution_engine import (
    TaskExecutor,
    ExecutionPlan,
    TaskDefinition,
)
from src.agents.verification_framework import VerificationFramework
from src.agents.memory_audit_system import MemoryManager
from src.agents.planner_agent import PlannerAgent
from src.agents.data_agent import DataAgent
from src.agents.verification_agent import VerificationAgent
from src.agents.response_agent import ResponseAgent
from src.agents.phase2_stub_agents import (
    TechnicalAnalysisAgent,
    RegimeDetectionAgent,
    InstabilityAnalysisAgent,
    PortfolioOptimizationAgent,
    GovernanceAgent,
    ExplainabilityAgent,
)

logger = logging.getLogger(__name__)


class MultiAgentManager:
    """
    Central orchestrator for the multi-agent adaptive system.
    
    Coordinates:
    - Task planning
    - Agent execution
    - Verification gating
    - Response assembly
    - Audit trail management
    """
    
    def __init__(self):
        """Initialize multi-agent system."""
        self.logger = logging.getLogger(__name__)
        
        # Core systems
        self.memory_manager = MemoryManager()
        self.verification_framework = VerificationFramework(min_confidence=0.5)
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Task executor
        self.executor = TaskExecutor(self.agents)
        
        self.logger.info("Multi-Agent Manager initialized with %d agents", len(self.agents))
    
    def _initialize_agents(self) -> Dict[AgentType, Any]:
        """Initialize all available agents."""
        agents = {}
        
        # Phase 1: Critical agents (fully implemented)
        agents[AgentType.PLANNER] = PlannerAgent(
            AgentConfig(
                agent_type=AgentType.PLANNER,
                name="PlannerAgent",
                timeout_seconds=10,
                tags=["critical", "phase1"],
            )
        )
        
        agents[AgentType.DATA] = DataAgent(
            AgentConfig(
                agent_type=AgentType.DATA,
                name="DataAgent",
                timeout_seconds=30,
                tags=["critical", "phase1"],
            )
        )
        
        agents[AgentType.VERIFICATION] = VerificationAgent(
            AgentConfig(
                agent_type=AgentType.VERIFICATION,
                name="VerificationAgent",
                timeout_seconds=10,
                tags=["critical", "phase1"],
            )
        )
        
        agents[AgentType.RESPONSE] = ResponseAgent(
            AgentConfig(
                agent_type=AgentType.RESPONSE,
                name="ResponseAgent",
                timeout_seconds=10,
                tags=["critical", "phase1"],
            )
        )
        
        # Phase 2: Specialist agents (stubs with valid structure)
        agents[AgentType.TECHNICAL_ANALYSIS] = TechnicalAnalysisAgent(
            AgentConfig(
                agent_type=AgentType.TECHNICAL_ANALYSIS,
                name="TechnicalAnalysisAgent",
                timeout_seconds=30,
                tags=["specialist", "phase2"],
            )
        )
        
        agents[AgentType.REGIME_DETECTION] = RegimeDetectionAgent(
            AgentConfig(
                agent_type=AgentType.REGIME_DETECTION,
                name="RegimeDetectionAgent",
                timeout_seconds=15,
                tags=["specialist", "phase2"],
            )
        )
        
        agents[AgentType.INSTABILITY_ANALYSIS] = InstabilityAnalysisAgent(
            AgentConfig(
                agent_type=AgentType.INSTABILITY_ANALYSIS,
                name="InstabilityAnalysisAgent",
                timeout_seconds=20,
                tags=["specialist", "phase2"],
            )
        )
        
        agents[AgentType.PORTFOLIO_OPTIMIZATION] = PortfolioOptimizationAgent(
            AgentConfig(
                agent_type=AgentType.PORTFOLIO_OPTIMIZATION,
                name="PortfolioOptimizationAgent",
                timeout_seconds=30,
                tags=["specialist", "phase2"],
            )
        )
        
        agents[AgentType.GOVERNANCE] = GovernanceAgent(
            AgentConfig(
                agent_type=AgentType.GOVERNANCE,
                name="GovernanceAgent",
                timeout_seconds=10,
                tags=["specialist", "phase2"],
            )
        )
        
        agents[AgentType.EXPLAINABILITY] = ExplainabilityAgent(
            AgentConfig(
                agent_type=AgentType.EXPLAINABILITY,
                name="ExplainabilityAgent",
                timeout_seconds=15,
                tags=["specialist", "phase2"],
            )
        )
        
        return agents
    
    async def execute(
        self,
        user_query: str,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute complete multi-agent workflow.
        
        Args:
            user_query: User's natural language query
            user_context: Optional context (portfolio, preferences, etc.)
        
        Returns:
            Complete response with audit trail and decision justification
        """
        # Create request in memory
        request_id = self.memory_manager.create_request(user_query)
        
        try:
            self.logger.info(f"Starting multi-agent execution for request {request_id}")
            
            # Prepare inputs
            inputs = {
                "request_id": request_id,
                "user_query": user_query,
                "data_type": "price_data",  # Default, can be overridden
            }
            if user_context:
                inputs.update(user_context)
            
            # ========== STEP 1: PLANNING ==========
            plan_output, plan_dict = await self._execute_planning_phase(inputs)
            
            if not plan_output or not plan_output.data:
                return self._build_error_response(
                    request_id,
                    "Failed to create execution plan",
                    "PLANNING_FAILED"
                )
            
            # ========== STEP 2: TASK EXECUTION ==========
            execution_context = await self._execute_tasks_phase(
                plan_dict,
                request_id,
            )
            
            # ========== STEP 3: VERIFICATION ==========
            verification_output = await self._execute_verification_phase(
                execution_context,
                request_id,
            )
            
            # ========== STEP 4: RESPONSE ASSEMBLY ==========
            response_output = await self._execute_response_phase(
                execution_context,
                verification_output,
                request_id,
            )
            
            # ========== STEP 5: FINALIZATION ==========
            result = self._build_success_response(
                request_id,
                response_output,
                execution_context,
                verification_output,
            )
            
            self.logger.info(f"Request {request_id} completed successfully")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Request {request_id} failed: {str(e)}", exc_info=True)
            
            # Log error in audit trail
            audit_log = self.memory_manager.get_audit_log(request_id)
            if audit_log:
                audit_log.log(
                    component="MultiAgentManager",
                    action="Workflow execution failed",
                    status="FAILED",
                    data={"error": str(e)},
                )
            
            return self._build_error_response(
                request_id,
                str(e),
                "EXECUTION_ERROR"
            )
    
    async def _execute_planning_phase(
        self,
        inputs: Dict[str, Any],
    ) -> tuple[Any, Dict[str, Any]]:
        """Execute planning phase."""
        self.logger.info("Phase 1: Planning")
        
        planner = self.agents[AgentType.PLANNER]
        
        plan_task = TaskDefinition(
            task_id="T000_planning",
            agent_type=AgentType.PLANNER,
            agent_name="PlannerAgent",
            description="Create execution plan",
            inputs=inputs,
        )
        
        plan_output = await planner.execute(plan_task, {})
        
        if plan_output.data:
            self.memory_manager.store_plan(
                inputs["request_id"],
                plan_output.to_dict()
            )
        
        return plan_output, plan_output.data or {}
    
    async def _execute_tasks_phase(
        self,
        plan_dict: Dict[str, Any],
        request_id: str,
    ) -> Any:
        """Execute task execution phase."""
        self.logger.info("Phase 2: Task Execution")
        
        # Build execution plan
        plan = ExecutionPlan(request_id=request_id)
        plan.plan_id = plan_dict.get("plan_id", "")
        
        for task_spec in plan_dict.get("tasks", []):
            try:
                agent_type = AgentType[task_spec["agent_type"].upper()]
            except KeyError:
                self.logger.warning(
                    f"Unknown agent type: {task_spec['agent_type']}, skipping task"
                )
                continue
            
            task = TaskDefinition(
                task_id=task_spec["task_id"],
                agent_type=agent_type,
                agent_name=task_spec["agent_name"],
                description=task_spec["description"],
                inputs=task_spec.get("inputs", {}),
                depends_on=task_spec.get("depends_on", []),
                priority=task_spec.get("priority", 5),
                timeout_seconds=task_spec.get("timeout_seconds", 30),
            )
            plan.add_task(task)
        
        # Execute plan
        execution_context = await self.executor.execute(plan)
        
        # Store results
        for task_id, result in execution_context.task_results.items():
            self.memory_manager.store_task_result(
                request_id,
                task_id,
                result.to_dict()
            )
        
        return execution_context
    
    async def _execute_verification_phase(
        self,
        execution_context: Any,
        request_id: str,
    ) -> Any:
        """Execute verification phase."""
        self.logger.info("Phase 3: Verification")
        
        verification_task = TaskDefinition(
            task_id="T999_verification",
            agent_type=AgentType.VERIFICATION,
            agent_name="VerificationAgent",
            description="Verify all outputs",
            inputs={"request_id": request_id},
        )
        
        # Create context from execution results
        verification_context = execution_context.get_all_outputs()
        verification_context["request_id"] = request_id
        
        verification_agent = self.agents[AgentType.VERIFICATION]
        verification_output = await verification_agent.execute(
            verification_task,
            verification_context,
        )
        
        self.memory_manager.store_verification(
            request_id,
            verification_output.to_dict()
        )
        
        return verification_output
    
    async def _execute_response_phase(
        self,
        execution_context: Any,
        verification_output: Any,
        request_id: str,
    ) -> Any:
        """Execute response assembly phase."""
        self.logger.info("Phase 4: Response Assembly")
        
        response_task = TaskDefinition(
            task_id="T999_response",
            agent_type=AgentType.RESPONSE,
            agent_name="ResponseAgent",
            description="Assemble final response",
            inputs={"request_id": request_id},
        )
        
        # Create response context with all outputs
        response_context = execution_context.get_all_outputs()
        response_context["verification"] = verification_output.to_dict()
        response_context["request_id"] = request_id
        
        response_agent = self.agents[AgentType.RESPONSE]
        response_output = await response_agent.execute(
            response_task,
            response_context,
        )
        
        self.memory_manager.store_response(
            request_id,
            response_output.to_dict()
        )
        
        return response_output
    
    def _build_success_response(
        self,
        request_id: str,
        response_output: Any,
        execution_context: Any,
        verification_output: Any,
    ) -> Dict[str, Any]:
        """Build successful response."""
        return {
            "status": "success",
            "request_id": request_id,
            "response": response_output.data if response_output else {},
            "verification_status": verification_output.data.get("verification_status", "UNKNOWN")
            if verification_output else "UNKNOWN",
            "audit_trail": self.memory_manager.get_audit_log(request_id).to_dict(),
            "execution_summary": execution_context.to_dict(),
        }
    
    def _build_error_response(
        self,
        request_id: str,
        error: str,
        error_code: str,
    ) -> Dict[str, Any]:
        """Build error response."""
        return {
            "status": "error",
            "request_id": request_id,
            "error": error,
            "error_code": error_code,
            "audit_trail": self.memory_manager.get_audit_log(request_id).to_dict()
            if self.memory_manager.get_request(request_id)
            else {},
        }
    
    def get_execution_summary(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of request execution."""
        return self.memory_manager.get_summary(request_id)
    
    def get_full_record(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get complete execution record."""
        return self.memory_manager.get_full_record(request_id)
    
    def cleanup_old_requests(self, max_age_hours: int = 24) -> int:
        """Clean up old requests from memory."""
        return self.memory_manager.cleanup_old_requests(max_age_hours)
