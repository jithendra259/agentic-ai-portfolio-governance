"""
Task Execution Engine

Orchestrates the execution of tasks with dependency management,
error handling, and auditability tracking.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from src.agents.agent_base import (
    AgentType,
    ExecutionStatus,
    TaskDefinition,
    TaskResult,
    Agent,
    SpecializedAgent,
)

logger = logging.getLogger(__name__)


class TaskExecutionError(Exception):
    """Raised when task execution fails."""
    pass


class DependencyError(Exception):
    """Raised when task dependencies cannot be resolved."""
    pass


@dataclass
class ExecutionPlan:
    """A plan for executing multiple tasks."""
    plan_id: str = field(default_factory=lambda: str(uuid4()))
    request_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    tasks: List[TaskDefinition] = field(default_factory=list)
    task_by_id: Dict[str, TaskDefinition] = field(default_factory=dict)
    
    # Dependency graph
    dependencies: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    dependents: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    def add_task(self, task: TaskDefinition) -> None:
        """Add a task to the plan."""
        self.tasks.append(task)
        self.task_by_id[task.task_id] = task
        
        # Update dependency graph
        for dep_id in task.depends_on:
            self.dependencies[task.task_id].add(dep_id)
            self.dependents[dep_id].add(task.task_id)
    
    def get_executable_tasks(self, completed: Set[str]) -> List[TaskDefinition]:
        """Get tasks that can be executed (dependencies satisfied)."""
        executable = []
        for task in self.tasks:
            if task.task_id in completed:
                continue  # Already completed
            
            # Check if all dependencies are satisfied
            if all(dep_id in completed for dep_id in task.depends_on):
                executable.append(task)
        
        # Sort by priority (lower number = higher priority)
        executable.sort(key=lambda t: t.priority)
        return executable
    
    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate the execution plan.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check for circular dependencies
        if self._has_circular_dependency():
            errors.append("Circular dependency detected in task graph")
        
        # Check for missing dependencies
        for task_id, deps in self.dependencies.items():
            for dep_id in deps:
                if dep_id not in self.task_by_id:
                    errors.append(f"Task {task_id} depends on non-existent task {dep_id}")
        
        return len(errors) == 0, errors
    
    def _has_circular_dependency(self) -> bool:
        """Check if there are circular dependencies."""
        visited = set()
        rec_stack = set()
        
        def visit(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            
            for dep_id in self.dependencies.get(task_id, []):
                if dep_id not in visited:
                    if visit(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        for task_id in self.task_by_id:
            if task_id not in visited:
                if visit(task_id):
                    return True
        
        return False


@dataclass
class ExecutionContext:
    """Context for executing a plan."""
    request_id: str
    plan: ExecutionPlan
    
    # Execution state
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    skipped_tasks: Set[str] = field(default_factory=set)
    
    # Results
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    
    # Timeline
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def get_task_output(self, task_id: str) -> Optional[Any]:
        """Get the output of a completed task."""
        result = self.task_results.get(task_id)
        return result.agent_output.data if result else None
    
    def get_all_outputs(self) -> Dict[str, Any]:
        """Get outputs of all completed tasks as context for dependent tasks."""
        return {
            task_id: result.agent_output.data
            for task_id, result in self.task_results.items()
            if task_id in self.completed_tasks
        }
    
    def mark_task_completed(self, task_id: str, result: TaskResult) -> None:
        """Mark a task as completed."""
        self.completed_tasks.add(task_id)
        self.task_results[task_id] = result
    
    def mark_task_failed(self, task_id: str, result: TaskResult) -> None:
        """Mark a task as failed."""
        self.failed_tasks.add(task_id)
        self.task_results[task_id] = result
    
    def mark_task_skipped(self, task_id: str) -> None:
        """Mark a task as skipped (due to dependency failure)."""
        self.skipped_tasks.add(task_id)
    
    def mark_complete(self) -> None:
        """Mark execution as complete."""
        self.completed_at = datetime.utcnow()
    
    def is_complete(self) -> bool:
        """Check if execution is complete."""
        total_tasks = len(self.plan.tasks)
        executed = len(self.completed_tasks) + len(self.failed_tasks) + len(self.skipped_tasks)
        return executed == total_tasks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'plan_id': self.plan.plan_id,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'completed_count': len(self.completed_tasks),
            'failed_count': len(self.failed_tasks),
            'skipped_count': len(self.skipped_tasks),
            'total_count': len(self.plan.tasks),
            'results': {
                task_id: result.to_dict()
                for task_id, result in self.task_results.items()
            }
        }


class TaskExecutor:
    """
    Executes tasks with dependency management.
    
    Features:
    - Respects task dependencies
    - Parallel execution where possible
    - Graceful failure handling
    - Complete audit trail
    """
    
    def __init__(self, agents: Dict[AgentType, Agent]):
        """
        Initialize executor with available agents.
        
        Args:
            agents: Mapping of agent types to agent instances
        """
        self.agents = agents
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, plan: ExecutionPlan) -> ExecutionContext:
        """
        Execute a task plan.
        
        Args:
            plan: The execution plan
        
        Returns:
            ExecutionContext with results
        """
        # Validate plan
        is_valid, errors = plan.validate()
        if not errors:
            self.logger.error(f"Invalid execution plan: {errors}")
            raise DependencyError(f"Plan validation failed: {errors}")
        
        # Create execution context
        context = ExecutionContext(request_id=str(uuid4()), plan=plan)
        
        self.logger.info(f"Starting execution of plan {plan.plan_id} with {len(plan.tasks)} tasks")
        
        # Execute tasks
        while not context.is_complete():
            # Get tasks ready to execute
            executable = plan.get_executable_tasks(context.completed_tasks)
            
            if not executable:
                # Check if we're stuck (no executable, but not complete)
                if context.failed_tasks:
                    # Dependencies failed, so cascade skip
                    remaining = [
                        t for t in plan.tasks
                        if t.task_id not in context.completed_tasks
                        and t.task_id not in context.failed_tasks
                        and t.task_id not in context.skipped_tasks
                    ]
                    for task in remaining:
                        # Check if any dependency failed
                        if any(dep in context.failed_tasks for dep in task.depends_on):
                            context.mark_task_skipped(task.task_id)
                            self.logger.info(f"Skipped task {task.task_id} (dependency failed)")
                    continue
                else:
                    # No failed tasks but also no executable tasks - should not happen
                    break
            
            # Execute tasks in parallel where possible
            tasks = [self._execute_task(context, task) for task in executable]
            await asyncio.gather(*tasks)
        
        context.mark_complete()
        
        self.logger.info(
            f"Execution complete: {len(context.completed_tasks)} succeeded, "
            f"{len(context.failed_tasks)} failed, {len(context.skipped_tasks)} skipped"
        )
        
        return context
    
    async def _execute_task(self, context: ExecutionContext, task: TaskDefinition) -> None:
        """
        Execute a single task.
        
        Args:
            context: The execution context
            task: The task to execute
        """
        try:
            agent = self.agents.get(task.agent_type)
            if not agent:
                raise TaskExecutionError(f"No agent available for type {task.agent_type}")
            
            self.logger.info(f"Starting task {task.task_id} ({task.agent_name})")
            
            # Get context from completed tasks
            task_context = context.get_all_outputs()
            
            # Execute with timeout
            agent_output = await asyncio.wait_for(
                agent.execute(task, task_context),
                timeout=task.timeout_seconds
            )
            
            # Create result
            result = TaskResult(
                task_id=task.task_id,
                agent_output=agent_output,
                task_definition=task,
            )
            result.mark_complete()
            
            if agent_output.status == ExecutionStatus.SUCCESS:
                context.mark_task_completed(task.task_id, result)
                self.logger.info(f"Task {task.task_id} completed successfully ({result.duration_ms:.0f}ms)")
            else:
                context.mark_task_failed(task.task_id, result)
                self.logger.error(f"Task {task.task_id} failed: {agent_output.error}")
        
        except asyncio.TimeoutError:
            self.logger.error(f"Task {task.task_id} timed out after {task.timeout_seconds}s")
            result = TaskResult(
                task_id=task.task_id,
                agent_output=None,
                task_definition=task,
                error="TIMEOUT"
            )
            result.mark_complete()
            context.mark_task_failed(task.task_id, result)
        
        except Exception as e:
            self.logger.error(f"Task {task.task_id} error: {str(e)}", exc_info=True)
            result = TaskResult(
                task_id=task.task_id,
                agent_output=None,
                task_definition=task,
                error=str(e)
            )
            result.mark_complete()
            context.mark_task_failed(task.task_id, result)
