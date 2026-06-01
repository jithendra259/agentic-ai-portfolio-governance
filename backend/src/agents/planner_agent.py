"""
Planner Agent

Creates high-level execution plans for user queries.
Decomposes goals into subtasks with dependency tracking.
"""

import logging
from typing import Any, Dict

from src.agents.agent_base import (
    Agent,
    AgentConfig,
    AgentOutput,
    AgentType,
    ExecutionStatus,
    SpecializedAgent,
    TaskDefinition,
)
from src.intent.intent_classifier import IntentClassifier

logger = logging.getLogger(__name__)


class PlannerAgent(SpecializedAgent):
    """
    Planner Agent: Creates execution plans for user queries.
    
    Responsibilities:
    - Analyze user intent
    - Determine required tasks
    - Create execution plan with dependencies
    - Estimate complexity
    - Store plan in memory
    """
    
    COMPLEXITY_SIMPLE = "LOW"
    COMPLEXITY_MODERATE = "MEDIUM"
    COMPLEXITY_COMPLEX = "HIGH"
    
    # Task templates for common request types
    TASK_TEMPLATES = {
        "TA_PLOT_RSI": [
            ("T001", AgentType.DATA, "Retrieve price data"),
            ("T002", AgentType.TECHNICAL_ANALYSIS, "Calculate RSI", ["T001"]),
        ],
        "TA_PLOT_MACD": [
            ("T001", AgentType.DATA, "Retrieve price data"),
            ("T002", AgentType.TECHNICAL_ANALYSIS, "Calculate MACD", ["T001"]),
        ],
        "TA_TREND_ANALYSIS": [
            ("T001", AgentType.DATA, "Retrieve price data"),
            ("T002", AgentType.TECHNICAL_ANALYSIS, "Calculate technical indicators", ["T001"]),
            ("T003", AgentType.REGIME_DETECTION, "Detect market regime", ["T002"]),
            ("T004", AgentType.EXPLAINABILITY, "Generate explanation", ["T002", "T003"]),
        ],
        "PORTFOLIO_ANALYSIS": [
            ("T001", AgentType.DATA, "Retrieve portfolio data"),
            ("T002", AgentType.TECHNICAL_ANALYSIS, "Analyze holdings", ["T001"]),
            ("T003", AgentType.INSTABILITY_ANALYSIS, "Assess instability", ["T001"]),
            ("T004", AgentType.REGIME_DETECTION, "Detect market regime", ["T001"]),
            ("T005", AgentType.GOVERNANCE, "Validate governance rules", ["T001"]),
            ("T006", AgentType.PORTFOLIO_OPTIMIZATION, "Optimize allocation", ["T001", "T002", "T003", "T004"]),
            ("T007", AgentType.GOVERNANCE, "Final governance check", ["T006"]),
            ("T008", AgentType.EXPLAINABILITY, "Generate explanation", ["T002", "T003", "T004", "T006"]),
        ],
        "RISK_ASSESSMENT": [
            ("T001", AgentType.DATA, "Retrieve market data"),
            ("T002", AgentType.TECHNICAL_ANALYSIS, "Calculate risk metrics", ["T001"]),
            ("T003", AgentType.INSTABILITY_ANALYSIS, "Assess instability", ["T001"]),
            ("T004", AgentType.REGIME_DETECTION, "Detect market regime", ["T002"]),
            ("T005", AgentType.EXPLAINABILITY, "Generate explanation", ["T002", "T003", "T004"]),
        ],
    }
    
    def __init__(self, config: AgentConfig):
        """Initialize planner agent."""
        super().__init__(config)
        self.intent_classifier = IntentClassifier()
    
    async def execute(
        self,
        task: TaskDefinition,
        context: Dict[str, Any],
    ) -> AgentOutput:
        """
        Create an execution plan.
        
        Args:
            task: Task definition with user query
            context: Execution context
        
        Returns:
            AgentOutput with plan
        """
        output = self._create_output(task.inputs.get("request_id", ""))
        
        try:
            # Validate inputs
            if not self._validate_inputs(task.inputs, ["user_query"]):
                output.mark_failed("Missing required input: user_query", "INVALID_INPUT")
                return output
            
            user_query = task.inputs["user_query"]
            
            # Classify intent
            self._log_info(f"Analyzing user query: {user_query[:100]}")
            intent_type, confidence = self.intent_classifier.classify(user_query)
            
            output.add_evidence(f"Intent classified as {intent_type} with {confidence:.1%} confidence")
            
            # Create plan based on intent
            plan = self._create_plan(intent_type, user_query, task.inputs)
            
            output.mark_success({
                "plan_id": plan["plan_id"],
                "user_intent": intent_type,
                "intent_confidence": confidence,
                "tasks": plan["tasks"],
                "complexity": plan["complexity"],
                "total_tasks": len(plan["tasks"]),
                "task_count": len(plan["tasks"]),
                "requires_governance_approval": plan["requires_governance_approval"],
            })
            
            output.add_evidence(
                f"Created plan with {len(plan['tasks'])} tasks, "
                f"complexity: {plan['complexity']}"
            )
            output.confidence = confidence
            
        except Exception as e:
            self._log_error(f"Planning failed: {str(e)}")
            output.mark_failed(f"Planning error: {str(e)}", "PLANNING_ERROR")
        
        return output
    
    def _create_plan(
        self,
        intent_type: str,
        user_query: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create execution plan based on intent.
        
        Returns:
            Plan dictionary
        """
        from uuid import uuid4
        
        plan_id = str(uuid4())
        
        # Get template or create custom plan
        template = self.TASK_TEMPLATES.get(intent_type)
        
        if template:
            tasks = self._build_tasks_from_template(template, inputs)
            complexity = self._estimate_complexity(len(tasks))
        else:
            # Default: data + technical analysis
            tasks = self._build_default_plan(inputs)
            complexity = self.COMPLEXITY_MODERATE
        
        # Determine if governance approval needed
        requires_governance = any(
            task["agent_type"] in [AgentType.GOVERNANCE, AgentType.PORTFOLIO_OPTIMIZATION]
            for task in tasks
        )
        
        return {
            "plan_id": plan_id,
            "intent_type": intent_type,
            "user_query": user_query,
            "tasks": tasks,
            "task_count": len(tasks),
            "complexity": complexity,
            "requires_governance_approval": requires_governance,
        }
    
    def _build_tasks_from_template(
        self,
        template: list,
        inputs: Dict[str, Any],
    ) -> list:
        """Build tasks from template."""
        tasks = []
        
        for task_id, agent_type, description, *deps_list in template:
            depends_on = deps_list[0] if deps_list else []
            
            task = {
                "task_id": task_id,
                "agent_type": agent_type.value,
                "agent_name": agent_type.value,
                "description": description,
                "priority": int(task_id[1:]) // 100,
                "depends_on": depends_on,
                "inputs": inputs.copy(),
                "timeout_seconds": 30,
            }
            tasks.append(task)
        
        return tasks
    
    def _build_default_plan(self, inputs: Dict[str, Any]) -> list:
        """Build default plan for unrecognized intents."""
        return [
            {
                "task_id": "T001",
                "agent_type": AgentType.DATA.value,
                "agent_name": AgentType.DATA.value,
                "description": "Retrieve data",
                "priority": 1,
                "depends_on": [],
                "inputs": inputs,
                "timeout_seconds": 30,
            },
            {
                "task_id": "T002",
                "agent_type": AgentType.TECHNICAL_ANALYSIS.value,
                "agent_name": AgentType.TECHNICAL_ANALYSIS.value,
                "description": "Perform technical analysis",
                "priority": 2,
                "depends_on": ["T001"],
                "inputs": inputs,
                "timeout_seconds": 30,
            },
        ]
    
    def _estimate_complexity(self, task_count: int) -> str:
        """Estimate plan complexity based on task count."""
        if task_count <= 2:
            return self.COMPLEXITY_SIMPLE
        elif task_count <= 4:
            return self.COMPLEXITY_MODERATE
        else:
            return self.COMPLEXITY_COMPLEX
