# Multi-Agent System Integration Guide

## Overview

This guide explains how to integrate the new multi-agent adaptive system into the existing portfolio governance chatbot while maintaining backward compatibility.

## Core Components Created

### 1. Foundation Layer (agent_base.py)
- `Agent` - Abstract base class for all agents
- `SpecializedAgent` - Base class with common patterns
- `AgentOutput` - Standardized output format
- `TaskDefinition` - Task specification
- `ExecutionStatus` - Enum for execution states

### 2. Execution Engine (task_execution_engine.py)
- `ExecutionPlan` - Task planning with dependencies
- `ExecutionContext` - Execution state management
- `TaskExecutor` - Executes tasks respecting dependencies
- Supports parallel execution where possible
- Implements dependency graph validation

### 3. Verification Framework (verification_framework.py)
- `VerificationFramework` - Anti-hallucination engine
- `VerificationReport` - Comprehensive verification results
- 8 verification check types
- 7 anti-hallucination rules enforced

### 4. Memory & Audit (memory_audit_system.py)
- `AuditLog` - Event logging for every decision
- `RequestMemory` - Per-request memory
- `MemoryManager` - Request lifecycle management
- Full auditability for compliance

### 5. Agent Implementations
Created in Phase 1:
- `PlannerAgent` - Creates execution plans
- `DataAgent` - Retrieves and validates data
- `VerificationAgent` - Anti-hallucination gatekeeper
- `ResponseAgent` - Assembles final response

Ready to create in Phase 2:
- `TechnicalAnalysisAgent`
- `RegimeDetectionAgent`
- `InstabilityAnalysisAgent`
- `GovernanceAgent`
- `ExplainabilityAgent`
- `PortfolioOptimizationAgent`

## System Architecture

```
User Query
    ↓
PlannerAgent
  ├─ Analyze intent
  ├─ Create task DAG
  └─ Store plan in memory
    ↓
TaskExecutor
  ├─ T001: DataAgent (price, portfolio, governance data)
  ├─ T002: TechnicalAnalysisAgent (indicators, signals)
  ├─ T003: RegimeDetectionAgent (market regime, confidence)
  ├─ T004: InstabilityAnalysisAgent (volatility, correlation drift)
  ├─ T005: PortfolioOptimizationAgent (allocation recommendation)
  ├─ T006: GovernanceAgent (compliance validation)
  ├─ T007: ExplainabilityAgent (explanations)
  └─ (parallel execution with dependency management)
    ↓
VerificationAgent
  ├─ Verify all outputs
  ├─ Check evidence
  ├─ Validate calculations
  ├─ Confirm governance
  └─ Block if issues found
    ↓
ResponseAgent
  └─ Assemble final response with audit trail
    ↓
AuditLog
  └─ Complete decision trail stored in memory/MongoDB
```

## Integration with Existing System

### Step 1: Install Multi-Agent Framework

The framework is now in place:
```
backend/src/agents/
├── agent_base.py                    # Foundation
├── task_execution_engine.py         # Orchestration
├── verification_framework.py        # Anti-hallucination
├── memory_audit_system.py          # Auditability
├── planner_agent.py                # Phase 1
├── data_agent.py                   # Phase 1
├── verification_agent.py           # Phase 1
├── response_agent.py               # Phase 1
```

### Step 2: Create Multi-Agent Manager

Create `backend/src/agents/multi_agent_manager.py`:

```python
import asyncio
from typing import Dict, Any

from src.agents.agent_base import Agent, AgentType, AgentConfig
from src.agents.task_execution_engine import TaskExecutor, ExecutionPlan, TaskDefinition
from src.agents.verification_framework import VerificationFramework
from src.agents.memory_audit_system import MemoryManager
from src.agents.planner_agent import PlannerAgent
from src.agents.data_agent import DataAgent
from src.agents.verification_agent import VerificationAgent
from src.agents.response_agent import ResponseAgent

class MultiAgentManager:
    """Manages multi-agent system execution."""
    
    def __init__(self):
        """Initialize multi-agent system."""
        self.memory_manager = MemoryManager()
        self.verification_framework = VerificationFramework()
        
        # Initialize agents
        self.agents = {
            AgentType.PLANNER: PlannerAgent(
                AgentConfig(
                    agent_type=AgentType.PLANNER,
                    name="PlannerAgent",
                    timeout_seconds=10,
                )
            ),
            AgentType.DATA: DataAgent(
                AgentConfig(
                    agent_type=AgentType.DATA,
                    name="DataAgent",
                    timeout_seconds=30,
                )
            ),
            AgentType.VERIFICATION: VerificationAgent(
                AgentConfig(
                    agent_type=AgentType.VERIFICATION,
                    name="VerificationAgent",
                    timeout_seconds=10,
                )
            ),
            AgentType.RESPONSE: ResponseAgent(
                AgentConfig(
                    agent_type=AgentType.RESPONSE,
                    name="ResponseAgent",
                    timeout_seconds=10,
                )
            ),
        }
        
        self.executor = TaskExecutor(self.agents)
    
    async def execute(self, user_query: str) -> Dict[str, Any]:
        """
        Execute multi-agent workflow.
        
        Args:
            user_query: User's natural language query
        
        Returns:
            Complete response with audit trail
        """
        # Create request in memory
        request_id = self.memory_manager.create_request(user_query)
        
        try:
            # Step 1: Planning
            planner = self.agents[AgentType.PLANNER]
            plan_task = TaskDefinition(
                task_id="T000_planning",
                agent_type=AgentType.PLANNER,
                agent_name="PlannerAgent",
                description="Create execution plan",
                inputs={"user_query": user_query, "request_id": request_id},
            )
            
            plan_output = await planner.execute(plan_task, {})
            plan_dict = plan_output.to_dict()
            self.memory_manager.store_plan(request_id, plan_dict)
            
            if not plan_output.data:
                return {
                    "error": "Failed to create execution plan",
                    "request_id": request_id,
                }
            
            # Step 2: Execute task plan
            plan = ExecutionPlan(request_id=request_id)
            plan.plan_id = plan_output.data.get("plan_id", "")
            
            for task_spec in plan_output.data.get("tasks", []):
                task = TaskDefinition(
                    task_id=task_spec["task_id"],
                    agent_type=AgentType[task_spec["agent_type"].upper()],
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
            
            # Store task results
            for task_id, result in execution_context.task_results.items():
                self.memory_manager.store_task_result(request_id, task_id, result.to_dict())
            
            # Step 3: Verification
            # (Create verification task with all outputs)
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
            
            verification_dict = verification_output.to_dict()
            self.memory_manager.store_verification(request_id, verification_dict)
            
            # Step 4: Response Assembly
            response_task = TaskDefinition(
                task_id="T999_response",
                agent_type=AgentType.RESPONSE,
                agent_name="ResponseAgent",
                description="Assemble final response",
                inputs={"request_id": request_id},
            )
            
            # Create response context with all outputs
            response_context = verification_context.copy()
            response_context["verification"] = verification_output.to_dict()
            
            response_agent = self.agents[AgentType.RESPONSE]
            response_output = await response_agent.execute(
                response_task,
                response_context,
            )
            
            response_dict = response_output.to_dict()
            self.memory_manager.store_response(request_id, response_dict)
            
            return {
                "status": "success",
                "request_id": request_id,
                "response": response_dict.get("data", {}),
                "audit_trail": self.memory_manager.get_audit_log(request_id).to_dict(),
            }
        
        except Exception as e:
            self.memory_manager.get_request(request_id).audit_log.log(
                component="MultiAgentManager",
                action="Execution failed",
                status="FAILED",
                data={"error": str(e)},
            )
            
            return {
                "status": "error",
                "request_id": request_id,
                "error": str(e),
                "audit_trail": self.memory_manager.get_audit_log(request_id).to_dict(),
            }
```

### Step 3: Create Stub Agents for Phase 2

For now, create stub implementations that return valid structures:

```python
# backend/src/agents/technical_analysis_agent.py
from src.agents.agent_base import Agent, AgentConfig, AgentOutput, TaskDefinition

class TechnicalAnalysisAgent(Agent):
    """Stub for Phase 2."""
    
    async def execute(self, task: TaskDefinition, context: dict) -> AgentOutput:
        output = self._create_output(task.inputs.get("request_id", ""))
        output.mark_success({
            "indicator_name": "TBD",
            "values": [],
            "signals": [],
        })
        output.confidence = 0.0  # Mark as not yet implemented
        return output
```

### Step 4: Integrate with Existing Orchestrator

In `backend/src/orchestrator/chatbot_orchestrator.py`, add:

```python
from src.agents.multi_agent_manager import MultiAgentManager

# Initialize multi-agent system
multi_agent_manager = MultiAgentManager()

# Add route to use multi-agent system
@router.post("/chat/multi-agent")
async def chat_multi_agent(message: str):
    """Route through multi-agent system."""
    result = await multi_agent_manager.execute(message)
    return result
```

### Step 5: Gradual Migration

1. **Phase 1 (Current)**: Multi-agent system operational for basic queries
2. **Phase 2**: Complete remaining agents
3. **Phase 3**: Route more complex queries to multi-agent
4. **Phase 4**: Migrate all queries (optional)

## API Usage Examples

### Example 1: Technical Analysis Request

```python
# Input
query = "Analyze AAPL trend for the last 6 months"

# Execution
result = await multi_agent_manager.execute(query)

# Output
{
    "status": "success",
    "request_id": "uuid...",
    "response": {
        "summary": "Analysis complete",
        "findings": {...},
        "recommendation": {...},
        "confidence_score": 0.85,
        "audit_trail": {...}
    }
}
```

### Example 2: Portfolio Analysis Request

```python
# Input
query = "Optimize my portfolio for growth given current market regime"

# Process
# 1. PlannerAgent creates complex plan with 8+ tasks
# 2. DataAgent retrieves portfolio + market data
# 3. TechnicalAnalysisAgent calculates indicators
# 4. RegimeDetectionAgent identifies bull/bear/sideways
# 5. InstabilityAnalysisAgent computes drift metrics
# 6. PortfolioOptimizationAgent recommends allocation
# 7. GovernanceAgent validates compliance
# 8. ExplainabilityAgent generates explanations
# 9. VerificationAgent gates recommendation
# 10. ResponseAgent assembles response

# Output
{
    "response": {
        "recommendation": {
            "action": "REBALANCE",
            "confidence": 0.78,
            "allocation_changes": [...]
        },
        "governance_review": {"status": "APPROVED"},
        "risks": ["High volatility", "Correlation shift"],
        "audit_trail": {"task_results": {...}}
    }
}
```

## Memory & Audit Trail

Every request is completely auditable:

```python
# Retrieve full execution record
record = multi_agent_manager.memory_manager.get_full_record(request_id)

# Structure:
{
    "request_id": "uuid",
    "user_query": "...",
    "plan": {...},                    # Execution plan
    "task_results": {                 # All task outputs
        "T001": {...},
        "T002": {...},
        ...
    },
    "verification_report": {...},     # Verification results
    "final_response": {...},          # User response
    "audit_trail": {                  # Complete decision trail
        "entries": [
            {
                "timestamp": "...",
                "component": "PlannerAgent",
                "action": "Created execution plan",
                "status": "SUCCESS",
                "confidence": 0.95
            },
            ...
        ]
    }
}
```

## Anti-Hallucination Enforcement

The verification framework prevents hallucinations through 7 rules:

1. **Never invent values** - Data agent returns FAILED if unavailable
2. **Data-driven recommendations** - Every claim needs source data
3. **Evidence required** - Every signal backed by calculation
4. **Governance approval** - Required before any recommendation
5. **Confidence scoring** - Mandatory for all outputs
6. **Low confidence handling** - If < 0.5, return warning
7. **Completeness check** - Block if any critical component missing

## Testing the System

```python
# Test suite
import asyncio
from src.agents.multi_agent_manager import MultiAgentManager

async def test():
    manager = MultiAgentManager()
    
    # Test 1: Simple query
    result = await manager.execute("What is the current trend for AAPL?")
    assert result["status"] == "success"
    
    # Test 2: Complex query
    result = await manager.execute("Optimize my portfolio")
    assert "audit_trail" in result
    
    # Test 3: Invalid query
    result = await manager.execute("invalid query")
    assert "request_id" in result  # Still tracked

asyncio.run(test())
```

## Next Steps

1. ✅ Create core framework
2. ✅ Implement critical agents (Planner, Data, Verification, Response)
3. ⏳ Create multi-agent manager
4. ⏳ Implement remaining agents (Phase 2)
5. ⏳ Integration with existing orchestrator
6. ⏳ Testing & validation
7. ⏳ Performance optimization

## Performance Targets

- **Response Time**: < 10 seconds (90th percentile)
- **Hallucination Rate**: < 5%
- **Explanation Quality**: 90%+ user satisfaction
- **Governance Compliance**: 100% (zero violations)
- **Auditability**: 100% decision trail preservation

## Troubleshooting

### Issue: Task fails but no blocking error

**Solution**: Check verification report in audit trail. Each failed check is logged.

### Issue: Recommendation blocked

**Solution**: Review verification report. Most likely cause: governance approval failed or confidence too low.

### Issue: Memory growing unbounded

**Solution**: Call `memory_manager.cleanup_old_requests(max_age_hours=24)` periodically.

### Issue: Performance degradation

**Solution**: Enable parallel execution of independent tasks. TaskExecutor already supports this.

## Monitoring

Monitor these metrics:

- **Execution time by agent** (in audit trail)
- **Verification pass rate** (goal: > 95%)
- **Recommendation approval rate** (governance)
- **Confidence distribution** (should be > 0.7 for most)
- **Memory usage** (run cleanup if exceeding 5GB)

## Security Considerations

1. ✅ No fabricated values (enforced by data agent)
2. ✅ No unsupported claims (enforced by verification agent)
3. ✅ Complete audit trail (for compliance)
4. ✅ Governance validation (required for recommendations)
5. ⚠️  Implement rate limiting at orchestrator level
6. ⚠️  Add request ID validation
7. ⚠️  Sanitize user inputs before planner

