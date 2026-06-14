"""
Agentic Task Execution System

Implements autonomous agent architecture with:
- ReAct (Reasoning and Acting) Loop
- Hierarchical Task Network (HTN) Planning
- Dynamic DAG execution with real-time replanning
- Vector-based episodic memory with HNSW search
- Blackboard Architecture for shared state
- Safety guardrails (loop detection, context management, schema validation)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import operator
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Callable, Dict, List, Optional, Set, Tuple, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph
from pymongo import MongoClient

from src.guardrails.loop_detector import SemanticLoopDetector
from src.agents.state_reducers import append_list, merge_blackboard_state
from src.agents.htn_plan_parser import (
    PlanValidationError,
    build_sequential_fallback_plan,
    extract_task_descriptions,
    parse_execution_plan,
    validate_execution_plan,
)
from src.agents.token_budget import (
    DEFAULT_MAX_TOKEN_BUDGET,
    cap_tool_output,
    check_token_budget,
    update_token_ledger,
)
from src.memory.audit_log import GLOBAL_AUDIT_LOGGER
from src.memory.calculation_scratchpad import CalculationScratchpad
from src.memory.state_sanitizer import sanitize_for_mongodb

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class TaskStatus(Enum):
    """Status of a task in the execution graph."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class AgentRole(Enum):
    """Specialized agent roles in the multi-agent system."""
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    MEMORY_MANAGER = "memory_manager"
    RECOVERY = "recovery"


class ActionType(Enum):
    """Types of actions an agent can take."""
    THINK = "think"
    CALL_TOOL = "call_tool"
    OBSERVE = "observe"
    UPDATE_PLAN = "update_plan"
    REQUEST_HUMAN_INPUT = "request_human_input"


class AgenticExecutorState(TypedDict, total=False):
    blackboard: Annotated[Dict[str, Any], merge_blackboard_state]
    goal: str
    context: Annotated[Dict[str, Any], merge_blackboard_state]
    iteration: int
    max_iterations: int
    total_tokens_used: int
    max_token_budget: int
    budget_depleted: bool
    critic_decision: str
    execution_history: Annotated[List[Dict[str, Any]], append_list]
    messages: Annotated[List[Dict[str, Any]], operator.add]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TaskNode:
    """
    A node in the Dynamic DAG representing a task.
    
    Unlike traditional DAGs, this can be mutated at runtime by the Planner Agent.
    """
    task_id: str
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5  # 1=highest, 10=lowest
    depends_on: List[str] = field(default_factory=list)
    agent_role: AgentRole = AgentRole.EXECUTOR
    tool_name: Optional[str] = None
    tool_params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'priority': self.priority,
            'depends_on': self.depends_on,
            'agent_role': self.agent_role.value,
            'tool_name': self.tool_name,
            'tool_params': self.tool_params,
            'result': self.result,
            'error': self.error,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata,
        }


@dataclass
class ReActStep:
    """A single step in the ReAct loop."""
    step_number: int
    action_type: ActionType
    thought: str
    action: Optional[Dict[str, Any]] = None
    observation: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_number': self.step_number,
            'action_type': self.action_type.value,
            'thought': self.thought,
            'action': self.action,
            'observation': self.observation,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class EpisodicMemory:
    """
    Episodic memory entry for vector storage.
    
    Stored in MongoDB with vector index for HNSW-like similarity search.
    """
    memory_id: str = field(default_factory=lambda: str(uuid4()))
    task_description: str = ""
    action_taken: str = ""
    outcome: str = ""
    success: bool = False
    error_type: Optional[str] = None
    embedding: List[float] = field(default_factory=list)  # To be computed by embedding model
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'memory_id': self.memory_id,
            'task_description': self.task_description,
            'action_taken': self.action_taken,
            'outcome': self.outcome,
            'success': self.success,
            'error_type': self.error_type,
            'embedding': self.embedding,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata,
        }


# ============================================================================
# BLACKBOARD ARCHITECTURE (Shared State Graph)
# ============================================================================

class BlackboardState:
    """
    Centralized shared state for multi-agent coordination.
    
    Implements the Blackboard Architecture pattern where specialized agents
    read and write information to coordinate complex task execution.
    """
    
    def __init__(self, request_id: str, goal: str):
        self.request_id = request_id
        self.goal = goal
        self.created_at = datetime.utcnow()
        
        # Task graph
        self.tasks: Dict[str, TaskNode] = {}
        self.task_order: List[str] = []  # Maintains insertion order
        
        # Execution state
        self.current_phase: str = "planning"
        self.active_task_id: Optional[str] = None
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self.skipped_tasks: Set[str] = set()
        
        # ReAct history for current task
        self.react_history: List[ReActStep] = []
        
        # Shared context and results
        self.context: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.calculation_scratchpad = CalculationScratchpad()
        
        # Error tracking for loop detection
        self.action_history: deque = deque(maxlen=100)  # Last 100 actions
        self.loop_detected: bool = False
        self.loop_events: List[Dict[str, Any]] = []
        
        # Memory references
        self.retrieved_memories: List[EpisodicMemory] = []
        
        # Human-in-the-loop
        self.pending_human_inputs: List[Dict[str, Any]] = []
        
        # Metadata
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.status: str = "initialized"
    
    def add_task(self, task: TaskNode) -> None:
        """Add a task to the blackboard."""
        self.tasks[task.task_id] = task
        self.task_order.append(task.task_id)
        task.updated_at = datetime.utcnow()
    
    def update_task(self, task_id: str, **kwargs) -> Optional[TaskNode]:
        """Update a task's attributes."""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)
        task.updated_at = datetime.utcnow()
        return task
    
    def get_task(self, task_id: str) -> Optional[TaskNode]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def get_executable_tasks(self) -> List[TaskNode]:
        """
        Get tasks that are ready to execute (all dependencies satisfied).
        
        Returns tasks sorted by priority (lower number = higher priority).
        """
        executable = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            deps_satisfied = all(
                dep_id in self.completed_tasks 
                for dep_id in task.depends_on
            )
            
            if deps_satisfied:
                executable.append(task)
        
        # Sort by priority
        executable.sort(key=lambda t: t.priority)
        return executable
    
    def add_react_step(self, step: ReActStep) -> None:
        """Add a ReAct step to the history."""
        self.react_history.append(step)
    
    def add_action_to_history(self, action: Dict[str, Any]) -> None:
        """Add action to history for loop detection."""
        self.action_history.append({
            'action': action,
            'timestamp': datetime.utcnow(),
        })

    def add_loop_event(self, event: Dict[str, Any]) -> None:
        """Record a semantic loop detection event for audit/debugging."""
        self.loop_detected = True
        self.loop_events.append({
            **event,
            'timestamp': datetime.utcnow().isoformat(),
        })

    def record_calculation(
        self,
        *,
        label: str,
        formula: str,
        inputs: Dict[str, Any] | None = None,
        result: Any = None,
        source: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Persist exact calculation details outside compressed chat context."""
        return self.calculation_scratchpad.record(
            label=label,
            formula=formula,
            inputs=inputs,
            result=result,
            source=source,
            metadata=metadata,
        )
    
    def detect_loop(self) -> bool:
        """
        Detect if the agent is stuck in an infinite loop.
        
        Uses a simple heuristic: if the same action (same tool + params)
        is attempted 3 times in the last 5 actions, it's likely a loop.
        """
        if len(self.action_history) < 3:
            return False
        
        recent_actions = list(self.action_history)[-5:]
        action_signatures = []
        
        for item in recent_actions:
            action = item['action']
            # Create a hashable signature
            sig = json.dumps(action, sort_keys=True, default=str)
            action_signatures.append(sig)
        
        # Check for repetitions
        from collections import Counter
        counts = Counter(action_signatures)
        most_common_count = counts.most_common(1)[0][1] if counts else 0
        
        return most_common_count >= 3
    
    def get_summary_context(self, max_steps: int = 10) -> Dict[str, Any]:
        """
        Get a summarized context to prevent context window overflow.
        
        Uses sliding window approach: summarizes older steps, keeps recent detailed.
        """
        total_tasks = len(self.tasks)
        completed = len(self.completed_tasks)
        
        # Summarize old ReAct steps if too many
        react_summary = None
        recent_react = self.react_history
        if len(self.react_history) > max_steps:
            react_summary = f"Completed {len(self.react_history) - max_steps} reasoning steps. "
            recent_react = self.react_history[-max_steps:]
        
        return {
            'goal': self.goal,
            'total_tasks': total_tasks,
            'completed': completed,
            'failed': len(self.failed_tasks),
            'pending': total_tasks - completed - len(self.failed_tasks) - len(self.skipped_tasks),
            'current_phase': self.current_phase,
            'active_task': self.active_task_id,
            'react_summary': react_summary,
            'recent_react_steps': [s.to_dict() for s in recent_react],
            'retrieved_memories': [m.to_dict() for m in self.retrieved_memories[:3]],  # Top 3
            'calculation_scratchpad': self.calculation_scratchpad.compact_summary(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            'request_id': self.request_id,
            'goal': self.goal,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status,
            'current_phase': self.current_phase,
            'active_task_id': self.active_task_id,
            'tasks': {tid: t.to_dict() for tid, t in self.tasks.items()},
            'task_order': self.task_order,
            'completed_tasks': list(self.completed_tasks),
            'failed_tasks': list(self.failed_tasks),
            'skipped_tasks': list(self.skipped_tasks),
            'react_history': [s.to_dict() for s in self.react_history],
            'context': self.context,
            'results': self.results,
            'action_history': list(self.action_history)[-20:],  # Last 20 for debugging
            'loop_detected': self.loop_detected,
            'loop_events': self.loop_events[-20:],
            'calculation_scratchpad': self.calculation_scratchpad.to_list(),
            'pending_human_inputs': self.pending_human_inputs,
        }


# ============================================================================
# VECTOR MEMORY STORE (MongoDB with Vector Index)
# ============================================================================

class VectorMemoryStore:
    """
    Vector-based episodic memory store using MongoDB.
    
    Implements HNSW-like similarity search via MongoDB's vector index.
    Stores past task executions for retrieval-augmented generation (RAG).
    """
    
    def __init__(self, mongo_uri: str, db_name: str = "agentic_memory"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["episodic_memories"]
        
        # Ensure vector index exists (MongoDB 6.0+)
        self._ensure_vector_index()
        
        logger.info("VectorMemoryStore initialized")
    
    def _ensure_vector_index(self) -> None:
        """Create vector index if it doesn't exist."""
        try:
            indexes = self.collection.index_information()
            if "embedding_vector_index" not in indexes:
                self.collection.create_index(
                    [("embedding", "vector")],
                    name="embedding_vector_index",
                    vectorOptions={
                        "dimensions": 1536,  # OpenAI embedding dimension
                        "similarity": "cosine",
                        "type": "vectorSearch"
                    }
                )
                logger.info("Created vector index on embedding field")
        except Exception as e:
            logger.warning(f"Could not create vector index: {e}. Will use fallback similarity search.")
    
    def store_memory(self, memory: EpisodicMemory) -> str:
        """Store an episodic memory."""
        doc = memory.to_dict()
        result = self.collection.insert_one(doc)
        logger.debug(f"Stored memory {memory.memory_id}")
        return str(result.inserted_id)
    
    def search_similar(
        self, 
        query_embedding: List[float], 
        limit: int = 5,
        filter_success: Optional[bool] = None
    ) -> List[EpisodicMemory]:
        """
        Search for similar memories using vector similarity.
        
        Args:
            query_embedding: The embedding vector to search for
            limit: Maximum number of results
            filter_success: If True, only successful memories; if False, only failures
        
        Returns:
            List of similar memories sorted by similarity
        """
        try:
            # MongoDB vector search (if available)
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "embedding_vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 10,
                        "limit": limit
                    }
                }
            ]
            
            if filter_success is not None:
                pipeline[0]["$vectorSearch"]["filter"] = {"success": filter_success}
            
            results = list(self.collection.aggregate(pipeline))
            
        except Exception as e:
            logger.warning(f"Vector search failed: {e}. Using fallback.")
            # Fallback: retrieve all and compute cosine similarity in Python
            results = self._fallback_similarity_search(
                query_embedding, limit, filter_success
            )
        
        memories = []
        for doc in results:
            doc['created_at'] = datetime.fromisoformat(doc['created_at'])
            memory = EpisodicMemory(**doc)
            memories.append(memory)
        
        return memories
    
    def _fallback_similarity_search(
        self,
        query_embedding: List[float],
        limit: int,
        filter_success: Optional[bool]
    ) -> List[Dict[str, Any]]:
        """Fallback cosine similarity search when vector index unavailable."""
        query = {}
        if filter_success is not None:
            query['success'] = filter_success
        
        # Retrieve candidates
        candidates = list(self.collection.find(query).limit(limit * 20))
        
        if not candidates or not query_embedding:
            return candidates[:limit]
        
        # Compute cosine similarity
        import math
        
        def cosine_similarity(v1: List[float], v2: List[float]) -> float:
            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        
        scored = []
        for doc in candidates:
            if doc.get('embedding'):
                score = cosine_similarity(query_embedding, doc['embedding'])
                scored.append((score, doc))
        
        # Sort by similarity descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:limit]]
    
    def get_failure_patterns(self, task_description: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Retrieve memories of similar failed tasks for learning."""
        # This would use embeddings in production
        # For now, simple keyword matching
        keywords = task_description.lower().split()
        
        query = {
            'success': False,
            '$or': [
                {'task_description': {'$regex': keyword, '$options': 'i'}}
                for keyword in keywords[:3]  # Use first 3 keywords
            ]
        }
        
        results = list(self.collection.find(query).limit(limit))
        for doc in results:
            doc['created_at'] = datetime.fromisoformat(doc['created_at'])
        
        return results


# ============================================================================
# HTN PLANNER (Hierarchical Task Network)
# ============================================================================

class HTNPlanner:
    """
    Hierarchical Task Network planner for decomposing high-level goals.
    
    Breaks abstract goals into primitive executable actions using a library
    of methods and operators.
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.method_library = self._build_method_library()
        logger.info("HTNPlanner initialized with method library")
    
    def _build_method_library(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build a library of decomposition methods.
        
        In production, this could be learned from experience or configured.
        """
        return {
            "research_and_summarize": [
                {
                    "name": "standard_research",
                    "preconditions": ["has_search_tool"],
                    "subtasks": [
                        {"name": "search_web", "tool": "web_search"},
                        {"name": "evaluate_sources", "tool": "source_evaluator"},
                        {"name": "extract_key_points", "tool": "text_extractor"},
                        {"name": "synthesize_summary", "tool": "text_synthesizer"},
                    ]
                }
            ],
            "execute_analysis": [
                {
                    "name": "standard_analysis",
                    "preconditions": ["has_data_access"],
                    "subtasks": [
                        {"name": "fetch_data", "tool": "data_fetcher"},
                        {"name": "validate_data", "tool": "data_validator"},
                        {"name": "compute_metrics", "tool": "metrics_calculator"},
                        {"name": "generate_insights", "tool": "insight_generator"},
                    ]
                }
            ],
            "create_content": [
                {
                    "name": "standard_creation",
                    "preconditions": [],
                    "subtasks": [
                        {"name": "outline_structure", "tool": "outliner"},
                        {"name": "draft_content", "tool": "content_writer"},
                        {"name": "review_quality", "tool": "quality_checker"},
                        {"name": "finalize", "tool": "finalizer"},
                    ]
                }
            ],
            "organize_event": [
                {
                    "name": "standard_event_planning",
                    "preconditions": [],
                    "subtasks": [
                        {"name": "define_requirements", "tool": "requirements_analyzer"},
                        {"name": "book_venue", "tool": "venue_booker"},
                        {"name": "schedule_speakers", "tool": "scheduler"},
                        {"name": "send_invitations", "tool": "notification_sender"},
                        {"name": "coordinate_logistics", "tool": "logistics_coordinator"},
                    ]
                }
            ],
            # Additional flexible patterns for better matching
            "research": [
                {
                    "name": "flexible_research",
                    "preconditions": [],
                    "subtasks": [
                        {"name": "search_information", "tool": "web_search"},
                        {"name": "analyze_findings", "tool": "analyzer"},
                        {"name": "compile_report", "tool": "report_writer"},
                    ]
                }
            ],
            "organize": [
                {
                    "name": "flexible_organization",
                    "preconditions": [],
                    "subtasks": [
                        {"name": "plan_requirements", "tool": "planner"},
                        {"name": "coordinate_resources", "tool": "coordinator"},
                        {"name": "execute_plan", "tool": "executor"},
                    ]
                }
            ],
        }
    
    def decompose_goal(
        self, 
        goal: str, 
        context: Dict[str, Any] = None
    ) -> List[TaskNode]:
        """
        Decompose a high-level goal into primitive tasks using HTN planning.
        
        In production, this would use LLM to select appropriate methods and
        instantiate subtasks. Here we use a combination of pattern matching
        and LLM-assisted decomposition.
        
        Args:
            goal: The high-level goal to decompose
            context: Additional context for decomposition
        
        Returns:
            List of TaskNode objects forming a DAG
        """
        context = context or {}
        
        # Try to match goal to known patterns
        goal_lower = goal.lower()
        matched_method = None
        
        for pattern, methods in self.method_library.items():
            if pattern.replace("_", " ") in goal_lower or pattern in goal_lower:
                matched_method = methods[0]  # Use first method
                break
        
        if matched_method:
            # Use predefined decomposition
            tasks = self._instantiate_method(matched_method, goal, context)
        else:
            # Use LLM to generate decomposition
            tasks = self._llm_decompose(goal, context)
        
        # Validate the generated DAG
        self._validate_dag(tasks)
        
        logger.info(f"Decomposed goal '{goal}' into {len(tasks)} tasks")
        return tasks
    
    def _instantiate_method(
        self, 
        method: Dict[str, Any], 
        goal: str,
        context: Dict[str, Any]
    ) -> List[TaskNode]:
        """Instantiate a method's subtasks into TaskNodes."""
        tasks = []
        prev_task_id = None
        
        for idx, subtask in enumerate(method['subtasks']):
            task_id = f"{method['name']}_{idx}_{uuid4().hex[:8]}"
            
            task = TaskNode(
                task_id=task_id,
                name=subtask['name'],
                description=f"Execute {subtask['name']} for goal: {goal}",
                priority=idx + 1,  # Sequential priority
                depends_on=[prev_task_id] if prev_task_id else [],
                tool_name=subtask.get('tool'),
                metadata={'method': method['name'], 'subtask_index': idx}
            )
            
            tasks.append(task)
            prev_task_id = task_id
        
        return tasks
    
    def _llm_decompose(self, goal: str, context: Dict[str, Any]) -> List[TaskNode]:
        """Use LLM structured planning with validation and sequential fallback."""
        if not self.llm_client:
            return self._execution_plan_to_tasks(
                build_sequential_fallback_plan(goal),
                decomposition_method="sequential_fallback",
            )

        last_error = ""
        fallback_steps: List[str] = []
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                prompt = self._build_planner_prompt(goal, context, last_error)
                raw_plan = self._invoke_planner_llm(prompt)
                fallback_steps = extract_task_descriptions(raw_plan) or fallback_steps
                plan = validate_execution_plan(parse_execution_plan(raw_plan))
                return self._execution_plan_to_tasks(
                    plan,
                    decomposition_method="llm_structured",
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "HTN plan generation failed on attempt %s/%s: %s",
                    attempt,
                    max_attempts,
                    last_error,
                )

        logger.warning("HTN planner degraded to sequential fallback for goal: %s", goal)
        return self._execution_plan_to_tasks(
            build_sequential_fallback_plan(goal, fallback_steps),
            decomposition_method="sequential_fallback",
            error=last_error,
        )

    def _invoke_planner_llm(self, prompt: str) -> Any:
        if hasattr(self.llm_client, "with_structured_output"):
            structured = self.llm_client.with_structured_output(dict)
            return structured.invoke(prompt)
        if hasattr(self.llm_client, "invoke"):
            return self.llm_client.invoke(prompt)
        if callable(self.llm_client):
            return self.llm_client(prompt)
        raise PlanValidationError("Planner LLM client must be callable or expose invoke()")

    def _build_planner_prompt(
        self,
        goal: str,
        context: Dict[str, Any],
        previous_error: str = "",
    ) -> str:
        prompt = [
            "Create a valid JSON execution DAG for this portfolio governance goal.",
            f"Goal: {goal}",
            f"Context: {json.dumps(context, sort_keys=True, default=str)}",
            "Return schema: {\"goal\": str, \"nodes\": [{\"task_id\": str, \"description\": str, \"dependencies\": [str], \"name\": str, \"tool\": str}]}",
            "The graph must be acyclic and every dependency must reference an existing task_id.",
        ]
        if previous_error:
            prompt.append(f"Your previous plan failed validation with this error: {previous_error}. Fix it and try again.")
        return "\n".join(prompt)

    def _execution_plan_to_tasks(
        self,
        plan,
        *,
        decomposition_method: str,
        error: str | None = None,
    ) -> List[TaskNode]:
        tasks: List[TaskNode] = []
        for index, node in enumerate(plan.nodes, start=1):
            task = TaskNode(
                task_id=node.task_id,
                name=node.name or node.task_id,
                description=node.description,
                priority=index,
                depends_on=list(node.dependencies),
                tool_name=node.tool or "generic_executor",
                metadata={
                    'decomposition_method': decomposition_method,
                    'planner_goal': plan.goal,
                    **({'planner_error': error} if error else {}),
                },
            )
            tasks.append(task)
        return tasks
    
    def _validate_dag(self, tasks: List[TaskNode]) -> None:
        """Validate that the task graph is a valid DAG (no cycles)."""
        task_ids = {t.task_id for t in tasks}
        
        # Check for missing dependencies
        for task in tasks:
            for dep_id in task.depends_on:
                if dep_id not in task_ids:
                    logger.warning(f"Task {task.task_id} depends on non-existent task {dep_id}")
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str, adj_map: Dict[str, List[str]]) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            
            for neighbor in adj_map.get(task_id, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, adj_map):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        # Build adjacency map (task -> dependents)
        adj_map = defaultdict(list)
        for task in tasks:
            for dep_id in task.depends_on:
                adj_map[dep_id].append(task.task_id)
        
        for task in tasks:
            if task.task_id not in visited:
                if has_cycle(task.task_id, adj_map):
                    raise ValueError("Circular dependency detected in task graph")


# ============================================================================
# SAFETY GUARDRAILS
# ============================================================================

class SafetyGuardrails:
    """
    Safety mechanisms to prevent common agentic failures.
    """
    
    @staticmethod
    def validate_action_schema(action: Dict[str, Any], allowed_tools: Set[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate that an action conforms to expected schema.
        
        Prevents tool hallucination by enforcing strict schema validation.
        """
        # Check required fields
        if 'tool' not in action:
            return False, "Missing 'tool' field in action"
        
        if 'parameters' not in action:
            return False, "Missing 'parameters' field in action"
        
        # Check tool exists
        tool_name = action['tool']
        if tool_name not in allowed_tools:
            return False, f"Unknown tool: {tool_name}. Available tools: {allowed_tools}"
        
        # Check parameters is a dict
        if not isinstance(action['parameters'], dict):
            return False, "Parameters must be a dictionary"
        
        return True, None
    
    @staticmethod
    def summarize_context(history: List[Dict[str, Any]], max_length: int = 4000) -> str:
        """
        Summarize long execution history to fit within context window.
        
        Uses sliding window: keeps recent steps detailed, summarizes older ones.
        """
        if len(history) <= 10:
            return json.dumps(history, indent=2)
        
        # Keep last 10 steps detailed
        recent = history[-10:]
        
        # Summarize older steps
        older = history[:-10]
        summary = {
            'summary': f"Executed {len(older)} previous steps",
            'outcomes': defaultdict(int)
        }
        for step in older:
            outcome = step.get('observation', {}).get('status', 'unknown')
            summary['outcomes'][outcome] += 1
        
        return json.dumps([summary] + recent, indent=2)[:max_length]
    
    @staticmethod
    def check_infinite_loop(action_history: deque, threshold: int = 3) -> bool:
        """Check if agent is stuck in an infinite loop."""
        if len(action_history) < threshold:
            return False
        
        recent = list(action_history)[-threshold:]
        signatures = [json.dumps(a['action'], sort_keys=True, default=str) for a in recent]
        
        # If all recent actions are identical, likely a loop
        return len(set(signatures)) == 1


# ============================================================================
# MAIN AGENTIC EXECUTOR (LangGraph Integration)
# ============================================================================

class AgenticTaskExecutor:
    """
    Main orchestrator for agentic task execution.
    
    Combines:
    - LangGraph for state machine orchestration
    - HTN Planner for task decomposition
    - Vector Memory for episodic recall
    - ReAct loop for reasoning and action
    - Safety guardrails for reliability
    """
    
    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017",
        llm_client=None,
        allowed_tools: Set[str] = None
    ):
        self.mongo_uri = mongo_uri
        self.llm_client = llm_client
        self.allowed_tools = allowed_tools or {
            "web_search", "data_fetcher", "text_synthesizer",
            "code_executor", "file_reader", "api_caller",
            "save_financial_metric",
        }
        
        # Initialize components
        self.vector_memory = VectorMemoryStore(mongo_uri)
        self.htn_planner = HTNPlanner(llm_client)
        self.guardrails = SafetyGuardrails()
        self.semantic_loop_detector = SemanticLoopDetector()
        
        # Build LangGraph state machine
        self.graph = self._build_graph()
        
        logger.info("AgenticTaskExecutor initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for task execution."""
        
        workflow = StateGraph(AgenticExecutorState)
        
        # Add nodes for each phase
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("critic", self._critic_node)
        workflow.add_node("recovery", self._recovery_node)
        workflow.add_node("human_input", self._human_input_node)
        
        # Define edges
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "critic")
        
        # Conditional edges from critic
        workflow.add_conditional_edges(
            "critic",
            self._critic_router,
            {
                "success": END,
                "needs_revision": "executor",
                "needs_recovery": "recovery",
                "needs_human_input": "human_input",
            }
        )
        
        workflow.add_edge("recovery", "planner")  # Replan after recovery
        workflow.add_edge("human_input", "executor")  # Continue after human input
        
        return workflow.compile()
    
    async def execute_goal(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a high-level goal autonomously.
        
        Args:
            goal: The goal to achieve (natural language)
            context: Additional context
        
        Returns:
            Execution results and state
        """
        request_id = str(uuid4())
        logger.info(f"Starting execution of goal: {goal} (request_id={request_id})")
        
        # Initialize blackboard state
        blackboard = BlackboardState(request_id, goal)
        blackboard.started_at = datetime.utcnow()
        
        # Decompose goal using HTN
        tasks = self.htn_planner.decompose_goal(goal, context or {})
        for task in tasks:
            blackboard.add_task(task)
        
        # Convert to initial state for LangGraph
        initial_state = {
            'blackboard': blackboard.to_dict(),
            'goal': goal,
            'context': context or {},
            'iteration': 0,
            'max_iterations': 50,  # Prevent infinite execution
            'total_tokens_used': 0,
            'max_token_budget': int((context or {}).get('max_token_budget') or DEFAULT_MAX_TOKEN_BUDGET),
            'budget_depleted': False,
            'execution_history': [],
        }
        
        # Execute via LangGraph
        try:
            final_state = await asyncio.wait_for(
                self.graph.ainvoke(initial_state),
                timeout=300  # 5 minute timeout
            )
            
            blackboard.completed_at = datetime.utcnow()
            blackboard.status = "completed"
            
            logger.info(f"Goal execution completed: {goal}")
            return final_state
            
        except asyncio.TimeoutError:
            logger.error(f"Goal execution timed out: {goal}")
            blackboard.status = "timeout"
            return sanitize_for_mongodb({'error': 'Execution timed out', 'blackboard': blackboard.to_dict()})
        except Exception as e:
            logger.error(f"Goal execution failed: {e}", exc_info=True)
            blackboard.status = "failed"
            return sanitize_for_mongodb({'error': str(e), 'blackboard': blackboard.to_dict()})
    
    async def _planner_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Planner node: Creates or updates the task DAG.
        
        Implements HTN planning and dynamic replanning.
        """
        blackboard_dict = state['blackboard']
        blackboard = self._dict_to_blackboard(blackboard_dict)
        
        blackboard.current_phase = "planning"
        logger.info("Planner node: Analyzing goals and updating task DAG")
        
        # Check if replanning is needed
        if blackboard.failed_tasks:
            logger.info(f"Replanning due to {len(blackboard.failed_tasks)} failed tasks")
            # Add recovery tasks or adjust dependencies
            self._adjust_plan_for_failures(blackboard)
        
        # Select next task to execute
        executable = blackboard.get_executable_tasks()
        if executable:
            next_task = executable[0]
            blackboard.active_task_id = next_task.task_id
            next_task.status = TaskStatus.IN_PROGRESS
            blackboard.update_task(next_task.task_id, status=TaskStatus.IN_PROGRESS)
            logger.info(f"Planner selected task: {next_task.name}")
        else:
            logger.info("No executable tasks found")
        
        return self._node_delta(state, blackboard, "planner")
    
    async def _executor_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executor node: Executes tasks using ReAct loop.
        
        Implements the ReAct (Reasoning and Acting) algorithm.
        """
        blackboard_dict = state['blackboard']
        blackboard = self._dict_to_blackboard(blackboard_dict)
        
        blackboard.current_phase = "executing"
        task_id = blackboard.active_task_id
        
        if not task_id:
            logger.warning("Executor called but no active task")
            return self._node_delta(state, blackboard, "executor:no_active_task")
        
        task = blackboard.get_task(task_id)
        if not task:
            logger.error(f"Active task {task_id} not found")
            return self._node_delta(state, blackboard, "executor:missing_task")
        
        logger.info(f"Executor starting ReAct loop for task: {task.name}")
        
        # ReAct loop
        max_react_steps = 10
        token_updates: Dict[str, Any] = {}
        for step_num in range(1, max_react_steps + 1):
            budget_check = check_token_budget(
                {**state, **token_updates},
                {
                    'task': task.to_dict(),
                    'blackboard_summary': blackboard.get_summary_context(max_steps=5),
                    'scratchpad': blackboard.calculation_scratchpad.compact_summary(),
                },
                reserved_tokens=1_000,
            )
            if not budget_check.allowed:
                logger.warning("Token budget depleted before executor step: %s", budget_check.reason)
                cash_out = self._build_budget_depleted_observation(task, blackboard, budget_check)
                task.status = TaskStatus.BLOCKED
                task.error = cash_out['message']
                task.result = cash_out
                blackboard.failed_tasks.add(task_id)
                blackboard.results[task_id] = cash_out
                token_updates = {
                    **token_updates,
                    'budget_depleted': True,
                    'total_tokens_used': budget_check.total_tokens_used,
                }
                break

            # Check for loops
            if self.guardrails.check_infinite_loop(blackboard.action_history):
                logger.error("Infinite loop detected! Aborting task.")
                task.status = TaskStatus.FAILED
                task.error = "Infinite loop detected"
                blackboard.failed_tasks.add(task_id)
                break
            
            # THOUGHT: Reason about what to do
            thought = self._generate_thought(task, blackboard)
            token_updates = update_token_ledger({**state, **token_updates}, response=thought)
            react_step = ReActStep(
                step_number=step_num,
                action_type=ActionType.THINK,
                thought=thought,
            )
            blackboard.add_react_step(react_step)
            
            # ACTION: Decide on action
            action = self._decide_action(task, blackboard, thought)
            token_updates = update_token_ledger(
                {**state, **token_updates},
                prompt=thought,
                response=action,
            )
            
            if action is None:
                # Task complete or needs human input
                break
            
            # Validate action schema
            is_valid, error = self.guardrails.validate_action_schema(
                action, self.allowed_tools
            )
            if not is_valid:
                logger.error(f"Invalid action schema: {error}")
                task.status = TaskStatus.FAILED
                task.error = f"Schema validation failed: {error}"
                blackboard.failed_tasks.add(task_id)
                break

            loop_result = self.semantic_loop_detector.detect_loop(
                action,
                blackboard.action_history,
            )
            if loop_result.detected:
                logger.warning(
                    "Semantic loop detected for tool %s with similarity %.3f",
                    loop_result.tool,
                    loop_result.max_similarity,
                )
                observation = self._build_loop_recovery_observation(action, loop_result)
                blackboard.add_loop_event(
                    {
                        'task_id': task_id,
                        'tool': loop_result.tool,
                        'similarity': loop_result.max_similarity,
                        'recovery_strategy': observation['recovery_strategy'],
                        'current_action': action,
                        'matched_action': loop_result.matched_action,
                    }
                )
                GLOBAL_AUDIT_LOGGER.log(
                    {
                        'session_id': blackboard.request_id,
                        'user_message': blackboard.goal,
                        'intent': 'semantic_loop_detection',
                        'tool_called': loop_result.tool,
                        'status': 'blocked',
                        'failure_reason': observation['message'],
                    }
                )
                react_step.action = action
                react_step.observation = observation
                blackboard.add_react_step(react_step)
                task.status = TaskStatus.FAILED
                task.error = observation['message']
                blackboard.failed_tasks.add(task_id)
                break
            
            # Record action for loop detection
            blackboard.add_action_to_history(action)
            
            # Execute action (call tool)
            observation = await self._execute_action(action, task, blackboard)
            observation = cap_tool_output(observation)
            token_updates = update_token_ledger(
                {**state, **token_updates},
                response=observation,
            )
            
            react_step.action = action
            react_step.observation = observation
            blackboard.add_react_step(react_step)
            
            # Check if task is complete
            if self._is_task_complete(task, observation):
                task.status = TaskStatus.COMPLETED
                task.result = observation
                blackboard.completed_tasks.add(task_id)
                logger.info(f"Task completed: {task.name}")
                
                # Store in episodic memory
                self._store_episodic_memory(task, observation, success=True)
                break
        
        return self._node_delta(state, blackboard, "executor", **token_updates)
    
    async def _critic_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Critic node: Evaluates execution quality.
        
        Acts as a separate agent reviewing the executor's work.
        """
        blackboard_dict = state['blackboard']
        blackboard = self._dict_to_blackboard(blackboard_dict)
        
        blackboard.current_phase = "reviewing"
        task_id = blackboard.active_task_id

        if state.get('budget_depleted'):
            logger.warning("Critic ending execution because token budget is depleted")
            return self._node_delta(
                state,
                blackboard,
                "critic:budget_depleted",
                critic_decision='success',
                budget_depleted=True,
                total_tokens_used=state.get('total_tokens_used'),
            )
        
        if not task_id:
            return self._node_delta(state, blackboard, "critic:no_active_task")
        
        task = blackboard.get_task(task_id)
        logger.info(f"Critic reviewing task: {task.name}")
        
        # Evaluate quality (in production, use LLM)
        quality_score = self._evaluate_quality(task, blackboard)
        
        if quality_score >= 0.8:
            logger.info(f"Critic approved task (score={quality_score:.2f})")
            state['critic_decision'] = 'success'
        elif quality_score >= 0.5:
            logger.info(f"Critic requested revision (score={quality_score:.2f})")
            state['critic_decision'] = 'needs_revision'
            # Add revision task
            self._add_revision_task(blackboard, task)
        elif task.retry_count < task.max_retries:
            logger.info(f"Critic requested recovery (score={quality_score:.2f})")
            state['critic_decision'] = 'needs_recovery'
            task.retry_count += 1
        else:
            logger.info(f"Critic requesting human input (score={quality_score:.2f})")
            state['critic_decision'] = 'needs_human_input'
            blackboard.pending_human_inputs.append({
                'task_id': task_id,
                'reason': f'Low quality score: {quality_score:.2f}',
                'result': task.result,
            })
        
        return self._node_delta(
            state,
            blackboard,
            "critic",
            critic_decision=state.get('critic_decision'),
        )
    
    async def _recovery_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recovery node: Handles failures and adds recovery tasks.
        """
        blackboard_dict = state['blackboard']
        blackboard = self._dict_to_blackboard(blackboard_dict)
        
        blackboard.current_phase = "recovering"
        logger.info("Recovery node: Attempting to recover from failure")
        
        # Retrieve similar failures from memory
        task_id = blackboard.active_task_id
        task = blackboard.get_task(task_id)
        
        if task:
            similar_failures = self.vector_memory.get_failure_patterns(
                task.description, limit=3
            )
            blackboard.retrieved_memories.extend([
                EpisodicMemory(**m) for m in similar_failures
            ])
            logger.info(f"Retrieved {len(similar_failures)} similar failure patterns")
        
        return self._node_delta(state, blackboard, "recovery")
    
    async def _human_input_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Human-in-the-loop node: Waits for human input.
        """
        blackboard_dict = state['blackboard']
        blackboard = self._dict_to_blackboard(blackboard_dict)
        
        blackboard.current_phase = "awaiting_human_input"
        logger.info("Human input node: Waiting for user guidance")
        
        # In production, this would pause and wait for actual human input
        # For now, we'll simulate continuation
        
        return self._node_delta(state, blackboard, "human_input")
    
    def _critic_router(self, state: Dict[str, Any]) -> str:
        """Route based on critic's decision."""
        decision = state.get('critic_decision', 'success')
        logger.info(f"Critic router: {decision}")
        return decision
    
    # Helper methods

    def _node_delta(
        self,
        state: Dict[str, Any],
        blackboard: BlackboardState,
        node_name: str,
        **extra: Any,
    ) -> AgenticExecutorState:
        delta: AgenticExecutorState = {
            'blackboard': self._blackboard_delta(state.get('blackboard', {}), blackboard),
            'execution_history': [
                {
                    'node': node_name,
                    'phase': blackboard.current_phase,
                    'active_task_id': blackboard.active_task_id,
                    'timestamp': datetime.utcnow().isoformat(),
                }
            ],
        }
        for key, value in extra.items():
            if value is not None:
                delta[key] = value
        return sanitize_for_mongodb(delta)

    def _blackboard_delta(
        self,
        original: Dict[str, Any],
        updated_blackboard: BlackboardState,
    ) -> Dict[str, Any]:
        updated = updated_blackboard.to_dict()
        original = original or {}
        delta: Dict[str, Any] = {
            'request_id': updated.get('request_id'),
            'goal': updated.get('goal'),
            'created_at': updated.get('created_at'),
            'started_at': updated.get('started_at'),
            'completed_at': updated.get('completed_at'),
            'status': updated.get('status'),
            'current_phase': updated.get('current_phase'),
            'active_task_id': updated.get('active_task_id'),
            'tasks': updated.get('tasks', {}),
            'task_order': updated.get('task_order', []),
            'context': updated.get('context', {}),
            'results': updated.get('results', {}),
            'loop_detected': updated.get('loop_detected', False),
        }

        for key in ('completed_tasks', 'failed_tasks', 'skipped_tasks'):
            before = set(original.get(key, []))
            delta[key] = [item for item in updated.get(key, []) if item not in before]

        for key in ('react_history', 'action_history', 'loop_events', 'pending_human_inputs'):
            before_len = len(original.get(key, []))
            delta[key] = updated.get(key, [])[before_len:]

        before_calcs = {
            item.get('calculation_id')
            for item in original.get('calculation_scratchpad', [])
            if isinstance(item, dict)
        }
        delta['calculation_scratchpad'] = [
            item
            for item in updated.get('calculation_scratchpad', [])
            if not isinstance(item, dict) or item.get('calculation_id') not in before_calcs
        ]

        return delta
    
    def _dict_to_blackboard(self, data: Dict[str, Any]) -> BlackboardState:
        """Convert dictionary back to BlackboardState."""
        bb = BlackboardState(data['request_id'], data['goal'])
        bb.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('started_at'):
            bb.started_at = datetime.fromisoformat(data['started_at'])
        if data.get('completed_at'):
            bb.completed_at = datetime.fromisoformat(data['completed_at'])
        bb.status = data['status']
        bb.current_phase = data['current_phase']
        bb.active_task_id = data.get('active_task_id')
        bb.completed_tasks = set(data.get('completed_tasks', []))
        bb.failed_tasks = set(data.get('failed_tasks', []))
        bb.skipped_tasks = set(data.get('skipped_tasks', []))
        bb.context = data.get('context', {})
        bb.results = data.get('results', {})
        bb.loop_detected = bool(data.get('loop_detected', False))
        bb.loop_events = data.get('loop_events', [])
        bb.calculation_scratchpad = CalculationScratchpad.from_list(
            data.get('calculation_scratchpad', [])
        )
        for step_data in data.get('react_history', []):
            if not isinstance(step_data, dict):
                continue
            bb.react_history.append(
                ReActStep(
                    step_number=int(step_data.get('step_number', len(bb.react_history) + 1)),
                    action_type=ActionType(step_data.get('action_type', ActionType.THINK.value)),
                    thought=step_data.get('thought', ''),
                    action=step_data.get('action'),
                    observation=step_data.get('observation'),
                    timestamp=datetime.fromisoformat(step_data['timestamp'])
                    if step_data.get('timestamp') else datetime.utcnow(),
                )
            )
        for item in data.get('action_history', []):
            if isinstance(item, dict) and 'action' in item:
                bb.action_history.append(item)
        
        # Reconstruct tasks
        for task_id, task_data in data.get('tasks', {}).items():
            task = TaskNode(
                task_id=task_data['task_id'],
                name=task_data['name'],
                description=task_data['description'],
                status=TaskStatus(task_data['status']),
                priority=task_data['priority'],
                depends_on=task_data.get('depends_on', []),
                agent_role=AgentRole(task_data.get('agent_role', 'executor')),
                tool_name=task_data.get('tool_name'),
                tool_params=task_data.get('tool_params', {}),
                result=task_data.get('result'),
                error=task_data.get('error'),
                retry_count=task_data.get('retry_count', 0),
                max_retries=task_data.get('max_retries', 3),
                created_at=datetime.fromisoformat(task_data['created_at']),
                updated_at=datetime.fromisoformat(task_data['updated_at']),
                metadata=task_data.get('metadata', {}),
            )
            bb.tasks[task_id] = task
        
        bb.task_order = data.get('task_order', [])
        
        return bb

    def _build_loop_recovery_observation(
        self,
        action: Dict[str, Any],
        loop_result,
    ) -> Dict[str, Any]:
        return {
            'status': 'loop_detected',
            'tool': loop_result.tool,
            'similarity': round(loop_result.max_similarity, 4),
            'recovery_strategy': 'strategy_shift',
            'message': (
                f"Semantic loop detected for {loop_result.tool}. "
                "Stop repeating similar parameters and switch strategy, decompose the task, "
                "or ask the user for clarification."
            ),
            'blocked_action': action,
            'matched_action': loop_result.matched_action,
        }

    def _build_budget_depleted_observation(
        self,
        task: TaskNode,
        blackboard: BlackboardState,
        budget_check,
    ) -> Dict[str, Any]:
        available_results = {
            key: value
            for key, value in blackboard.results.items()
            if key != task.task_id
        }
        return {
            'status': 'budget_depleted',
            'task_id': task.task_id,
            'task_name': task.name,
            'message': (
                "Token budget depleted before another reasoning/tool step. "
                "Cash out using only already collected scratchpad metrics, completed tasks, and current observations."
            ),
            'budget': {
                'total_tokens_used': budget_check.total_tokens_used,
                'estimated_next_context_tokens': budget_check.estimated_context_tokens,
                'max_token_budget': budget_check.max_token_budget,
            },
            'completed_tasks': list(blackboard.completed_tasks),
            'scratchpad': blackboard.calculation_scratchpad.compact_summary(),
            'available_results': available_results,
            'missing': (
                "Further analysis was stopped by the configured token budget. "
                "Ask a narrower follow-up or raise max_token_budget for deeper processing."
            ),
        }
    
    def _generate_thought(self, task: TaskNode, blackboard: BlackboardState) -> str:
        """Generate reasoning thought for current step."""
        scratchpad_context = self._format_scratchpad_for_prompt(blackboard)
        # In production, use LLM
        return (
            f"Analyzing task: {task.name}. Current progress: "
            f"{len(blackboard.completed_tasks)}/{len(blackboard.tasks)} tasks completed.\n\n"
            f"{scratchpad_context}"
        )
    
    def _decide_action(
        self, 
        task: TaskNode, 
        blackboard: BlackboardState,
        thought: str
    ) -> Optional[Dict[str, Any]]:
        """Decide on next action based on thought."""
        # In production, use LLM with function calling
        if task.tool_name:
            return {
                'tool': task.tool_name,
                'parameters': task.tool_params,
            }
        return None
    
    async def _execute_action(
        self, 
        action: Dict[str, Any],
        task: TaskNode,
        blackboard: BlackboardState
    ) -> Any:
        """Execute an action (call a tool)."""
        tool_name = action['tool']
        params = action['parameters']
        
        logger.info(f"Executing tool: {tool_name} with params: {params}")

        if tool_name == "save_financial_metric":
            return self._save_financial_metric(action, blackboard)
        
        # In production, actually call the tool
        # For now, simulate execution
        await asyncio.sleep(0.1)  # Simulate work
        
        return {
            'status': 'success',
            'data': f"Result from {tool_name}",
            'timestamp': datetime.utcnow().isoformat(),
        }

    def _save_financial_metric(
        self,
        action: Dict[str, Any],
        blackboard: BlackboardState,
    ) -> Dict[str, Any]:
        params = action.get('parameters', {})
        metric_name = str(params.get('metric_name') or params.get('label') or "").strip()
        if not metric_name:
            return {
                'status': 'error',
                'error': 'metric_name is required',
                'tool': 'save_financial_metric',
            }

        exact_value = params.get('exact_value', params.get('value'))
        formula = str(params.get('formula') or params.get('context') or "user_verified_metric")
        record = blackboard.record_calculation(
            label=metric_name,
            formula=formula,
            inputs=params.get('inputs') or {},
            result=exact_value,
            source='save_financial_metric',
            metadata={'context': params.get('context')},
        )
        return {
            'status': 'success',
            'tool': 'save_financial_metric',
            'metric_name': metric_name,
            'exact_value': record.get('result'),
            'calculation_id': record.get('calculation_id'),
            'message': f"Saved exact financial metric {metric_name} to scratchpad.",
        }

    def _format_scratchpad_for_prompt(self, blackboard: BlackboardState) -> str:
        records = blackboard.calculation_scratchpad.compact_summary()
        lines = [
            "Scratchpad exact financial metrics:",
            "Before calculating any metric, check the Scratchpad. If the required value exists, use that exact number and do not recalculate it.",
            "Whenever Python, a financial API, or another tool calculates a critical metric, immediately call save_financial_metric with the exact output.",
        ]
        if not records:
            lines.append("- No saved financial metrics yet.")
            return "\n".join(lines)

        for record in records:
            lines.append(
                "- {calculation_id} | {label} = {result} | formula: {formula} | source: {source}".format(
                    calculation_id=record.get('calculation_id'),
                    label=record.get('label'),
                    result=record.get('result'),
                    formula=record.get('formula'),
                    source=record.get('source'),
                )
            )
        return "\n".join(lines)
    
    def _is_task_complete(self, task: TaskNode, observation: Any) -> bool:
        """Check if task is complete based on observation."""
        # In production, use more sophisticated checking
        return observation.get('status') == 'success'
    
    def _evaluate_quality(self, task: TaskNode, blackboard: BlackboardState) -> float:
        """Evaluate the quality of task execution."""
        # In production, use LLM as critic
        if task.result:
            return 0.9  # Assume good quality
        return 0.3
    
    def _store_episodic_memory(
        self, 
        task: TaskNode, 
        result: Any, 
        success: bool
    ) -> None:
        """Store execution in episodic memory for future retrieval."""
        memory = EpisodicMemory(
            task_description=task.description,
            action_taken=task.tool_name or "manual",
            outcome=str(result),
            success=success,
            error_type=task.error if not success else None,
            metadata={
                'task_id': task.task_id,
                'duration_ms': 0,  # Could track actual duration
            }
        )
        
        # In production, compute embedding here
        # memory.embedding = compute_embedding(task.description)
        
        self.vector_memory.store_memory(memory)
    
    def _adjust_plan_for_failures(self, blackboard: BlackboardState) -> None:
        """Dynamically adjust the plan based on failures."""
        # Add recovery tasks or bypass failed dependencies
        for failed_id in blackboard.failed_tasks:
            failed_task = blackboard.get_task(failed_id)
            if failed_task:
                # Find tasks that depend on this failed task
                for task in blackboard.tasks.values():
                    if failed_id in task.depends_on:
                        # Option 1: Skip dependent tasks
                        # task.depends_on.remove(failed_id)
                        
                        # Option 2: Add alternative path (recovery task)
                        recovery_task = TaskNode(
                            task_id=f"recovery_{uuid4().hex[:8]}",
                            name=f"recovery_for_{failed_task.name}",
                            description=f"Alternative approach for {failed_task.description}",
                            priority=task.priority - 1,  # Higher priority
                            depends_on=[],
                            tool_name="recovery_executor",
                            metadata={'recovery_for': failed_id}
                        )
                        blackboard.add_task(recovery_task)
                        
                        # Update dependent task
                        task.depends_on = [recovery_task.task_id]
                        logger.info(f"Added recovery task for {failed_task.name}")
    
    def _add_revision_task(self, blackboard: BlackboardState, original_task: TaskNode) -> None:
        """Add a revision task to the DAG."""
        revision_task = TaskNode(
            task_id=f"revision_{uuid4().hex[:8]}",
            name=f"revise_{original_task.name}",
            description=f"Revise and improve: {original_task.description}",
            priority=original_task.priority,
            depends_on=[],  # Can start immediately
            tool_name="revision_executor",
            metadata={'revises': original_task.task_id}
        )
        
        blackboard.add_task(revision_task)
        blackboard.active_task_id = revision_task.task_id
        logger.info(f"Added revision task: {revision_task.name}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example of using the AgenticTaskExecutor."""
    
    # Initialize executor
    executor = AgenticTaskExecutor(
        mongo_uri="mongodb://localhost:27017",
        # llm_client=your_llm_client,  # Pass your LLM client
    )
    
    # Example 1: Simple research task
    goal1 = "Research competitors and write a summary"
    result1 = await executor.execute_goal(goal1)
    print(f"Result 1: {result1}")
    
    # Example 2: Complex event planning
    goal2 = "Organize a company offsite event"
    result2 = await executor.execute_goal(goal2)
    print(f"Result 2: {result2}")
    
    # Example 3: Analysis task
    goal3 = "Analyze Q4 sales data and generate insights"
    result3 = await executor.execute_goal(goal3)
    print(f"Result 3: {result3}")


if __name__ == "__main__":
    asyncio.run(example_usage())
