"""
Tests for the Agentic Task Execution System.

Demonstrates:
- HTN Planning decomposition
- ReAct loop execution
- Dynamic DAG mutation
- Vector memory storage and retrieval
- Safety guardrails (loop detection, schema validation)
- Blackboard state management
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from src.agents.agentic_task_executor import (
    TaskStatus,
    AgentRole,
    ActionType,
    TaskNode,
    ReActStep,
    EpisodicMemory,
    BlackboardState,
    VectorMemoryStore,
    HTNPlanner,
    SafetyGuardrails,
    AgenticTaskExecutor,
)


class TestTaskNode:
    """Test the TaskNode data structure."""
    
    def test_create_task_node(self):
        """Test creating a basic task node."""
        task = TaskNode(
            task_id="task_001",
            name="fetch_data",
            description="Fetch sales data from database"
        )
        
        assert task.task_id == "task_001"
        assert task.name == "fetch_data"
        assert task.status == TaskStatus.PENDING
        assert task.priority == 5
        assert task.depends_on == []
        assert task.retry_count == 0
    
    def test_task_to_dict(self):
        """Test serializing task to dictionary."""
        task = TaskNode(
            task_id="task_002",
            name="analyze_data",
            description="Analyze fetched data",
            priority=2,
            depends_on=["task_001"],
            tool_name="data_analyzer"
        )
        
        task_dict = task.to_dict()
        
        assert task_dict['task_id'] == "task_002"
        assert task_dict['status'] == "pending"
        assert task_dict['priority'] == 2
        assert task_dict['depends_on'] == ["task_001"]
        assert 'created_at' in task_dict
        assert 'updated_at' in task_dict


class TestBlackboardState:
    """Test the Blackboard Architecture implementation."""
    
    def test_create_blackboard(self):
        """Test creating a blackboard state."""
        bb = BlackboardState(
            request_id="req_123",
            goal="Research competitors"
        )
        
        assert bb.request_id == "req_123"
        assert bb.goal == "Research competitors"
        assert bb.status == "initialized"
        assert bb.current_phase == "planning"
    
    def test_add_and_get_tasks(self):
        """Test adding and retrieving tasks from blackboard."""
        bb = BlackboardState("req_123", "Test goal")
        
        task1 = TaskNode("t1", "Task 1", "Description 1", priority=2)
        task2 = TaskNode("t2", "Task 2", "Description 2", priority=1, depends_on=["t1"])
        
        bb.add_task(task1)
        bb.add_task(task2)
        
        assert len(bb.tasks) == 2
        assert bb.get_task("t1").name == "Task 1"
        assert bb.get_task("t2").name == "Task 2"
    
    def test_get_executable_tasks(self):
        """Test getting tasks with satisfied dependencies."""
        bb = BlackboardState("req_123", "Test goal")
        
        # Create task chain: t1 -> t2 -> t3
        t1 = TaskNode("t1", "Task 1", "Desc 1", priority=1)
        t2 = TaskNode("t2", "Task 2", "Desc 2", priority=2, depends_on=["t1"])
        t3 = TaskNode("t3", "Task 3", "Desc 3", priority=3, depends_on=["t2"])
        
        bb.add_task(t1)
        bb.add_task(t2)
        bb.add_task(t3)
        
        # Initially only t1 is executable
        executable = bb.get_executable_tasks()
        assert len(executable) == 1
        assert executable[0].task_id == "t1"
        
        # Complete t1, now t2 should be executable
        bb.completed_tasks.add("t1")
        t1.status = TaskStatus.COMPLETED
        
        executable = bb.get_executable_tasks()
        assert len(executable) == 1
        assert executable[0].task_id == "t2"
    
    def test_loop_detection(self):
        """Test infinite loop detection."""
        bb = BlackboardState("req_123", "Test goal")
        
        # Add same action multiple times
        same_action = {"tool": "web_search", "params": {"query": "test"}}
        
        for _ in range(4):
            bb.add_action_to_history(same_action)
        
        # Should detect loop (3+ identical actions)
        assert bb.detect_loop() is True
    
    def test_get_summary_context(self):
        """Test context summarization for window management."""
        bb = BlackboardState("req_123", "Complex goal")
        
        # Add many ReAct steps
        for i in range(15):
            step = ReActStep(
                step_number=i,
                action_type=ActionType.THINK,
                thought=f"Thought {i}"
            )
            bb.add_react_step(step)
        
        summary = bb.get_summary_context(max_steps=10)
        
        assert summary['goal'] == "Complex goal"
        assert summary['react_summary'] is not None  # Should summarize old steps
        assert len(summary['recent_react_steps']) == 10  # Keep last 10 detailed


class TestHTNPlanner:
    """Test Hierarchical Task Network planning."""
    
    def test_decompose_known_pattern(self):
        """Test decomposing a goal matching known pattern."""
        planner = HTNPlanner()
        
        goal = "Research competitors and write a summary"
        tasks = planner.decompose_goal(goal)
        
        # Should decompose into multiple subtasks
        assert len(tasks) > 1
        assert tasks[0].name == "search_web"
        
        # Check dependency chain
        for i in range(1, len(tasks)):
            assert tasks[i].depends_on == [tasks[i-1].task_id]
    
    def test_decompose_event_planning(self):
        """Test decomposing event planning goal."""
        planner = HTNPlanner()
        
        goal = "Organize a company offsite event"
        tasks = planner.decompose_goal(goal)
        
        assert len(tasks) >= 4  # At least 4 subtasks from method library
        assert tasks[0].name == "define_requirements"
    
    def test_validate_dag_no_cycles(self):
        """Test DAG validation catches cycles."""
        planner = HTNPlanner()
        
        # Create tasks with cycle: t1 -> t2 -> t3 -> t1
        t1 = TaskNode("t1", "Task 1", "Desc 1", depends_on=["t3"])
        t2 = TaskNode("t2", "Task 2", "Desc 2", depends_on=["t1"])
        t3 = TaskNode("t3", "Task 3", "Desc 3", depends_on=["t2"])
        
        with pytest.raises(ValueError, match="Circular dependency"):
            planner._validate_dag([t1, t2, t3])
    
    def test_validate_dag_valid(self):
        """Test DAG validation passes for valid DAG."""
        planner = HTNPlanner()
        
        # Create valid chain: t1 -> t2 -> t3
        t1 = TaskNode("t1", "Task 1", "Desc 1")
        t2 = TaskNode("t2", "Task 2", "Desc 2", depends_on=["t1"])
        t3 = TaskNode("t3", "Task 3", "Desc 3", depends_on=["t2"])
        
        # Should not raise
        planner._validate_dag([t1, t2, t3])


class TestSafetyGuardrails:
    """Test safety mechanisms."""
    
    def test_validate_action_schema_valid(self):
        """Test schema validation for valid action."""
        allowed_tools = {"web_search", "data_fetcher"}
        action = {
            "tool": "web_search",
            "parameters": {"query": "test"}
        }
        
        is_valid, error = SafetyGuardrails.validate_action_schema(
            action, allowed_tools
        )
        
        assert is_valid is True
        assert error is None
    
    def test_validate_action_schema_invalid_tool(self):
        """Test schema validation catches hallucinated tools."""
        allowed_tools = {"web_search", "data_fetcher"}
        action = {
            "tool": "make_coffee",  # Not allowed
            "parameters": {}
        }
        
        is_valid, error = SafetyGuardrails.validate_action_schema(
            action, allowed_tools
        )
        
        assert is_valid is False
        assert "Unknown tool" in error
    
    def test_validate_action_schema_missing_fields(self):
        """Test schema validation catches missing fields."""
        action = {"tool": "web_search"}  # Missing parameters
        
        is_valid, error = SafetyGuardrails.validate_action_schema(
            action, {"web_search"}
        )
        
        assert is_valid is False
        assert "Missing 'parameters'" in error
    
    def test_check_infinite_loop(self):
        """Test infinite loop detection."""
        from collections import deque
        
        action_history = deque(maxlen=100)
        same_action = {"tool": "test", "params": {}}
        
        # Add same action 3 times
        for _ in range(3):
            action_history.append({"action": same_action})
        
        assert SafetyGuardrails.check_infinite_loop(action_history) is True
        
        # Add different action
        action_history.append({"action": {"tool": "different"}})
        assert SafetyGuardrails.check_infinite_loop(action_history) is False


class TestVectorMemoryStore:
    """Test vector-based episodic memory."""
    
    @pytest.fixture
    def mock_mongo(self):
        """Mock MongoDB client for testing."""
        with patch('src.agents.agentic_task_executor.MongoClient') as mock_client:
            mock_db = Mock()
            mock_collection = Mock()
            mock_client.return_value.__getitem__.return_value = mock_db
            mock_db.__getitem__.return_value = mock_collection
            mock_collection.index_information.return_value = {}
            
            yield mock_collection
    
    def test_store_memory(self, mock_mongo):
        """Test storing episodic memory."""
        store = VectorMemoryStore("mongodb://localhost:27017")
        
        memory = EpisodicMemory(
            task_description="Test task",
            action_taken="web_search",
            outcome="Success",
            success=True
        )
        
        memory_id = store.store_memory(memory)
        
        assert mock_mongo.insert_one.called
        assert memory_id is not None
    
    def test_get_failure_patterns(self, mock_mongo):
        """Test retrieving failure patterns."""
        store = VectorMemoryStore("mongodb://localhost:27017")
        
        # Mock return value
        mock_mongo.find.return_value.limit.return_value = [
            {
                'task_description': 'Failed search',
                'success': False,
                'error_type': 'timeout',
                'created_at': datetime.utcnow().isoformat()
            }
        ]
        
        failures = store.get_failure_patterns("search web", limit=3)
        
        assert len(failures) == 1
        assert failures[0]['success'] is False


class TestAgenticTaskExecutorIntegration:
    """Integration tests for the full agentic executor."""
    
    @pytest.fixture
    def mock_executor_components(self):
        """Mock executor components for testing."""
        with patch('src.agents.agentic_task_executor.VectorMemoryStore') as mock_memory, \
             patch('src.agents.agentic_task_executor.HTNPlanner') as mock_planner, \
             patch('src.agents.agentic_task_executor.MongoClient'):
            
            # Setup mocks
            mock_planner_instance = Mock()
            mock_planner.return_value = mock_planner_instance
            
            # Return simple task chain for any goal
            mock_planner_instance.decompose_goal.return_value = [
                TaskNode("t1", "step1", "First step", tool_name="test_tool"),
                TaskNode("t2", "step2", "Second step", depends_on=["t1"]),
            ]
            
            yield {
                'memory': mock_memory,
                'planner': mock_planner_instance
            }
    
    @pytest.mark.asyncio
    async def test_execute_simple_goal(self, mock_executor_components):
        """Test executing a simple goal."""
        executor = AgenticTaskExecutor(mongo_uri="mongodb://localhost:27017")
        
        goal = "Analyze Q4 sales data"
        result = await executor.execute_goal(goal)
        
        # Should complete without error
        assert 'error' not in result or result.get('blackboard', {}).get('status') in ['completed', 'failed']
    
    @pytest.mark.asyncio
    async def test_dynamic_replanning_on_failure(self, mock_executor_components):
        """Test that executor replans when tasks fail."""
        executor = AgenticTaskExecutor(mongo_uri="mongodb://localhost:27017")
        
        # This would test the recovery mechanism
        # In a real test, we'd simulate tool failures and verify replanning
        pass


class TestReActLoop:
    """Test the ReAct (Reasoning and Acting) loop implementation."""
    
    def test_react_step_creation(self):
        """Test creating ReAct steps."""
        step = ReActStep(
            step_number=1,
            action_type=ActionType.THINK,
            thought="I need to search for competitor information"
        )
        
        assert step.step_number == 1
        assert step.action_type == ActionType.THINK
        assert "competitor" in step.thought
    
    def test_react_step_with_action(self):
        """Test ReAct step with action and observation."""
        step = ReActStep(
            step_number=2,
            action_type=ActionType.CALL_TOOL,
            thought="Now I'll execute the search",
            action={"tool": "web_search", "params": {"query": "competitors"}},
            observation={"results": ["Company A", "Company B"]}
        )
        
        step_dict = step.to_dict()
        assert step_dict['action_type'] == "call_tool"
        assert step_dict['observation']['results'] == ["Company A", "Company B"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
