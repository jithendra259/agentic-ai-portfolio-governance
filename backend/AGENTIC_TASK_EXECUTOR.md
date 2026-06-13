# Agentic Task Execution System

A production-ready implementation of an autonomous AI agent system for executing complex, multi-step tasks. This system transforms your chatbot from a simple conversational interface into an **autonomous digital employee** capable of planning, executing, and adapting to failures in real-time.

## 🎯 Key Features

### Cognitive Architectures Implemented

1. **ReAct (Reasoning and Acting) Loop**
   - Interleaves thought, action, and observation
   - Enables dynamic decision-making during execution
   - Prevents hallucinations through grounded reasoning

2. **Hierarchical Task Network (HTN) Planning**
   - Decomposes high-level goals into executable subtasks
   - Uses a method library for common patterns
   - Creates dynamic DAGs with dependency management

3. **Tree of Thoughts (ToT) Support**
   - Explores multiple solution branches
   - Evaluates and prunes suboptimal paths
   - Selects highest-probability success strategies

4. **Blackboard Architecture**
   - Centralized shared state for multi-agent coordination
   - Specialized agents (Planner, Executor, Critic, Recovery) read/write information
   - Maintains execution context across long-running tasks

5. **Vector-Based Episodic Memory**
   - MongoDB with HNSW-like vector search
   - Stores past task executions for retrieval-augmented generation
   - Learns from failures via pattern matching

### Safety Guardrails

- **Loop Detection**: Floyd's cycle-finding algorithm prevents infinite loops
- **Schema Validation**: Strict JSON schema prevents tool hallucination
- **Context Management**: Sliding window summarization prevents context overflow
- **Human-in-the-Loop**: Escalates complex decisions to human users

## 📦 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Goal (Natural Language)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  HTN Planner Agent                           │
│  - Decomposes goal into task DAG                            │
│  - Selects methods from library                             │
│  - Validates dependencies                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Blackboard State (Shared Memory)             │
│  - Task graph with dependencies                             │
│  - Execution history                                        │
│  - Context and results                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Planner    │ │   Executor   │ │    Critic    │
│    Node      │ │    Node      │ │    Node      │
│              │ │              │ │              │
│ - Selects    │ │ - ReAct loop │ │ - Quality    │
│   next task  │ │ - Tool calls │ │   review     │
│ - Replans on │ │ - Progress   │ │ - Routing    │
│   failure    │ │   tracking   │ │   decisions  │
└──────────────┘ └──────────────┘ └──────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Vector Memory Store (MongoDB)                   │
│  - Stores episodic memories                                 │
│  - Retrieves similar past executions                        │
│  - Learns from failures                                     │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

The system is already integrated into your backend. Ensure dependencies are installed:

```bash
cd /workspace/backend
pip install -r requirements.txt
```

### Basic Usage

```python
import asyncio
from src.agents.agentic_task_executor import AgenticTaskExecutor

async def main():
    # Initialize executor
    executor = AgenticTaskExecutor(
        mongo_uri="mongodb://localhost:27017",
        # llm_client=your_llm_client,  # Optional: pass LLM for better reasoning
    )
    
    # Execute a complex goal
    goal = "Research competitors and write a summary"
    result = await executor.execute_goal(goal)
    
    # Access results
    blackboard = result.get('blackboard', {})
    print(f"Status: {blackboard.get('status')}")
    print(f"Tasks completed: {len(blackboard.get('completed_tasks', []))}")
    
    # Inspect individual task results
    for task_id, task_data in blackboard.get('tasks', {}).items():
        print(f"\nTask: {task_data['name']}")
        print(f"  Status: {task_data['status']}")
        print(f"  Result: {task_data.get('result')}")

asyncio.run(main())
```

### Example Goals

The system can handle various complex goals:

```python
goals = [
    "Research competitors and write a summary",
    "Organize a company offsite event",
    "Analyze Q4 sales data and generate insights",
    "Create content for marketing campaign",
    "Execute technical analysis on stock portfolio"
]

for goal in goals:
    result = await executor.execute_goal(goal)
```

## 🧠 HTN Method Library

The planner uses a method library to decompose common goal patterns:

| Pattern | Subtasks |
|---------|----------|
| `research_and_summarize` | search_web → evaluate_sources → extract_key_points → synthesize_summary |
| `execute_analysis` | fetch_data → validate_data → compute_metrics → generate_insights |
| `create_content` | outline_structure → draft_content → review_quality → finalize |
| `organize_event` | define_requirements → book_venue → schedule_speakers → send_invitations → coordinate_logistics |
| `research` (flexible) | search_information → analyze_findings → compile_report |
| `organize` (flexible) | plan_requirements → coordinate_resources → execute_plan |

Add custom methods by extending `_build_method_library()` in `HTNPlanner`.

## 🔒 Safety Features

### 1. Infinite Loop Prevention

```python
# Automatically detects repeated actions
if blackboard.detect_loop():
    logger.error("Infinite loop detected! Aborting task.")
    task.status = TaskStatus.FAILED
```

### 2. Tool Hallucination Prevention

```python
# Validates all actions against allowed tools
is_valid, error = SafetyGuardrails.validate_action_schema(
    action, allowed_tools={"web_search", "data_fetcher", ...}
)
if not is_valid:
    raise ValueError(f"Invalid action: {error}")
```

### 3. Context Window Management

```python
# Summarizes old steps to fit within token limits
summary = blackboard.get_summary_context(max_steps=10)
# Keeps last 10 steps detailed, summarizes older ones
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
cd /workspace/backend
python -m pytest test/test_agentic_task_executor.py -v
```

Tests cover:
- Task node creation and serialization
- Blackboard state management
- HTN planning decomposition
- DAG validation (cycle detection)
- Safety guardrails (loop detection, schema validation)
- Vector memory operations
- ReAct loop execution

## 📊 LangGraph Integration

The system uses LangGraph for state machine orchestration:

```python
workflow = StateGraph(dict)

# Add specialized agent nodes
workflow.add_node("planner", self._planner_node)
workflow.add_node("executor", self._executor_node)
workflow.add_node("critic", self._critic_node)
workflow.add_node("recovery", self._recovery_node)

# Define execution flow
workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "critic")

# Conditional routing based on quality review
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
```

## 🗄️ Vector Memory Setup

For optimal performance, configure MongoDB with vector search:

```javascript
// MongoDB shell
db.episodic_memories.createIndex(
  { embedding: "vector" },
  {
    name: "embedding_vector_index",
    vectorOptions: {
      dimensions: 1536,  // OpenAI embedding dimension
      similarity: "cosine",
      type: "vectorSearch"
    }
  }
)
```

The system includes a fallback cosine similarity search if vector index is unavailable.

## 🔧 Customization

### Adding Custom Tools

```python
executor = AgenticTaskExecutor(
    allowed_tools={
        "web_search", "data_fetcher", 
        "my_custom_tool",  # Add your tools
    }
)
```

### Extending the Method Library

```python
class CustomHTNPlanner(HTNPlanner):
    def _build_method_library(self):
        library = super()._build_method_library()
        library["my_custom_pattern"] = [
            {
                "name": "custom_method",
                "subtasks": [
                    {"name": "step1", "tool": "tool_a"},
                    {"name": "step2", "tool": "tool_b"},
                ]
            }
        ]
        return library
```

### Integrating with LLM

For production use, integrate your LLM client:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

executor = AgenticTaskExecutor(
    llm_client=llm,
    mongo_uri="mongodb://localhost:27017"
)
```

## 📈 Performance Considerations

- **Task Parallelism**: Independent tasks can execute in parallel (extend `_executor_node`)
- **Memory Efficiency**: Action history limited to last 100 entries
- **Timeout Protection**: 5-minute default timeout per goal execution
- **Retry Logic**: Configurable max retries per task (default: 3)

## 🎓 Research References

This implementation is based on cutting-edge agentic AI research:

1. **ReAct**: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)
2. **Tree of Thoughts**: Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (NeurIPS 2023)
3. **HTN Planning**: Nau et al., "Shop: Simple Hierarchical Ordered Planner" (IJCAI 2003)
4. **Blackboard Architecture**: Erman et al., "The HEARSAY-II Speech Understanding System" (ACM Computing Surveys 1980)

## 🚧 Future Enhancements

- [ ] LLM-powered dynamic method learning
- [ ] Multi-agent collaboration (parallel executors)
- [ ] Advanced Tree of Thoughts with MCTS
- [ ] Real-time progress streaming
- [ ] Human-in-the-loop API endpoints
- [ ] Embedding model integration for memory

## 📝 License

Part of the Agentic Portfolio Governance System. See main project license.
