from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Import MongoDB-backed historical tools only.
from agents.live_data_tools import (
    analyze_institutional_network,
    get_historical_prices,
    get_stock_database_snapshot,
    get_stocks_by_sector,
    get_stocks_by_universe,
    get_universe_overview,
    list_available_sectors,
    run_historical_cvar_optimization,
)

# Define the State: This is the Chatbot's Memory!
class AgentState(TypedDict):
    # 'add_messages' ensures new chat messages are appended, not overwritten
    messages: Annotated[list[BaseMessage], add_messages]
    user_portfolio: list[str]
    risk_profile: str

# 1. Initialize the LLM (Your Local Qwen 2.5)
llm = ChatOllama(model="qwen2.5:7b", temperature=0.2)

# 2. Bind the Tools to the LLM
# Historical database lookup + advisory optimization only. No execution tools are exposed.
tools = [
    list_available_sectors,
    get_stocks_by_sector,
    get_stocks_by_universe,
    get_universe_overview,
    get_stock_database_snapshot,
    get_historical_prices,
    analyze_institutional_network,
    run_historical_cvar_optimization,
]
llm_with_tools = llm.bind_tools(tools)

# 3. Define the System Prompt
SYSTEM_PROMPT = """You are an elite Quantitative Portfolio Governance Agent.
You strictly use historical data (2005-2025) from your local MongoDB.
ABSOLUTE RULE: You are an advisory system. ZERO execution, buying, or selling.
ABSOLUTE RULE: NEVER hallucinate or invent data. If a tool fails, tell the user the tool failed.

YOUR STRUCTURED PIPELINE:
When a user asks for a portfolio analysis or optimization, you MUST follow this exact sequence:

STEP 1: Identify the Portfolio & Date
- Check if the user has provided stock tickers and a historical target date (for example, 2008-09-15).
- If either is missing, politely ask the user for the missing information.
- Do NOT proceed to Step 2 without both tickers and a historical target date.
- If the user asks for available sectors, use the list_available_sectors tool.
- If the user asks for stocks by sector, use the get_stocks_by_sector tool.
- If the user asks for stocks in a universe such as U1 through U11, use the get_stocks_by_universe tool.
- If the user asks what sector a universe belongs to or asks for a universe summary, use the get_universe_overview tool.
- If the user asks for all stored MongoDB data or a full ticker snapshot, use the get_stock_database_snapshot tool.

STEP 2: Fetch Historical Data (Agent 1)
- Use the get_historical_prices tool to query MongoDB for the tickers on the target date.
- Wait for the tool to return the actual database prices.

STEP 3: Analyze Systemic Risk (Agent 2)
- Use the analyze_institutional_network tool to calculate the bipartite graph centrality risk scores for those tickers.

STEP 4: G-CVaR Optimization (Agent 3)
- Use the run_historical_cvar_optimization tool to calculate the mathematically safest allocation weights for that historical point in time.

STEP 5: Human-in-the-Loop Governance Report (Agent 4)
- Present the final results to the user in a professional, structured report.
- Clearly state the historical date, the prices found, the network risk scores, and the final recommended G-CVaR allocation percentages.

GENERAL RULES:
1. Use only MongoDB-backed historical tools. No live internet data and no yfinance.
2. Never execute trades. This system is read-only and advisory only.
3. If a tool fails, say so clearly and do not invent missing values.
4. Always explain the allocation recommendation mathematically and transparently.
5. Never call get_stocks_by_sector with an empty sector. Use list_available_sectors for sector discovery.
6. If the user asks for comprehensive stored ticker information, prefer get_stock_database_snapshot before summarizing.
7. If the user asks about universe membership or requests a universe roster, use get_stocks_by_universe or get_stock_database_snapshot as appropriate.
8. If the user asks about a universe's sector identity or composition, use get_universe_overview.
"""

# 4. Define the Nodes
def chatbot_node(state: AgentState):
    """The main LLM brain that reads the chat and decides what to do."""
    messages = state["messages"]
    
    # Ensure the system prompt is always guiding the behavior
    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 5. Build the LangGraph State Machine
builder = StateGraph(AgentState)

# Add the nodes
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools)) # This node automatically runs the Python tools

# Define the routing logic
builder.set_entry_point("chatbot")

# If the LLM decides it needs a MongoDB-backed historical tool, route to 'tools'
# Otherwise, route to END to output the chat response to the user
builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# After the tool runs, always go back to the chatbot so it can read the result and reply
builder.add_edge("tools", "chatbot")

# 6. Add Conversational Memory (Crucial for the Thesis!)
memory = MemorySaver()

# Compile the final conversational agent
portfolio_assistant = builder.compile(checkpointer=memory)

print("Conversational Agentic Supervisor Initialized with Memory!")
