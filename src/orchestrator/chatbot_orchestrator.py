import json
import re
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Import MongoDB-backed historical tools only.
from src.agents.generate_dynamic_plot import generate_financial_plot
from src.agents.live_data_tools import (
    get_stock_database_snapshot,
    get_stocks_by_sector,
    get_stocks_by_universe,
    get_universe_overview,
    list_available_sectors,
    run_full_governance_pipeline,
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
    run_full_governance_pipeline,
]
llm_with_tools = llm.bind_tools(tools)

# 3. Define the System Prompt
SYSTEM_PROMPT = """You are an elite Quantitative Portfolio Governance Agent.
You strictly use historical data (2005-2025) from your local MongoDB.
ABSOLUTE RULE: You are an advisory system. ZERO execution, buying, or selling.
ABSOLUTE RULE: NEVER hallucinate or invent data. If a tool fails, tell the user the tool failed.

RESPONSE MODES:
1. Fast mode is the default.
2. Multimodal mode is only for explicit visualization requests.

STEP 1: Identify the Portfolio & Date
- Check if the user has provided stock tickers and a historical target date (for example, 2008-09-15).
- If either is missing, politely ask the user for the missing information.
- Do NOT proceed to Step 2 without both tickers and a historical target date.
- If the user provides a custom list of stock tickers such as AAPL, MSFT, and NVDA, you MUST bypass predefined universe lookup and pass that exact custom ticker list directly into run_full_governance_pipeline.
- If the user asks for available sectors, use the list_available_sectors tool.
- If the user asks for stocks by sector, use the get_stocks_by_sector tool.
- If the user asks for stocks in a universe such as U1 through U11, use the get_stocks_by_universe tool.
- If the user asks what sector a universe belongs to or asks for a universe summary, use the get_universe_overview tool.
- If the user asks for all stored MongoDB data or a full ticker snapshot, use the get_stock_database_snapshot tool.

STEP 2: Run Governance Pipeline
- Use the run_full_governance_pipeline tool exactly once for the heavy deterministic analysis.
- The tool already performs the historical price lookup, institutional network analysis, and historical G-CVaR optimization back-to-back using local MongoDB data only.
- The tool returns structured JSON. Read it carefully instead of inventing any values.
- In fast mode, stop after the governance pipeline and present the result clearly in text.

STEP 3: Multimodal Mode Only When Explicitly Requested
- Only enter multimodal mode if the user explicitly asks for a chart, plot, graph, heatmap, network view, or visualization.
- If the user does not explicitly ask for visuals, do not request charts.
- When the user does ask for visuals, the system may generate one or two relevant charts after the governance pipeline.

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


def _message_content_to_text(message_or_content) -> str:
    content = getattr(message_or_content, "content", message_or_content)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content) if content is not None else ""


def _latest_user_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_content_to_text(message)
    return ""


def _has_visual_intent(user_text: str) -> bool:
    if not user_text:
        return False

    patterns = [
        r"\bplot\b",
        r"\bchart\b",
        r"\bvisual\b",
        r"\bvisualize\b",
        r"\bheatmap\b",
        r"\bgraph\b",
        r"\bnetwork\b",
        r"\bpie\b",
        r"\bbar\b",
        r"\bline chart\b",
        r"\bshow .*chart\b",
        r"\bshow .*plot\b",
    ]
    lowered = user_text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def _select_plot_recommendations(messages: list[BaseMessage], payload: dict | None) -> list[dict]:
    if not payload or payload.get("status") not in {"success", "partial_success"}:
        return []

    recommendations = payload.get("plot_recommendations", [])
    if not isinstance(recommendations, list) or not recommendations:
        return []

    user_text = _latest_user_text(messages).lower()
    if not _has_visual_intent(user_text):
        return []

    recommendations_by_type = {
        rec.get("plot_type"): rec
        for rec in recommendations
        if isinstance(rec, dict) and rec.get("plot_type")
    }

    selected_types = []
    keyword_map = {
        "heatmap": ["heatmap", "correlation", "covariance"],
        "line": ["line chart", "price chart", "price comparison", "historical comparison", "trend"],
        "bar": ["bar", "risk chart", "risk comparison", "systemic risk"],
        "network": ["network", "bipartite", "holder graph", "institutional graph", "institutional network"],
        "pie": ["pie", "allocation chart", "allocation breakdown", "weight chart", "weights chart"],
    }

    for plot_type, keywords in keyword_map.items():
        if any(keyword in user_text for keyword in keywords) and plot_type in recommendations_by_type:
            selected_types.append(plot_type)

    if not selected_types:
        for fallback_type in ["pie", "bar"]:
            if fallback_type in recommendations_by_type:
                selected_types.append(fallback_type)

    selected = []
    seen = set()
    for plot_type in selected_types:
        if plot_type in seen:
            continue
        seen.add(plot_type)
        selected.append(recommendations_by_type[plot_type])

    return selected[:2]


def _extract_latest_governance_payload(messages: list[BaseMessage]) -> tuple[dict | None, str]:
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "run_full_governance_pipeline":
            raw = _message_content_to_text(message)
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed, raw
            except Exception:
                return None, raw
    return None, ""


def _collect_recent_plot_outputs(messages: list[BaseMessage]) -> list[str]:
    plot_outputs = []
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "generate_financial_plot":
            plot_outputs.append(_message_content_to_text(message))
            continue
        if plot_outputs:
            break
    plot_outputs.reverse()
    return plot_outputs


def _build_governance_markdown(payload: dict | None, raw_text: str) -> str:
    if not payload:
        return raw_text or "Unable to generate a response for this request."

    lines = [
        "## Historical Governance Report",
        f"- Status: {payload.get('status', 'unknown')}",
        f"- Target date: {payload.get('target_date', 'unknown')}",
        f"- Requested tickers: {', '.join(payload.get('requested_tickers', [])) or 'None'}",
        f"- Valid tickers used: {', '.join(payload.get('valid_tickers', [])) or 'None'}",
    ]

    dropped = payload.get("dropped_tickers", [])
    if dropped:
        lines.append("- Dropped tickers:")
        for item in dropped:
            lines.append(f"  - {item.get('ticker', 'UNKNOWN')}: {item.get('reason', 'unspecified reason')}")

    price_snapshot = payload.get("price_snapshot", [])
    if price_snapshot:
        lines.append("- Historical prices used:")
        for item in price_snapshot:
            lines.append(
                f"  - {item.get('ticker', 'UNKNOWN')}: {item.get('close', 'N/A')} on {item.get('effective_date', 'N/A')}"
            )

    systemic_risk = payload.get("systemic_risk", {})
    scores = systemic_risk.get("scores", {}) if isinstance(systemic_risk, dict) else {}
    if scores:
        lines.append("- Structural risk scores:")
        for ticker, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"  - {ticker}: {score:.4f}")

    optimization = payload.get("optimization", {})
    if isinstance(optimization, dict):
        optimization_type = optimization.get("optimization_type")
        if optimization_type:
            lines.append(f"- Optimizer: {optimization_type}")

        instability_index = optimization.get("instability_index")
        lambda_t = optimization.get("lambda_t")
        if instability_index is not None:
            lines.append(f"- Instability index (I_t): {instability_index:.4f}")
        if lambda_t is not None:
            lines.append(f"- Graph penalty (lambda_t): {lambda_t:.4f}")

        weights = optimization.get("weights", {})
        if weights:
            lines.append("- Recommended allocation weights:")
            for ticker, weight in weights.items():
                lines.append(f"  - {ticker}: {weight:.2%}")

        expected_return = optimization.get("expected_annualized_return")
        expected_cvar = optimization.get("expected_cvar_95")
        if expected_return is not None:
            lines.append(f"- Expected annualized return: {expected_return:.2%}")
        if expected_cvar is not None:
            lines.append(f"- Expected 95% CVaR: {expected_cvar:.2%}")

    message = payload.get("message")
    if message:
        lines.extend(["", message])

    return "\n".join(lines)


def tool_result_router(state: AgentState):
    """Default to fast text mode, only branching to visuals on explicit user request."""
    messages = state["messages"]
    if not messages:
        return "chatbot"

    last_message = messages[-1]
    if isinstance(last_message, ToolMessage) and last_message.name == "run_full_governance_pipeline":
        payload, _ = _extract_latest_governance_payload(messages)
        if _select_plot_recommendations(messages, payload):
            return "generate_visuals"
        return "finalize_governance"

    if isinstance(last_message, ToolMessage) and last_message.name == "generate_financial_plot":
        return "finalize_governance"

    return "chatbot"


def generate_visuals_node(state: AgentState):
    """Generate a small set of requested visuals without another LLM turn."""
    messages = state["messages"]
    payload, _ = _extract_latest_governance_payload(messages)
    plot_requests = _select_plot_recommendations(messages, payload)

    tool_messages = []
    for index, request in enumerate(plot_requests, start=1):
        plot_output = generate_financial_plot.invoke(
            {
                "data": request.get("data", {}),
                "plot_type": request.get("plot_type", ""),
                "title": request.get("title", "Financial Plot"),
            }
        )
        tool_messages.append(
            ToolMessage(
                content=plot_output,
                tool_call_id=f"deterministic-plot-{index}",
                name="generate_financial_plot",
            )
        )

    return {"messages": tool_messages}


def finalize_governance_node(state: AgentState):
    """Combine governance JSON and generated plot markdown into a final assistant reply."""
    messages = state["messages"]
    if not messages:
        return {"messages": [AIMessage(content="Unable to generate a response for this request.")]}

    governance_payload, raw_text = _extract_latest_governance_payload(messages)
    plot_outputs = _collect_recent_plot_outputs(messages)
    content_parts = [_build_governance_markdown(governance_payload, raw_text)]

    if plot_outputs:
        content_parts.append("## Generated Visuals")
        content_parts.extend(plot_outputs)

    return {"messages": [AIMessage(content="\n\n".join(part for part in content_parts if part))]}

# 5. Build the LangGraph State Machine
builder = StateGraph(AgentState)

# Add the nodes
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools)) # This node automatically runs the Python tools
builder.add_node("generate_visuals", generate_visuals_node)
builder.add_node("finalize_governance", finalize_governance_node)

# Define the routing logic
builder.set_entry_point("chatbot")

# If the LLM decides it needs a MongoDB-backed historical tool, route to 'tools'
# Otherwise, route to END to output the chat response to the user
builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Fast-path the heavy governance tool so we do not pay for a second LLM pass.
builder.add_conditional_edges(
    "tools",
    tool_result_router,
    {
        "generate_visuals": "generate_visuals",
        "finalize_governance": "finalize_governance",
        "chatbot": "chatbot",
    },
)
builder.add_edge("generate_visuals", "finalize_governance")
builder.add_edge("finalize_governance", END)

# 6. Add Conversational Memory (Crucial for the Thesis!)
memory = MemorySaver()

# Compile the final conversational agent
portfolio_assistant = builder.compile(checkpointer=memory)

print("Conversational Agentic Supervisor Initialized with Memory!")
