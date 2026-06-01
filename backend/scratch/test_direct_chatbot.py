import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orchestrator.chatbot_orchestrator import portfolio_assistant
from langchain_core.messages import HumanMessage

async def main():
    import uuid
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("Invoking chatbot...")
    async for event in portfolio_assistant.astream_events(
        {"messages": [HumanMessage(content='Plot AAPL price as a candlestick chart for 2024. Use generate_financial_plot.')]},
        config=config,
        version="v2"
    ):
        event_type = event.get("event", "")
        event_name = event.get("name", "")
        if event_type == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content"):
                print(chunk.content, end="", flush=True)
        elif event_type == "on_tool_start":
            print(f"\n[Tool Start]: {event_name} inputs: {event.get('data', {}).get('input')}")
        elif event_type == "on_tool_end":
            output = event.get('data', {}).get('output')
            print(f"\n[Tool End]: {event_name} output: {str(output)[:200]}")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
