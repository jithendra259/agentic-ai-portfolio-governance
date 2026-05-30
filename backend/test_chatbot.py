import os
from langchain_core.messages import HumanMessage
from src.orchestrator.chatbot_orchestrator import portfolio_assistant
import traceback

def run_test():
    messages = [HumanMessage(content="Scatter plot of annualised volatility vs total return for U3 from 2020 to 2025")]
    state = {"messages": messages}
    config = {"configurable": {"thread_id": "test_thread_5"}}
    try:
        for chunk in portfolio_assistant.stream(state, config=config):
            print("======== CHUNK ========")
            # Try to print dict clearly
            if isinstance(chunk, dict):
                for k, v in chunk.items():
                    print(f"NODE: {k}")
                    if "messages" in v:
                        for m in v["messages"]:
                            print(f"  {type(m).__name__}: {m.content}")
                            if hasattr(m, "tool_calls") and m.tool_calls:
                                print(f"  Tool Calls: {m.tool_calls}")
    except Exception as e:
        print(f"CRASH: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
