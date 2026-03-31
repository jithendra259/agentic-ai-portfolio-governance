import uuid
from typing import Any

import gradio as gr
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from orchestrator.chatbot_orchestrator import portfolio_assistant


session_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": session_id}}


def _chunk_to_text(chunk: Any) -> str:
    """Best-effort conversion of LangChain/LangGraph streaming chunks into plain text."""
    if chunk is None:
        return ""

    if isinstance(chunk, str):
        return chunk

    if isinstance(chunk, AIMessageChunk):
        content = chunk.content
    elif isinstance(chunk, AIMessage):
        content = chunk.content
    else:
        content = getattr(chunk, "content", chunk)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                text = getattr(item, "text", "")
                if text:
                    parts.append(text)
        return "".join(parts)

    return str(content) if content else ""


def _message_to_text(message: Any) -> str:
    """Extract text from a final LangChain message object."""
    if message is None:
        return ""

    if isinstance(message, str):
        return message

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)

    return str(content) if content else ""


async def chat_with_advisor(user_message: str, history):
    """
    Stream the LangGraph portfolio assistant response incrementally to Gradio.
    This UI surfaces advisory analysis only; it does not perform any trade execution.
    """
    accumulated_response = ""
    saw_stream_tokens = False

    try:
        inputs = {"messages": [HumanMessage(content=user_message)]}

        async for event in portfolio_assistant.astream_events(
            inputs,
            config=config,
            version="v2",
        ):
            event_type = event.get("event", "")

            if event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                text = _chunk_to_text(chunk)
                if text:
                    saw_stream_tokens = True
                    accumulated_response += text
                    yield accumulated_response

            elif event_type == "on_chain_end" and event.get("name") == "LangGraph":
                if saw_stream_tokens:
                    continue

                output = event.get("data", {}).get("output", {})
                messages = output.get("messages", []) if isinstance(output, dict) else []
                if messages:
                    final_text = _message_to_text(messages[-1])
                    if final_text:
                        accumulated_response = final_text
                        yield accumulated_response

        if not accumulated_response:
            yield "Unable to generate a response for this request."

    except Exception as exc:
        yield f"System Error: Unable to process request. Details: {exc}"


with gr.Blocks() as demo:
    gr.Markdown("# Agentic Portfolio Management Assistant")
    gr.Markdown(
        "Welcome to your personal quantitative advisor. You can ask for live market data, "
        "portfolio risk assessments, and CVaR-based optimization guidance.\n\n"
        "*Governance Note: This system provides mathematical allocation advice only and does not execute trades.*"
    )

    gr.ChatInterface(
        fn=chat_with_advisor,
        textbox=gr.Textbox(
            placeholder="Ask about your portfolio, risk, or allocation optimization...",
            container=False,
            scale=7,
        ),
        submit_btn="Send Request",
        stop_btn="Stop",
    )


if __name__ == "__main__":
    print("Launching Conversational Agentic UI...")
    demo.launch(share=False, theme=gr.themes.Soft())
