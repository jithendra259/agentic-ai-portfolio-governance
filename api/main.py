import logging
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.orchestrator.llm_router import portfolio_assistant


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic Portfolio Governance API",
    description="Advisory-only backend for historical portfolio governance using local MongoDB data.",
    version="1.0.0",
)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


class ChatRequest(BaseModel):
    session_id: str
    user_message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str


def _message_to_text(message: Any) -> str:
    if message is None:
        return ""

    content = getattr(message, "content", message)
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

    return str(content) if content else ""


def _chunk_to_text(chunk: Any) -> str:
    if chunk is None:
        return ""

    content = getattr(chunk, "content", chunk)
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

    return str(content) if content else ""


def _stream_event(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload) + "\n").encode("utf-8")


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "mode": "advisory-only",
        "data_source": "local-mongodb-historical-only",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty")

    if not request.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message cannot be empty")

    try:
        logger.info("Processing chat request for session_id=%s", request.session_id)

        result = portfolio_assistant.invoke(
            {"messages": [HumanMessage(content=request.user_message)]},
            config={"configurable": {"thread_id": request.session_id}},
        )

        messages = result.get("messages", [])
        if not messages:
            response_text = "Unable to generate a response for this request."
        else:
            last_message = messages[-1]
            response_text = getattr(last_message, "content", "") or str(last_message)

        return ChatResponse(session_id=request.session_id, response=response_text)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Backend advisory request failed")
        raise HTTPException(
            status_code=500,
            detail=f"Backend error while processing advisory request: {exc}",
        ) from exc


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty")

    if not request.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message cannot be empty")

    async def event_generator():
        accumulated_response = ""
        saw_tokens = False

        try:
            logger.info("Streaming chat request for session_id=%s", request.session_id)
            yield _stream_event({"type": "status", "content": "Analyzing your request..."})

            async for event in portfolio_assistant.astream_events(
                {"messages": [HumanMessage(content=request.user_message)]},
                config={"configurable": {"thread_id": request.session_id}},
                version="v2",
            ):
                event_type = event.get("event", "")
                event_name = event.get("name", "")

                if event_type == "on_chat_model_stream":
                    text = _chunk_to_text(event.get("data", {}).get("chunk"))
                    if text:
                        saw_tokens = True
                        accumulated_response += text
                        yield _stream_event({"type": "token", "content": text})

                elif event_type == "on_tool_start":
                    yield _stream_event({"type": "status", "content": f"Running tool: {event_name}..."})

                elif event_type == "on_tool_end":
                    yield _stream_event({"type": "status", "content": f"Finished tool: {event_name}."})

                elif event_type == "on_chain_end" and event_name == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    messages = output.get("messages", []) if isinstance(output, dict) else []
                    if messages and not saw_tokens:
                        final_text = _message_to_text(messages[-1])
                        if final_text:
                            accumulated_response = final_text
                            yield _stream_event({"type": "final", "content": final_text})

            if saw_tokens:
                yield _stream_event({"type": "final", "content": accumulated_response})
            elif not accumulated_response:
                yield _stream_event({"type": "final", "content": "Unable to generate a response for this request."})

        except Exception as exc:
            logger.exception("Backend streaming advisory request failed")
            yield _stream_event(
                {
                    "type": "error",
                    "content": f"Backend error while processing advisory request: {exc}",
                }
            )

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
