import json
import logging
import shutil
import socket
import subprocess
import sys
from contextlib import asynccontextmanager

print("=== MAIN.PY LOADED ===")
print("PATH:", sys.executable)

from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def _port_owner(port: int) -> str:
    command = (
        "$conn = Get-NetTCPConnection -LocalPort "
        f"{port} "
        "-ErrorAction SilentlyContinue | "
        "Where-Object { $_.State -eq 'Listen' } | Select-Object -First 1; "
        "if ($conn) { "
        "$proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue; "
        "if ($proc) { Write-Output \"$($proc.ProcessName) (PID $($proc.Id))\" } "
        "else { Write-Output \"PID $($conn.OwningProcess)\" } "
        "}"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
    except Exception:
        return ""

    return result.stdout.strip()


def _print_port_conflict(host: str, port: int) -> None:
    owner = _port_owner(port)
    owner_line = f"\nPort owner: {owner}" if owner else ""
    print(
        f"Backend API is already running or port is occupied at http://{host}:{port}.\n"
        "Open http://127.0.0.1:8000/health to verify it, or stop the existing Python/uvicorn process before starting another."
        f"{owner_line}"
    )


if __name__ == "__main__" and _is_port_open("127.0.0.1", 8000):
    _print_port_conflict("127.0.0.1", 8000)
    sys.exit(1)


from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from src.memory.mongodb_memory_layer import MongoMemoryManager
from src.orchestrator.chatbot_orchestrator import (
    FALLBACK_OLLAMA_MODEL,
    INSTALLED_OLLAMA_MODELS,
    PRIMARY_OLLAMA_MODEL,
    portfolio_assistant,
)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LEGACY_OUTPUTS_DIR = PROJECT_ROOT / "src" / "outputs"


def _sync_legacy_outputs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if not LEGACY_OUTPUTS_DIR.exists():
        return

    for legacy_file in LEGACY_OUTPUTS_DIR.glob("*.png"):
        target = OUTPUTS_DIR / legacy_file.name
        if not target.exists():
            shutil.copy2(legacy_file, target)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing MongoDB indexes to fix search latency...")
    _sync_legacy_outputs()
    try:
        memory = MongoMemoryManager()
        memory.setup_indexes()
        logger.info("MongoDB indexes are active.")
        app.state.mongo_available = True
    except Exception as exc:
        logger.error("Failed to build MongoDB indexes: %s", exc)
        app.state.mongo_available = False
    yield


app = FastAPI(
    title="Agentic Portfolio Governance API",
    description="Advisory-only backend for historical portfolio governance using local MongoDB data.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Support for robust local development
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


class ChatRequest(BaseModel):
    session_id: str
    user_message: str
    model: str | None = None


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
    mongo_status = getattr(app.state, "mongo_available", False)
    
    # Basic check for Ollama model presence
    ollama_status = PRIMARY_OLLAMA_MODEL in INSTALLED_OLLAMA_MODELS
    
    return {
        "status": "ok" if mongo_status and ollama_status else "degraded",
        "mode": "advisory-only",
        "data_source": "local-mongodb-historical-only",
        "components": {
            "mongodb": "connected" if mongo_status else "disconnected",
            "ollama": "ready" if ollama_status else "model_missing",
        },
        "models": {
            "primary": PRIMARY_OLLAMA_MODEL,
            "fallback": FALLBACK_OLLAMA_MODEL,
            "available": INSTALLED_OLLAMA_MODELS
        }
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
            config={"configurable": {"thread_id": request.session_id, "override_model": request.model}},
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


@app.get("/plot_data/{session_id}")
def get_plot_data(session_id: str):
    if session_id in GLOBAL_PLOT_DATA:
        return GLOBAL_PLOT_DATA[session_id]
    return {}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty")

    if not request.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message cannot be empty")

    async def event_generator():
        import uuid
        msg_id = str(uuid.uuid4())
        text_id = f"{msg_id}-text"
        
        yield _stream_event({"type": "start", "messageId": msg_id})
        yield _stream_event({"type": "text-start", "id": text_id})

        accumulated_response = ""
        saw_tokens = False
        suppressed_stream_tokens = False
        final_sent = False
        tool_runs = set()
        tool_names_run = set()

        try:
            logger.info("Streaming chat request for session_id=%s", request.session_id)

            async for event in portfolio_assistant.astream_events(
                {"messages": [HumanMessage(content=request.user_message)]},
                config={"configurable": {"thread_id": request.session_id, "override_model": request.model}},
                version="v2",
            ):
                event_type = event.get("event", "")
                event_name = event.get("name", "")
                run_id = event.get("run_id", "")

                if event_type == "on_chat_model_stream":
                    tags = event.get("tags", [])
                    if "orchestrator_llm" not in tags:
                        continue

                    text = _chunk_to_text(event.get("data", {}).get("chunk"))
                    if text:
                        saw_tokens = True
                        accumulated_response += text
                        
                        # STREAMING GUARD: If the AI is leaking code, stop sending tokens to the UI.
                        _leak_markers = (
                            "```python", "import matplotlib", "plt.style.use", 
                            "pd.DataFrame", "plt.show(", "sns.set(", "import pandas"
                        )
                        is_leaking = any(m in accumulated_response for m in _leak_markers)
                        
                        if not is_leaking:
                            yield _stream_event({"type": "text-delta", "id": text_id, "delta": text})
                        else:
                            suppressed_stream_tokens = True
                            if len(accumulated_response) % 100 == 0:
                                logger.warning("Streaming Guard: Suppressing code leak in session %s", request.session_id)

                elif event_type == "on_tool_start":
                    # Filter out internal langgraph components
                    if not event_name.startswith("_") and event_name != "LangGraph":
                        tool_runs.add(run_id)
                        tool_names_run.add(event_name)
                        inputs = event.get("data", {}).get("input", {})
                        yield _stream_event({
                            "type": "tool-input-start", 
                            "toolCallId": run_id, 
                            "toolName": event_name, 
                            "dynamic": True
                        })
                        yield _stream_event({
                            "type": "tool-input-available", 
                            "toolCallId": run_id, 
                            "toolName": event_name, 
                            "input": inputs,
                            "dynamic": True
                        })

                elif event_type == "on_tool_end":
                    if run_id in tool_runs:
                        output = event.get("data", {}).get("output", "")
                        out_str = str(output)
                        # Truncate long outputs to avoid massive tool execution bubbles
                        if len(out_str) > 200:
                            out_str = out_str[:200] + "... [truncated]"
                        yield _stream_event({
                            "type": "tool-output-available", 
                            "toolCallId": run_id, 
                            "output": {"result": out_str}
                        })

                elif event_type == "on_chain_end" and event_name == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    messages = output.get("messages", []) if isinstance(output, dict) else []
                    if messages and (not saw_tokens or suppressed_stream_tokens):
                        final_text = _message_to_text(messages[-1])
                        if final_text:
                            accumulated_response = final_text
                            yield _stream_event({"type": "text-delta", "id": text_id, "delta": final_text})
                            suppressed_stream_tokens = False
                            final_sent = True

            if saw_tokens and not suppressed_stream_tokens and not final_sent:
                # Already yielded deltas, no need to yield final accumulated text again
                pass
            elif not accumulated_response and not final_sent:
                yield _stream_event({"type": "text-delta", "id": text_id, "delta": "Unable to generate a response for this request."})

            yield _stream_event({"type": "text-end", "id": text_id})

            # --- Emit interactive Plot events for all charts generated this request ---
            from src.agents.plot_store import GLOBAL_PLOT_IDS
            plot_ids = GLOBAL_PLOT_IDS.pop(request.session_id, None)

            if plot_ids:
                if isinstance(plot_ids, str):
                    plot_ids = [plot_ids]
                for p_id in plot_ids:
                    yield _stream_event({"type": "data-plot", "plotId": p_id})
            elif any(tool in tool_names_run for tool in ["generate_financial_plot", "plot_historical_prices"]):
                # Tool ran but produced no spec (e.g. PNG-only types like network/heatmap)
                pass

            yield _stream_event({"type": "finish", "messageId": msg_id, "finishReason": "stop"})

        except Exception as exc:
            logger.exception("Backend streaming advisory request failed")
            yield _stream_event({"type": "text-delta", "id": text_id, "delta": f"\n\nError: {exc}"})
            yield _stream_event({"type": "text-end", "id": text_id})
            yield _stream_event({"type": "finish", "messageId": msg_id, "finishReason": "error"})

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.get("/api/plots/{plot_id}")
def get_plot_data(plot_id: str):
    from src.memory.mongodb_memory_layer import MongoMemoryManager
    mongo = MongoMemoryManager()
    data = mongo.retrieve_plot(plot_id)
    if not data:
        raise HTTPException(status_code=404, detail="Plot not found or expired")
    return data

if __name__ == "__main__":
    import uvicorn

    host = "127.0.0.1"
    port = 8000

    if _is_port_open(host, port):
        owner = _port_owner(port)
        owner_line = f"\nPort owner: {owner}" if owner else ""
        print(
            f"Backend API is already running or port is occupied at http://{host}:{port}.\n"
            "Open http://127.0.0.1:8000/health to verify it, or stop the existing Python/uvicorn process before starting another."
            f"{owner_line}"
        )
    else:
        uvicorn.run("api.main:app", host=host, port=port, reload=False)
