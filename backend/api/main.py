import json
import logging
import shutil
import socket
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

print("=== MAIN.PY LOADED ===")
print("PATH:", sys.executable)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

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
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from src.memory.mongodb_memory_layer import MongoMemoryManager
from src.memory.context_resolver import (
    ContextResolver,
    build_direct_context_response,
    build_missing_input_response,
    build_pending_execution_response,
)
from src.memory.audit_log import GLOBAL_AUDIT_LOGGER
from src.memory.memory_store import InProcessSessionMemoryStore
from src.memory.missing_data_resolver import MissingDataResolver
from src.memory.response_contract import build_response_contract, contract_summary
from src.agents.plot_store import GLOBAL_PLOT_DATA
from src.decision.apg_bench_response import build_apg_bench_response
from src.decision.plot_prompt_response import build_plot_prompt_response
from src.decision.regime_response import build_regime_only_response

from src.orchestrator.chatbot_orchestrator import (
    CONFIGURED_DEFAULT_LLM_MODEL,
    FALLBACK_OLLAMA_MODEL,
    INSTALLED_OLLAMA_MODELS,
    PRIMARY_OLLAMA_MODEL,
    memory_manager,
    portfolio_assistant,
    streaming_portfolio_assistant,
)

session_memory_store = InProcessSessionMemoryStore()
context_resolver = ContextResolver()
missing_data_resolver = MissingDataResolver()



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LEGACY_OUTPUTS_DIR = PROJECT_ROOT / "src" / "outputs"
PLOT_TOKEN = "__PLOTSPEC__:"


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


from api.analytics_router import router as analytics_router
from api.governance_router import router as governance_router

app = FastAPI(
    title="Agentic Portfolio Governance API",
    description="Advisory-only backend for historical portfolio governance using local MongoDB data.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(analytics_router)
app.include_router(governance_router)


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


class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessageResponse]


class ChatSessionResponse(BaseModel):
    session_id: str
    title: str
    message_count: int
    created_at: str
    updated_at: str


class ChatSessionsResponse(BaseModel):
    sessions: list[ChatSessionResponse]


class DeleteChatSessionResponse(BaseModel):
    session_id: str
    deleted_count: int


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


async def _stream_text_delta(text_id: str, text: str, chunk_size: int = 18):
    """Yield text-delta events in small chunks so deterministic responses feel streamed."""
    import asyncio

    content = str(text or "")
    if not content:
        return

    for index in range(0, len(content), chunk_size):
        yield _stream_event({"type": "text-delta", "id": text_id, "delta": content[index:index + chunk_size]})
        await asyncio.sleep(0.01)


async def _run_blocking_io(func, *args, **kwargs):
    """Run synchronous Mongo/Supabase work without blocking the async stream loop."""
    import anyio

    return await anyio.to_thread.run_sync(lambda: func(*args, **kwargs))


def _attach_inline_plot_tokens(session_id: str, response_text: str, resolved: dict[str, Any]) -> str:
    if PLOT_TOKEN in str(response_text or ""):
        return response_text

    fallback = resolved.get("fallback_result", {}) if isinstance(resolved, dict) else {}
    payload = fallback.get("plot_payload") if isinstance(fallback.get("plot_payload"), dict) else None
    validation = resolved.get("validation_result") or fallback.get("validation_result") or {}
    if fallback.get("status") != "success" or not payload or not validation.get("can_render", True):
        return response_text

    import uuid

    inline_plot_id = f"detplot-{uuid.uuid4().hex}"
    plot_spec = _plot_spec_from_payload(payload)
    if not memory_manager.store_plot(inline_plot_id, plot_spec, ttl_days=7):
        logger.warning("Failed to store deterministic inline plot for session %s", session_id)
        return response_text
    return f"{response_text}\n\n{PLOT_TOKEN}{inline_plot_id}"


def _plot_spec_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    spec = dict(payload)
    chart_type = payload.get("chart_type")
    if chart_type in {"bar", "rangeBar", "histogram", "mirroredBar"}:
        spec["plot_type"] = "bar"
    elif chart_type in {"pie", "donut", "center_label_donut", "nested_donut", "semi_donut"}:
        spec["plot_type"] = "pie"
    elif chart_type in {"scatter", "bubble_scatter", "scatter_regression", "webgl_scatter"}:
        spec["plot_type"] = "scatter"
    else:
        spec["plot_type"] = payload.get("plot_type") or chart_type or "bar"
    spec["plot_id"] = payload.get("plot_short_id") or payload.get("plot_id")
    spec["title"] = payload.get("title") or spec["plot_id"] or "Chart"
    spec["series"] = payload.get("series") or [
        {
            "key": payload.get("x_axis") or "allocation_percent",
            "label": "Current allocation",
        }
    ]
    if spec["plot_type"] == "pie":
        spec["centerLabel"] = payload.get("centerLabel") or payload.get("center_label")
    if spec["plot_type"] == "line":
        spec["x_label"] = payload.get("x_label") or payload.get("x_axis") or "Date"
        spec["y_label"] = payload.get("y_label") or payload.get("y_axis") or "Value"
        spec["connect_nulls"] = bool(payload.get("connect_nulls", False))
        spec["curve"] = payload.get("curve") or "linear"
    if spec["plot_type"] == "scatter":
        x_axis_label = payload.get("xAxis", [{}])[0].get("label") if isinstance(payload.get("xAxis"), list) and payload.get("xAxis") else None
        y_axis_label = payload.get("yAxis", [{}])[0].get("label") if isinstance(payload.get("yAxis"), list) and payload.get("yAxis") else None
        spec["x_label"] = payload.get("x_label") or x_axis_label or payload.get("x_axis") or "X"
        spec["y_label"] = payload.get("y_label") or y_axis_label or payload.get("y_axis") or "Y"
        if payload.get("x_unit") == "%":
            spec["x_format"] = "percent"
        if payload.get("y_unit") == "%":
            spec["y_format"] = "percent"
    spec["layout"] = payload.get("layout") or ("horizontal" if payload.get("bar_mode") == "horizontal" else "vertical")
    spec["sort"] = payload.get("sort") or "descending"
    if spec["plot_type"] == "pie":
        spec["height"] = payload.get("height") or 420
    elif spec["plot_type"] == "scatter":
        spec["height"] = payload.get("height") or 420
    else:
        spec["height"] = payload.get("height") or max(360, min(860, len(payload.get("data", [])) * 36 + 116))
    return spec


def _persist_chat_message(
    session_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    session_memory_store.append_message(session_id, role, content, metadata=metadata)
    try:
        memory_manager.append_chat_message(session_id, role, content, metadata=metadata)
    except Exception as exc:
        logger.warning("Failed to persist %s chat message for session %s: %s", role, session_id, exc)


def _resolve_chat_memory(session_id: str, message: str) -> dict[str, Any]:
    persisted_messages = memory_manager.list_chat_messages(session_id, limit=25)
    if persisted_messages:
        session_memory_store.hydrate_messages(session_id, persisted_messages)
    last_25 = session_memory_store.get_last_messages(session_id, limit=25)
    state = session_memory_store.get_state(session_id)
    resolved = context_resolver.resolve(message, state, chat_history_last_25=last_25)
    resolved = missing_data_resolver.resolve(resolved)
    session_memory_store.save_state(session_id, resolved["session_state"])
    _audit_resolved_memory(session_id, message, resolved)
    return resolved


def _persist_memory_response(session_id: str, response_text: str, resolved: dict[str, Any], metadata: dict[str, Any]) -> None:
    state = dict(resolved.get("session_state") or {})
    state["last_response_summary"] = response_text[:500]
    session_memory_store.save_state(session_id, state)
    contract = build_response_contract(response_text, resolved, result=response_text[:500])
    enriched_metadata = {
        **(metadata or {}),
        "response_contract_summary": contract_summary(contract),
        "response_contract": {k: v for k, v in contract.items() if k != "debug_user_message"},
    }
    _persist_chat_message(session_id, "assistant", response_text, metadata=enriched_metadata)


def _audit_resolved_memory(session_id: str, message: str, resolved: dict[str, Any]) -> None:
    fallback = resolved.get("fallback_result", {})
    payload = fallback.get("plot_payload") if isinstance(fallback.get("plot_payload"), dict) else {}
    validation = resolved.get("validation_result") or fallback.get("validation_result") or {}
    if not payload and fallback.get("status") == "not_applicable":
        return
    status = "success" if fallback.get("status") == "success" and validation.get("can_render", True) else "blocked"
    reason = validation.get("reason") or fallback.get("reason")
    GLOBAL_AUDIT_LOGGER.log(
        {
            "session_id": session_id,
            "user_message": message,
            "intent": resolved.get("intent_lock", {}).get("intent") or resolved.get("session_state", {}).get("last_sub_intent"),
            "resolved_universe": resolved.get("session_state", {}).get("active_universe"),
            "tool_called": "missing_data_resolver",
            "plot_id": payload.get("plot_id") or resolved.get("session_state", {}).get("last_plot_id"),
            "data_source": payload.get("data_source") or fallback.get("data_source"),
            "status": status,
            "failure_reason": None if status == "success" else reason,
        }
    )


@app.get("/health")
def health_check() -> dict:
    mongo_status = getattr(app.state, "mongo_available", False)
    
    import os
    has_ashna_key = bool(os.getenv("ASHNA_API_KEY"))
    
    # Basic check for model presence
    ollama_status = (
        PRIMARY_OLLAMA_MODEL in INSTALLED_OLLAMA_MODELS 
        or (PRIMARY_OLLAMA_MODEL.startswith("ashna") and has_ashna_key)
    )
    default_llm_status = bool(
        CONFIGURED_DEFAULT_LLM_MODEL
        and CONFIGURED_DEFAULT_LLM_MODEL.startswith("ashna")
        and has_ashna_key
    )
    
    available_models = list(INSTALLED_OLLAMA_MODELS)
    if has_ashna_key:
        ashna_models = [
            "ashnaai",
            "ashna-x1",
            "ashna/gpt-4o",
            "ashna/gpt-4o-mini",
            "ashna/gpt-3.5-turbo"
        ]
        for model in reversed(ashna_models):
            if model not in available_models:
                available_models.insert(0, model)
    
    return {
        "status": "ok" if mongo_status and (ollama_status or default_llm_status) else "degraded",
        "mode": "advisory-only",
        "data_source": "local-mongodb-historical-only",
        "components": {
            "mongodb": "connected" if mongo_status else "disconnected",
            "supabase_postgres": "connected" if memory_manager.pg_pool else "not_configured",
            "ollama": "ready" if ollama_status else "model_missing",
            "default_llm": "ready" if default_llm_status else "not_configured",
            "ashnaai": "ready" if has_ashna_key else "not_configured"
        },
        "models": {
            "primary": PRIMARY_OLLAMA_MODEL,
            "fallback": FALLBACK_OLLAMA_MODEL,
            "default": CONFIGURED_DEFAULT_LLM_MODEL,
            "available": available_models
        }
    }


@app.get("/chat/sessions", response_model=ChatSessionsResponse)
def chat_sessions(limit: int = 50) -> ChatSessionsResponse:
    safe_limit = max(1, min(int(limit or 50), 100))
    sessions = memory_manager.list_chat_sessions(limit=safe_limit)
    return ChatSessionsResponse(sessions=sessions)


@app.get("/chat/{session_id}/messages", response_model=ChatHistoryResponse)
def chat_messages(session_id: str, limit: int = 200) -> ChatHistoryResponse:
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty")

    rows = memory_manager.list_chat_messages(session_id, limit=limit)
    return ChatHistoryResponse(session_id=session_id, messages=rows)


@app.delete("/chat/{session_id}", response_model=DeleteChatSessionResponse)
def delete_chat_session(session_id: str) -> DeleteChatSessionResponse:
    clean_session_id = str(session_id or "").strip()
    if not clean_session_id:
        raise HTTPException(status_code=400, detail="session_id cannot be empty")

    deleted_count = 0
    if hasattr(memory_manager, "delete_chat_session"):
        deleted_count = int(memory_manager.delete_chat_session(clean_session_id) or 0)
    session_memory_store.delete_session(clean_session_id)
    return DeleteChatSessionResponse(session_id=clean_session_id, deleted_count=deleted_count)


@app.get("/audit/recent")
def recent_audit_records(limit: int = 100) -> dict[str, Any]:
    return {"records": GLOBAL_AUDIT_LOGGER.recent(limit=limit)}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty")

    if not request.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message cannot be empty")

    try:
        logger.info("Processing chat request for session_id=%s", request.session_id)
        _persist_chat_message(
            request.session_id,
            "user",
            request.user_message,
            metadata={"model": request.model, "transport": "rest"},
        )
        resolved_memory = _resolve_chat_memory(request.session_id, request.user_message)

        pending_execution_response = build_pending_execution_response(resolved_memory)
        if pending_execution_response is not None:
            pending_execution_response = _attach_inline_plot_tokens(
                request.session_id,
                pending_execution_response,
                resolved_memory,
            )
            _persist_memory_response(
                request.session_id,
                pending_execution_response,
                resolved_memory,
                metadata={"model": request.model, "transport": "rest", "router_fast_path": "pending_action"},
            )
            return ChatResponse(session_id=request.session_id, response=pending_execution_response)

        direct_context_response = build_direct_context_response(resolved_memory)
        if direct_context_response is not None:
            direct_context_response = _attach_inline_plot_tokens(
                request.session_id,
                direct_context_response,
                resolved_memory,
            )
            _persist_memory_response(
                request.session_id,
                direct_context_response,
                resolved_memory,
                metadata={"model": request.model, "transport": "rest", "router_fast_path": "direct_context"},
            )
            return ChatResponse(session_id=request.session_id, response=direct_context_response)

        missing_input_response = build_missing_input_response(resolved_memory)
        if missing_input_response is not None:
            _persist_memory_response(
                request.session_id,
                missing_input_response,
                resolved_memory,
                metadata={"model": request.model, "transport": "rest", "router_fast_path": "memory_missing_input"},
            )
            return ChatResponse(session_id=request.session_id, response=missing_input_response)

        benchmark_response = build_apg_bench_response(request.user_message)
        if benchmark_response is not None:
            _persist_memory_response(
                request.session_id,
                benchmark_response,
                resolved_memory,
                metadata={"model": request.model, "transport": "rest", "router_fast_path": "apg_bench"},
            )
            return ChatResponse(session_id=request.session_id, response=benchmark_response)

        plot_response = build_plot_prompt_response(request.user_message)
        if plot_response is not None:
            _persist_memory_response(
                request.session_id,
                plot_response,
                resolved_memory,
                metadata={"model": request.model, "transport": "rest", "router_fast_path": "plot_prompt"},
            )
            return ChatResponse(session_id=request.session_id, response=plot_response)

        regime_response = build_regime_only_response(
            request.user_message,
            previous_analysis={
                "analysis_id": resolved_memory.get("session_state", {}).get("active_analysis_id"),
                "entities": {
                    "universe": resolved_memory.get("session_state", {}).get("active_universe"),
                    "tickers": resolved_memory.get("session_state", {}).get("active_tickers", []),
                    "current_weights": resolved_memory.get("session_state", {}).get("active_weights", {}).get("weights", {}),
                    "start_date": resolved_memory.get("session_state", {}).get("active_date_range", {}).get("start"),
                    "end_date": resolved_memory.get("session_state", {}).get("active_date_range", {}).get("end"),
                },
            },
        )
        if regime_response is not None:
            _persist_memory_response(
                request.session_id,
                regime_response,
                resolved_memory,
                metadata={"model": request.model, "transport": "rest", "router_fast_path": "regime_only"},
            )
            return ChatResponse(session_id=request.session_id, response=regime_response)

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

        _persist_memory_response(
            request.session_id,
            response_text,
            resolved_memory,
            metadata={"model": request.model, "transport": "rest"},
        )
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
            yield _stream_event({
                "type": "status",
                "stage": "data_access",
                "label": "Saving message and opening Mongo/Supabase context",
            })
            await _run_blocking_io(
                _persist_chat_message,
                request.session_id,
                "user",
                request.user_message,
                metadata={"model": request.model, "transport": "stream"},
            )
            yield _stream_event({
                "type": "status",
                "stage": "context",
                "label": "Loading recent memory and portfolio context",
            })
            resolved_memory = await _run_blocking_io(
                _resolve_chat_memory,
                request.session_id,
                request.user_message,
            )
            yield _stream_event({
                "type": "status",
                "stage": "generation",
                "label": "Streaming response",
            })

            pending_execution_response = build_pending_execution_response(resolved_memory)
            if pending_execution_response is not None:
                pending_execution_response = await _run_blocking_io(
                    _attach_inline_plot_tokens,
                    request.session_id,
                    pending_execution_response,
                    resolved_memory,
                )
                accumulated_response = pending_execution_response
                async for chunk in _stream_text_delta(text_id, pending_execution_response):
                    yield chunk
                yield _stream_event({"type": "text-end", "id": text_id})
                await _run_blocking_io(
                    _persist_memory_response,
                    request.session_id,
                    accumulated_response,
                    resolved_memory,
                    metadata={
                        "model": request.model,
                        "transport": "stream",
                        "router_fast_path": "pending_action",
                        "plot_ids": [],
                    },
                )
                yield _stream_event({"type": "finish", "messageId": msg_id, "finishReason": "stop"})
                return

            direct_context_response = build_direct_context_response(resolved_memory)
            if direct_context_response is not None:
                direct_context_response = await _run_blocking_io(
                    _attach_inline_plot_tokens,
                    request.session_id,
                    direct_context_response,
                    resolved_memory,
                )
                accumulated_response = direct_context_response
                async for chunk in _stream_text_delta(text_id, direct_context_response):
                    yield chunk
                yield _stream_event({"type": "text-end", "id": text_id})
                await _run_blocking_io(
                    _persist_memory_response,
                    request.session_id,
                    accumulated_response,
                    resolved_memory,
                    metadata={
                        "model": request.model,
                        "transport": "stream",
                        "router_fast_path": "direct_context",
                        "plot_ids": [],
                    },
                )
                yield _stream_event({"type": "finish", "messageId": msg_id, "finishReason": "stop"})
                return

            missing_input_response = build_missing_input_response(resolved_memory)
            if missing_input_response is not None:
                accumulated_response = missing_input_response
                async for chunk in _stream_text_delta(text_id, missing_input_response):
                    yield chunk
                yield _stream_event({"type": "text-end", "id": text_id})
                await _run_blocking_io(
                    _persist_memory_response,
                    request.session_id,
                    accumulated_response,
                    resolved_memory,
                    metadata={
                        "model": request.model,
                        "transport": "stream",
                        "router_fast_path": "memory_missing_input",
                        "plot_ids": [],
                    },
                )
                yield _stream_event({"type": "finish", "messageId": msg_id, "finishReason": "stop"})
                return

            benchmark_response = build_apg_bench_response(request.user_message)
            if benchmark_response is not None:
                accumulated_response = benchmark_response
                async for chunk in _stream_text_delta(text_id, benchmark_response):
                    yield chunk
                yield _stream_event({"type": "text-end", "id": text_id})
                await _run_blocking_io(
                    _persist_memory_response,
                    request.session_id,
                    accumulated_response,
                    resolved_memory,
                    metadata={
                        "model": request.model,
                        "transport": "stream",
                        "router_fast_path": "apg_bench",
                        "plot_ids": [],
                    },
                )
                yield _stream_event({"type": "finish", "messageId": msg_id, "finishReason": "stop"})
                return

            plot_response = build_plot_prompt_response(request.user_message)
            if plot_response is not None:
                accumulated_response = plot_response
                async for chunk in _stream_text_delta(text_id, plot_response):
                    yield chunk
                yield _stream_event({"type": "text-end", "id": text_id})
                await _run_blocking_io(
                    _persist_memory_response,
                    request.session_id,
                    accumulated_response,
                    resolved_memory,
                    metadata={
                        "model": request.model,
                        "transport": "stream",
                        "router_fast_path": "plot_prompt",
                        "plot_ids": [],
                    },
                )
                yield _stream_event({"type": "finish", "messageId": msg_id, "finishReason": "stop"})
                return

            regime_response = build_regime_only_response(
                request.user_message,
                previous_analysis={
                    "analysis_id": resolved_memory.get("session_state", {}).get("active_analysis_id"),
                    "entities": {
                        "universe": resolved_memory.get("session_state", {}).get("active_universe"),
                        "tickers": resolved_memory.get("session_state", {}).get("active_tickers", []),
                        "current_weights": resolved_memory.get("session_state", {}).get("active_weights", {}).get("weights", {}),
                        "start_date": resolved_memory.get("session_state", {}).get("active_date_range", {}).get("start"),
                        "end_date": resolved_memory.get("session_state", {}).get("active_date_range", {}).get("end"),
                    },
                },
            )
            if regime_response is not None:
                accumulated_response = regime_response
                async for chunk in _stream_text_delta(text_id, regime_response):
                    yield chunk
                yield _stream_event({"type": "text-end", "id": text_id})
                await _run_blocking_io(
                    _persist_memory_response,
                    request.session_id,
                    accumulated_response,
                    resolved_memory,
                    metadata={
                        "model": request.model,
                        "transport": "stream",
                        "router_fast_path": "regime_only",
                        "plot_ids": [],
                    },
                )
                yield _stream_event({"type": "finish", "messageId": msg_id, "finishReason": "stop"})
                return

            async for event in streaming_portfolio_assistant.astream_events(
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
                            async for chunk in _stream_text_delta(text_id, final_text):
                                yield chunk
                            suppressed_stream_tokens = False
                            final_sent = True

            if saw_tokens and not suppressed_stream_tokens and not final_sent:
                # Already yielded deltas, no need to yield final accumulated text again
                pass
            elif not accumulated_response and not final_sent:
                async for chunk in _stream_text_delta(text_id, "Unable to generate a response for this request."):
                    yield chunk

            yield _stream_event({"type": "text-end", "id": text_id})

            # --- Emit interactive Plot events for all charts generated this request ---
            from src.agents.plot_store import GLOBAL_PLOT_IDS
            plot_ids = GLOBAL_PLOT_IDS.pop(request.session_id, None)

            if plot_ids:
                if isinstance(plot_ids, str):
                    plot_ids = [plot_ids]
                for p_id in plot_ids:
                    accumulated_response += f"\n{PLOT_TOKEN}{p_id}"
                    yield _stream_event({"type": "data-plot", "plotId": p_id})
            elif any(tool in tool_names_run for tool in ["generate_financial_plot", "plot_historical_prices"]):
                # Tool ran but produced no spec (e.g. PNG-only types like network/heatmap)
                pass

            await _run_blocking_io(
                _persist_memory_response,
                request.session_id,
                accumulated_response,
                resolved_memory,
                metadata={
                    "model": request.model,
                    "transport": "stream",
                    "tool_names": sorted(tool_names_run),
                    "plot_ids": plot_ids or [],
                },
            )
            yield _stream_event({"type": "finish", "messageId": msg_id, "finishReason": "stop"})

        except Exception as exc:
            logger.exception("Backend streaming advisory request failed")
            error_detail = str(exc) or exc.__class__.__name__
            yield _stream_event({"type": "text-delta", "id": text_id, "delta": f"\n\nError: {error_detail}"})
            yield _stream_event({"type": "text-end", "id": text_id})
            yield _stream_event({"type": "finish", "messageId": msg_id, "finishReason": "error"})

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.get("/api/plots/{plot_id}")
def get_plot_data(plot_id: str):
    if plot_id == "test-line":
        return {
            "plot_type": "line",
            "title": "AAPL vs MSFT Price Trend (Interactive)",
            "x_label": "Date",
            "x_type": "time",
            "y_label": "Close Price (USD)",
            "y_format": "currency",
            "grid": {"horizontal": True},
            "curve": "monotoneX",
            "highlightScope": {"highlight": "series", "fade": "global"},
            "experimentalFeatures": {"enablePositionBasedPointerInteraction": True},
            "series": [
                {
                    "name": "AAPL",
                    "label": "AAPL",
                    "color": "#3b82f6",
                    "showMark": True,
                    "data": [
                        {"x": "2024-01-02", "y": 184.89},
                        {"x": "2024-02-01", "y": 186.86},
                        {"x": "2024-03-01", "y": 179.66},
                        {"x": "2024-04-01", "y": 170.03},
                        {"x": "2024-05-01", "y": 169.30},
                        {"x": "2024-06-03", "y": 194.03},
                    ],
                },
                {
                    "name": "MSFT",
                    "label": "MSFT",
                    "color": "#10b981",
                    "showMark": True,
                    "data": [
                        {"x": "2024-01-02", "y": 370.87},
                        {"x": "2024-02-01", "y": 403.78},
                        {"x": "2024-03-01", "y": 415.50},
                        {"x": "2024-04-01", "y": 424.57},
                        {"x": "2024-05-01", "y": 394.94},
                        {"x": "2024-06-03", "y": 413.52},
                    ],
                },
            ],
        }
    if plot_id == "test-candlestick":
        return {
            "plot_type": "candlestick",
            "title": "AAPL Candlestick Chart (Interactive)",
            "series": [
                {
                    "name": "AAPL",
                    "data": [
                        {"date": "2024-01-02", "open": 183.73, "high": 185.22, "low": 182.5, "close": 184.89, "volume": 120000000},
                        {"date": "2024-01-03", "open": 184.89, "high": 186.15, "low": 184.22, "close": 185.92, "volume": 98000000},
                        {"date": "2024-01-04", "open": 185.92, "high": 187.3, "low": 185.5, "close": 186.8, "volume": 85000000},
                        {"date": "2024-01-05", "open": 186.8, "high": 189.5, "low": 186.0, "close": 188.9, "volume": 105000000},
                        {"date": "2024-01-08", "open": 188.9, "high": 190.2, "low": 187.5, "close": 188.2, "volume": 92000000},
                        {"date": "2024-01-09", "open": 188.2, "high": 191.5, "low": 187.0, "close": 190.5, "volume": 88000000},
                        {"date": "2024-01-10", "open": 190.5, "high": 192.3, "low": 189.8, "close": 191.9, "volume": 95000000},
                        {"date": "2024-01-11", "open": 191.9, "high": 193.0, "low": 190.5, "close": 191.2, "volume": 80000000},
                        {"date": "2024-01-12", "open": 191.2, "high": 192.5, "low": 189.2, "close": 190.1, "volume": 78000000},
                        {"date": "2024-01-16", "open": 190.1, "high": 191.0, "low": 188.5, "close": 189.3, "volume": 82000000},
                        {"date": "2024-01-17", "open": 189.3, "high": 190.5, "low": 187.2, "close": 188.0, "volume": 90000000},
                        {"date": "2024-01-18", "open": 188.0, "high": 192.0, "low": 187.8, "close": 191.5, "volume": 115000000},
                        {"date": "2024-01-19", "open": 191.5, "high": 193.5, "low": 191.0, "close": 192.8, "volume": 102000000},
                        {"date": "2024-01-22", "open": 192.8, "high": 194.2, "low": 192.0, "close": 193.5, "volume": 85000000},
                        {"date": "2024-01-23", "open": 193.5, "high": 195.0, "low": 193.0, "close": 194.5, "volume": 72000000},
                        {"date": "2024-01-24", "open": 194.5, "high": 196.3, "low": 194.0, "close": 195.2, "volume": 94000000},
                        {"date": "2024-01-25", "open": 195.2, "high": 195.5, "low": 192.8, "close": 193.8, "volume": 89000000},
                        {"date": "2024-01-26", "open": 193.8, "high": 194.8, "low": 192.5, "close": 193.2, "volume": 68000000},
                        {"date": "2024-01-29", "open": 193.2, "high": 195.2, "low": 192.8, "close": 194.9, "volume": 74000000},
                        {"date": "2024-01-30", "open": 194.9, "high": 196.0, "low": 194.0, "close": 195.5, "volume": 80000000},
                        {"date": "2024-01-31", "open": 195.5, "high": 195.8, "low": 192.2, "close": 193.1, "volume": 122000000},
                    ]
                }
            ]
        }
    elif plot_id == "test-pie":
        return {
            "plot_type": "pie",
            "title": "Portfolio Asset Weights (Interactive)",
            "centerLabel": "Allocation",
            "series": [
                {
                    "data": [
                        {"id": "AAPL", "value": 0.40, "color": "#3b82f6"},
                        {"id": "MSFT", "value": 0.30, "color": "#10b981"},
                        {"id": "GOOG", "value": 0.20, "color": "#f59e0b"},
                        {"id": "AMZN", "value": 0.10, "color": "#ef4444"}
                    ],
                    "innerRadius": 60,
                    "outerRadius": 110,
                    "paddingAngle": 3,
                    "cornerRadius": 6,
                    "arcLabel": "percent"
                }
            ]
        }
    elif plot_id == "test-bar":
        return {
            "plot_type": "bar",
            "title": "Quarterly Performance (Interactive)",
            "xAxis": [
                {
                    "dataKey": "label",
                    "scaleType": "band",
                    "categoryGapRatio": 0.3,
                    "barGapRatio": 0.1
                }
            ],
            "series": [
                {
                    "name": "Revenue",
                    "label": "Revenue",
                    "color": "#3b82f6",
                    "barLabel": "value",
                    "data": [
                        {"x": "Q1", "y": 120},
                        {"x": "Q2", "y": 150},
                        {"x": "Q3", "y": 180},
                        {"x": "Q4", "y": 220}
                    ]
                },
                {
                    "name": "Expenses",
                    "label": "Expenses",
                    "color": "#ef4444",
                    "barLabel": "value",
                    "data": [
                        {"x": "Q1", "y": 90},
                        {"x": "Q2", "y": 100},
                        {"x": "Q3", "y": 110},
                        {"x": "Q4", "y": 130}
                    ]
                }
            ]
        }
    elif plot_id == "test-smart-bar":
        tickers = [
            "AAPL", "MSFT", "NVDA", "AMZN", "JPM", "GOOGL", "META", "AVGO", "LLY", "V",
            "XOM", "UNH", "MA", "COST", "HD", "PG", "NFLX", "CRM", "ADBE", "BAC",
        ]
        return {
            "plot_type": "bar",
            "plot_id": "ticker_concentration_plot",
            "chart_type": "bar",
            "bar_mode": "horizontal",
            "title": "Ticker Concentration",
            "description": "APG-style horizontal bar chart fixture.",
            "universe": "U1",
            "analysis_id": "fixture-smart-bar-u1",
            "x_axis": "allocation_percent",
            "y_axis": "ticker",
            "unit": "percent",
            "sort": "descending",
            "series": [{"key": "allocation_percent", "label": "Current allocation"}],
            "data": [
                {"ticker": ticker, "allocation_percent": round(14.5 - index * 0.48, 2)}
                for index, ticker in enumerate(tickers)
            ],
            "thresholds": [{"name": "Max ticker cap", "value": 20}],
            "interpretation": "Shows whether individual ticker exposures exceed advisory concentration thresholds.",
        }
    elif plot_id == "test-sankey":
        return {
            "plot_type": "sankey",
            "title": "Governance Fund Flow (Interactive)",
            "valueFormatter": "currency",
            "nodes": [
                {"id": "A", "label": "Dividends Received", "color": "#3b82f6"},
                {"id": "B", "label": "Reinvestment", "color": "#10b981"},
                {"id": "C", "label": "Taxes Paid", "color": "#ef4444"},
                {"id": "D", "label": "Retained Cash Reserve", "color": "#f59e0b"}
            ],
            "links": [
                {"source": "A", "target": "B", "value": 50000},
                {"source": "A", "target": "C", "value": 15000},
                {"source": "A", "target": "D", "value": 35000}
            ]
        }
    elif plot_id == "test-heatmap":
        return {
            "plot_type": "heatmap",
            "title": "Return Correlation Heatmap (Premium)",
            "xAxis": [{"data": ["AAPL", "MSFT", "NVDA", "JPM"]}],
            "yAxis": [{"data": ["AAPL", "MSFT", "NVDA", "JPM"]}],
            "series": [{
                "data": [
                    [0, 0, 1.0], [1, 0, 0.72], [2, 0, 0.64], [3, 0, 0.28],
                    [0, 1, 0.72], [1, 1, 1.0], [2, 1, 0.69], [3, 1, 0.31],
                    [0, 2, 0.64], [1, 2, 0.69], [2, 2, 1.0], [3, 2, 0.22],
                    [0, 3, 0.28], [1, 3, 0.31], [2, 3, 0.22], [3, 3, 1.0],
                ]
            }],
            "height": 360,
        }
    elif plot_id == "test-funnel":
        return {
            "plot_type": "funnel",
            "title": "Data Quality Validation Funnel (Premium)",
            "series": [{
                "label": "Validation",
                "layout": "vertical",
                "curve": "linear",
                "borderRadius": 6,
                "data": [
                    {"id": "loaded", "label": "Loaded Tickers", "value": 100, "color": "#3b82f6"},
                    {"id": "fresh", "label": "Fresh Data", "value": 92, "color": "#10b981"},
                    {"id": "complete", "label": "Complete Returns", "value": 86, "color": "#f59e0b"},
                    {"id": "validated", "label": "Governance Validated", "value": 78, "color": "#8b5cf6"},
                ]
            }],
            "height": 360,
        }
    elif plot_id == "test-radar":
        return {
            "plot_type": "radar",
            "title": "Governance Score Radar (Premium)",
            "radar": {"metrics": ["Quality", "Diversification", "Downside", "Liquidity", "Traceability"]},
            "series": [
                {"label": "Current", "data": [76, 61, 58, 84, 91], "fillArea": True, "color": "#3b82f6"},
                {"label": "Advisory", "data": [82, 78, 73, 80, 95], "fillArea": True, "color": "#10b981"},
            ],
            "height": 360,
        }
    elif plot_id == "test-gauge":
        return {
            "plot_type": "gauge",
            "title": "Confidence Score Gauge (Premium)",
            "value": 87,
            "valueMin": 0,
            "valueMax": 100,
            "height": 260,
            "text": "87%",
        }
    elif plot_id == "test-radial-bar":
        return {
            "plot_type": "radial_bar",
            "title": "Risk Contribution by Sector (Premium)",
            "categories": ["Tech", "Finance", "Health", "Energy", "Cash"],
            "series": [
                {"label": "Risk", "data": [38, 24, 17, 13, 8], "color": "#3b82f6"},
                {"label": "Allocation", "data": [32, 25, 19, 14, 10], "color": "#10b981"},
            ],
            "height": 360,
        }
    elif plot_id == "test-radial-line":
        return {
            "plot_type": "radial_line",
            "title": "Regime Component Profile (Premium)",
            "categories": ["Volatility", "Correlation", "Drawdown", "Turnover", "Liquidity"],
            "series": [
                {"label": "Current", "data": [62, 74, 48, 35, 22], "curve": "linear", "closePath": True, "color": "#f59e0b"},
                {"label": "Threshold", "data": [70, 70, 70, 70, 70], "curve": "linear", "closePath": True, "color": "#ef4444"},
            ],
            "height": 360,
        }
    elif plot_id == "test-scatter":
        return {
            "plot_type": "scatter",
            "title": "Risk vs Expected Return (Interactive)",
            "x_label": "Volatility (Risk %)",
            "y_label": "Expected Return (%)",
            "grid": {"horizontal": True, "vertical": True},
            "hitAreaRadius": 20,
            "highlightScope": {"highlight": "series", "fade": "global"},
            "xAxis": [{"min": 0, "label": "Volatility (Risk %)", "height": 36}],
            "yAxis": [{"min": 0, "label": "Expected Return (%)", "width": 60}],
            "zAxis": [{"min": 0, "max": 10}],
            "series": [
                {
                    "name": "Equities",
                    "label": "Equities",
                    "color": "#3b82f6",
                    "markerSize": 8,
                    "data": [
                        {"x": 12, "y": 8, "z": 7, "id": "AAPL"},
                        {"x": 15, "y": 10, "z": 9, "id": "TSLA"},
                        {"x": 10, "y": 6, "z": 6, "id": "MSFT"}
                    ],
                },
                {
                    "name": "Bonds",
                    "label": "Bonds",
                    "color": "#10b981",
                    "markerSize": 6,
                    "data": [
                        {"x": 4, "y": 2, "z": 3, "id": "UST10Y"},
                        {"x": 5, "y": 3, "z": 4, "id": "CORP"}
                    ],
                }
            ]
        }
    elif plot_id == "test-sparkline":
        return {
            "plot_type": "sparkline",
            "title": "Asset Valuation History (Interactive Sparkline)",
            "data": [10, 12, 15, 13, 16, 18, 17, 21, 24, 23, 26],
            "plotType": "line",
            "area": True,
            "curve": "natural",
            "color": "#10b981",
            "height": 60
        }

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

