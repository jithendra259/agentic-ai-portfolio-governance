import logging
import os
import uuid
import json
import re
from urllib.parse import urlparse

import gradio as gr
import requests


API_URL = os.getenv("PORTFOLIO_API_URL", "http://127.0.0.1:8000/chat")
STREAM_API_URL = os.getenv("PORTFOLIO_STREAM_API_URL", "http://127.0.0.1:8000/chat/stream")
SESSION_ID = os.getenv("PORTFOLIO_SESSION_ID", str(uuid.uuid4()))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_session_id() -> str:
    return SESSION_ID


def _backend_base_url() -> str:
    parsed = urlparse(STREAM_API_URL)
    return f"{parsed.scheme}://{parsed.netloc}"


def _rewrite_plot_markdown(content: str) -> str:
    if not content:
        return content

    base_url = _backend_base_url().rstrip("/")

    def replace_relative(match: re.Match) -> str:
        path = match.group(1).lstrip("./")
        return f"]({base_url}/{path})"

    def replace_rooted(match: re.Match) -> str:
        path = match.group(1)
        return f"]({base_url}{path})"

    def replace_absolute_outputs(match: re.Match) -> str:
        filename = match.group(1)
        return f"]({base_url}/outputs/{filename})"

    content = re.sub(r"\]\((outputs/[^)]+)\)", replace_relative, content)
    content = re.sub(r"\]\((/outputs/[^)]+)\)", replace_rooted, content)
    content = re.sub(
        r"\]\((?:[A-Za-z]:)?[^)]*[\\/]+outputs[\\/]+([^\\/)\s]+)\)",
        replace_absolute_outputs,
        content,
    )
    return content


def chat_with_api(user_message: str, history, session_id: str):
    """
    Stream the user's request through the local FastAPI backend.
    This interface is advisory-only and relies on historical MongoDB-backed analysis.
    """
    try:
        response = requests.post(
            STREAM_API_URL,
            json={"session_id": session_id, "user_message": user_message},
            timeout=(10, None),
            stream=True,
        )
        response.raise_for_status()

        accumulated_response = ""
        status_message = ""

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            event = json.loads(line)
            event_type = event.get("type")
            content = event.get("content", "")

            if event_type == "status":
                status_message = content
                if not accumulated_response:
                    yield _rewrite_plot_markdown(status_message)
            elif event_type == "token":
                accumulated_response += content
                yield _rewrite_plot_markdown(accumulated_response)
            elif event_type == "final":
                accumulated_response = content or accumulated_response
                yield _rewrite_plot_markdown(accumulated_response)
            elif event_type == "error":
                yield _rewrite_plot_markdown(f"System Error: {content}")
                return

        if not accumulated_response and status_message:
            yield _rewrite_plot_markdown(status_message)
        elif not accumulated_response:
            yield "No response returned from backend."
    except requests.exceptions.ConnectionError:
        logger.exception("Backend API is not reachable")
        yield (
            "Backend API is not running on http://127.0.0.1:8000.\n\n"
            "Start it in a separate terminal with:\n"
            ".\\venv\\Scripts\\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload"
        )
    except requests.exceptions.ReadTimeout:
        logger.exception("Backend streaming request timed out")
        yield (
            "The backend took too long to respond.\n\n"
            "Please check whether the model or MongoDB query is stuck, then retry."
        )
    except requests.exceptions.RequestException as exc:
        logger.exception("Failed to call backend API")
        yield f"System Error: Unable to reach backend API. Details: {exc}"


def build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        session_state = gr.State(create_session_id())

        gr.Markdown("# Agentic Portfolio Governance Assistant")
        gr.Markdown(
            """
            ### Advisory-Only System
            - No live market data: all analysis uses historical MongoDB snapshots.
            - No trade execution: this is a research prototype, not a live portfolio manager.
            - Historical analysis only: governance pipeline backtests scenarios from 2005 to 2025.
            - Human in the loop: elevated instability regimes are intended for researcher review.
            """
        )

        gr.ChatInterface(
            fn=chat_with_api,
            additional_inputs=[session_state],
            textbox=gr.Textbox(
                placeholder="Ask about historical portfolio risk, universes, sectors, or CVaR optimization...",
                container=False,
                scale=7,
            ),
        )

    return demo


demo = build_demo()


def main() -> None:
    demo.launch(theme=gr.themes.Soft(), share=False)


if __name__ == "__main__":
    main()
