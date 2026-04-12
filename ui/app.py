import logging
import os
import uuid
import json
import re
import datetime
from urllib.parse import urlparse

import gradio as gr
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
API_URL = os.getenv("PORTFOLIO_API_URL", "http://127.0.0.1:8000/chat")
STREAM_API_URL = os.getenv("PORTFOLIO_STREAM_API_URL", "http://127.0.0.1:8000/chat/stream")
HEALTH_URL = os.getenv("PORTFOLIO_HEALTH_URL", "http://127.0.0.1:8000/health")
SESSION_ID_FILE = ".session_id"

PREMIUM_CSS = """
/* Premium Deep Sea Blue Theme with Glassmorphism */
:root {
    --bg-color: #050a15;
    --card-bg: rgba(16, 25, 45, 0.7);
    --accent-color: #00d2ff;
    --accent-glow: rgba(0, 210, 255, 0.3);
    --text-primary: #e6f1ff;
    --text-secondary: #8892b0;
    --border-color: rgba(0, 210, 255, 0.2);
    --success-color: #64ffda;
    --error-color: #ff4d4d;
}

body, .gradio-container {
    background-color: var(--bg-color) !important;
    background-image: 
        radial-gradient(circle at 20% 30%, rgba(0, 210, 255, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 80% 70%, rgba(100, 255, 218, 0.05) 0%, transparent 40%) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.glass-card {
    background: var(--card-bg) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
}

#header-area {
    margin-bottom: 2rem;
    padding: 1.5rem;
    border-bottom: 1px solid var(--border-color);
}

.status-badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.status-online { background: rgba(100, 255, 218, 0.15); color: var(--success-color); }
.status-offline { background: rgba(255, 77, 77, 0.15); color: var(--error-color); }

/* Custom Chat Components */
.gradio-container .chat-interface, .gradio-container .chatbot {
    background: transparent !important;
    border: none !important;
}

.message {
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin-bottom: 12px !important;
    line-height: 1.5 !important;
}

.message.user {
    background: var(--accent-glow) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    box-shadow: 0 4px 15px rgba(0, 210, 255, 0.1) !important;
}

.message.bot {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: var(--text-primary) !important;
}

/* Fix for white bubble area in Gradio 6 */
.chatbot > .wrapper { background-color: transparent !important; }
.chatbot .message-row { background: transparent !important; }
.chatbot .message-wrap { background: transparent !important; }

/* Input area styling */
#component-7 {  /* The textbox container */
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
}

#component-7 textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
}

/* Tabs Styling */
.tabs {
    border-bottom: 1px solid var(--border-color) !important;
    background: transparent !important;
}

.tab-nav {
    border-bottom: 1px solid var(--border-color) !important;
    gap: 1.5rem !important;
    padding: 0 1rem !important;
}

.tab-nav button {
    background: transparent !important;
    border: none !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    padding: 1rem 0.5rem !important;
    transition: all 0.3s ease !important;
}

.tab-nav button.selected {
    color: var(--accent-color) !important;
    border-bottom: 2px solid var(--accent-color) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg-color); }
::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-color); }

#health-bar {
    background: rgba(0, 0, 0, 0.2);
    padding: 10px 20px;
    border-radius: 30px;
    margin-top: 10px;
    border: 1px solid var(--border-color);
}

.quick-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px var(--accent-glow);
}

.universe-badge {
    cursor: pointer;
    transition: all 0.3s ease;
}

.universe-badge:hover {
    background: var(--accent-glow) !important;
    border-color: var(--accent-color) !important;
    transform: scale(1.05);
}
"""


UNIVERSE_JS = """
function() {
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('universe-badge')) {
            const universe = e.target.getAttribute('data-universe');
            const query = `universe overview ${universe}`;
            const textbox = document.querySelector('#chat-input textarea');
            if (textbox) {
                // Set the value
                const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                valueSetter.call(textbox, query);
                
                // Trigger events so Gradio picks up the change
                textbox.dispatchEvent(new Event('input', { bubbles: true }));
                textbox.dispatchEvent(new Event('change', { bubbles: true }));
                
                // Trigger the 'Enter' key to submit
                const enterEvent = new KeyboardEvent('keydown', {
                    bubbles: true,
                    cancelable: true,
                    key: 'Enter',
                    code: 'Enter',
                    keyCode: 13
                });
                textbox.dispatchEvent(enterEvent);
                
                // Switch to the first tab (Advisory Chat)
                const tabs = document.querySelectorAll('.tab-nav button');
                if (tabs.length > 0) {
                    tabs[0].click();
                }
            }
        }
    });
}
"""


def get_persistent_session_id() -> str:
    if os.path.exists(SESSION_ID_FILE):
        try:
            with open(SESSION_ID_FILE, "r") as f:
                sid = f.read().strip()
                if sid:
                    return sid
        except Exception:
            pass
    
    new_id = str(uuid.uuid4())
    try:
        with open(SESSION_ID_FILE, "w") as f:
            f.write(new_id)
    except Exception:
        pass
    return new_id


SESSION_ID = get_persistent_session_id()


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

    # Robust URL rewriting for various markdown formats
    content = re.sub(r"\]\(\s*(outputs/[^)\s]+)\s*\)", replace_relative, content)
    content = re.sub(r"\]\(\s*(/outputs/[^)\s]+)\s*\)", replace_rooted, content)
    content = re.sub(
        r"\]\(\s*(?:[A-Za-z]:)?[^)\s]*[\\/]+outputs[\\/]+([^\\/)\s]+)\s*\)",
        replace_absolute_outputs,
        content,
    )
    return content


def check_api_health():
    """Poll the backend health endpoint and return HTML status summary."""
    try:
        response = requests.get(HEALTH_URL, timeout=3)
        if response.status_code == 200:
            data = response.json()
            comp = data.get("components", {})
            models = data.get("models", {})
            
            mongo_icon = "🟢" if comp.get("mongodb") == "connected" else "🔴"
            ollama_icon = "🟢" if comp.get("ollama") == "ready" else "🔴"
            
            status_html = f"""
            <div style="display: flex; gap: 20px; font-size: 0.9rem;">
                <span><b>System:</b> <span class="status-badge status-online">ONLINE</span></span>
                <span><b>MongoDB:</b> {mongo_icon} {comp.get('mongodb', 'unknown')}</span>
                <span><b>Model:</b> {ollama_icon} {models.get('primary', 'unknown')}</span>
                <span><b>Memory:</b> {len(models.get('available', []))} models cached</span>
            </div>
            """
            return status_html
        return f'<span><b>System:</b> <span class="status-badge status-offline">API ERROR ({response.status_code})</span></span>'
    except Exception:
        return '<span><b>System:</b> <span class="status-badge status-offline">OFFLINE</span></span>'


def chat_with_api(user_message: str, history, session_id: str):
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
                    yield _rewrite_plot_markdown(f"*{status_message}*")
            elif event_type == "token":
                accumulated_response += content
                yield _rewrite_plot_markdown(accumulated_response)
            elif event_type == "final":
                accumulated_response = content or accumulated_response
                yield _rewrite_plot_markdown(accumulated_response)
            elif event_type == "error":
                yield _rewrite_plot_markdown(f"❌ **System Error**: {content}")
                return

        if not accumulated_response and status_message:
            yield _rewrite_plot_markdown(status_message)
        elif not accumulated_response:
            yield "No response returned from backend."
    except requests.exceptions.ConnectionError:
        yield "❌ **Backend API unreachable**. Please ensure the FastAPI server is running."
    except Exception as exc:
        yield f"❌ **System Error**: {exc}"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Portfolio Governance") as demo:
        session_state = gr.State(SESSION_ID)

        with gr.Column(elem_id="header-area", elem_classes="glass-card"):
            gr.Markdown("# 🛡️ Agentic Portfolio Governance")
            gr.Markdown("### Advanced Research Prototype • Advisory-Only System")
            
            health_display = gr.HTML(check_api_health, every=5)

        with gr.Tabs(elem_classes="glass-card"):
            with gr.Tab("💬 Advisory Chat"):
                gr.ChatInterface(
                    fn=chat_with_api,
                    additional_inputs=[session_state],
                    textbox=gr.Textbox(
                        placeholder="Ask about historical risk, sectors, or CVaR optimization...",
                        container=False,
                        scale=7,
                        elem_id="chat-input"
                    ),
                    examples=[
                        ["Analyze U1 for 2008-09-15"],
                        ["Show all available sectors"],
                        ["Plot AAPL vs MSFT from 2020 to 2023"],
                        ["What institutions connect GS and JPM?"],
                        ["Run historical CVaR for U10 on 2022-01-01"]
                    ],
                )

            with gr.Tab("🔍 Universe Explorer"):
                gr.Markdown("### Curated Historical Universes (U1 - U11)")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        **Sector Breakdown**
                        - **U1-U3**: Financials & Tech
                        - **U4-U6**: Consumer & Industrials
                        - **U7-U9**: Energy & Materials
                        - **U10-U11**: Defensive & Utilities
                        """)
                    with gr.Column(scale=2):
                        gr.Markdown("""
                        **Lookup Commands**
                        - `list universes`: See the full roster.
                        - `universe overview U1`: Get sector composition.
                        - `stocks in U1`: List every ticker in the set.
                        """)
                
                gr.HTML("""
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 15px; margin-top: 20px;">
                    <div class="status-badge universe-badge" data-universe="U1" style="background: rgba(0, 210, 255, 0.1); border: 1px solid var(--border-color); justify-content: center;">U1: Finance</div>
                    <div class="status-badge universe-badge" data-universe="U2" style="background: rgba(0, 210, 255, 0.1); border: 1px solid var(--border-color); justify-content: center;">U2: Tech</div>
                    <div class="status-badge universe-badge" data-universe="U3" style="background: rgba(0, 210, 255, 0.1); border: 1px solid var(--border-color); justify-content: center;">U3: Healthcare</div>
                    <div class="status-badge universe-badge" data-universe="U4" style="background: rgba(0, 210, 255, 0.1); border: 1px solid var(--border-color); justify-content: center;">U4: Energy</div>
                    <div class="status-badge universe-badge" data-universe="U5" style="background: rgba(0, 210, 255, 0.1); border: 1px solid var(--border-color); justify-content: center;">U5: Retail</div>
                    <div class="status-badge universe-badge" data-universe="U6" style="background: rgba(0, 210, 255, 0.1); border: 1px solid var(--border-color); justify-content: center;">U6: Industrial</div>
                    <div class="status-badge universe-badge" data-universe="U7" style="background: rgba(0, 210, 255, 0.1); border: 1px solid var(--border-color); justify-content: center;">U7: Materials</div>
                    <div class="status-badge universe-badge" data-universe="U8" style="background: rgba(0, 210, 255, 0.1); border: 1px solid var(--border-color); justify-content: center;">U8: Consumer</div>
                    <div class="status-badge universe-badge" data-universe="U9" style="background: rgba(0, 210, 255, 0.1); border: 1px solid var(--border-color); justify-content: center;">U9: Utilities</div>
                    <div class="status-badge universe-badge" data-universe="U10" style="background: rgba(0, 210, 255, 0.1); border: 1px solid var(--border-color); justify-content: center;">U10: Real Estate</div>
                    <div class="status-badge universe-badge" data-universe="U11" style="background: rgba(0, 210, 255, 0.1); border: 1px solid var(--border-color); justify-content: center;">U11: Tech Focus</div>
                </div>
                """)

            with gr.Tab("🧠 Persistent Memory"):
                gr.Markdown("### Cross-Session & Historical Intelligence")
                gr.Markdown("""
                This system utilizes a **MongoDB** persistence layer to maintain context across your research sessions.
                - **Session Persistence**: Your thread (ID: `{}`) is saved automatically.
                - **Semantic Cache**: Governance runs are cached for 7 days to reduce computation.
                - **Global Activity**: The assistant monitors collective regime patterns in the last 24h.
                """.format(SESSION_ID))
                
                with gr.Accordion("System Architecture Details", open=False):
                    gr.Markdown("""
                    - **Orchestration**: LangGraph DAG
                    - **Memory**: MongoDBSaver + Custom TTL L2 Cache
                    - **RAG**: Graph-Context Traversal + Methodology PDF Vector Search
                    - **Optimization**: CVXPY (CLARABEL/OSQP)
                    """)

        gr.Markdown(
            """
            <div style="text-align: center; color: var(--text-secondary); margin-top: 2rem; font-size: 0.8rem;">
                © 2026 Agentic AI Portfolio Governance • Advisory-Only Research Platform
            </div>
            """
        )

        # Inject global JS for universe-badge interactivity
        demo.load(None, None, None, js=UNIVERSE_JS)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(theme=gr.themes.Soft(), css=PREMIUM_CSS, share=False)
