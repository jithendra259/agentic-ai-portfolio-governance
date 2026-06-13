---
name: AshnaAI backend config
description: Critical config facts for AshnaAI LLM integration and backend Python environment
---

## AshnaAI URL

- `ASHNA_BASE_URL=https://api.ashna.ai/v1/api` in `backend/.env`
- The correct completions endpoint is `https://api.ashna.ai/v1/api/chat/completions`
- Do NOT strip `/api` from the base_url — old normalization code (`if base_url.endswith("/api"): base_url = base_url[:-4]`) was wrong and caused 404s
- LangChain's ChatOpenAI appends `/chat/completions` to `base_url` directly

**Why:** The URL normalization logic assumed AshnaAI used a standard OpenAI-style `/v1` base, but AshnaAI's endpoint path includes `/api` after `/v1`.

## streaming=False fix

- `ChatOpenAI(streaming=True).bind_tools(tools).invoke()` returns `AIMessage(content='')` for AshnaAI text responses
- Fix: set `"streaming": False, "timeout": 60` in `_get_chat_llm` for AshnaAI (chatbot_orchestrator.py)
- The `on_chain_end` LangGraph fallback in main.py (event_name=="LangGraph") reads messages[-1] and streams it — correct path for non-streaming AshnaAI

## Python environment

- Packages are installed in `.pythonlibs/lib/python3.12/site-packages`
- `.pythonlibs/bin/python3` and `.pythonlibs/bin/python3.11` are Python 3.11 and CANNOT find the packages
- `.pythonlibs/bin/python3.12` is the correct binary to use
- `backend/start.sh` must use: `exec /home/runner/workspace/.pythonlibs/bin/python3.12 -m uvicorn api.main:app --host 127.0.0.1 --port 8000`

**Why:** Replit's pythonlibs defaulted to installing packages for 3.12 but the symlink `python3` points to 3.11.
