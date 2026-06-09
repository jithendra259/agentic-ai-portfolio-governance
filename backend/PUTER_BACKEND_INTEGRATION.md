Puter Backend Integration

This repository includes an optional backend adapter for Puter. Use it when you want the server to call Puter REST endpoints instead of making browser-only Puter.js calls.

Files
- `backend/src/providers/puter_provider.py` — main adapter. Exposes `call_puter_chat(prompt, model, **kwargs)`.
- `backend/test_puter_provider.py` — quick CLI test script.

Configuration
Set the following environment variables in `backend/.env` or your environment:

- `PUTER_BASE_URL` (required): base URL for Puter REST API, e.g. `https://api.puter.com` or particular region
- `PUTER_API_KEY` (optional): bearer key if the API requires authentication

How it works
- `call_puter_chat` sends POST to `{PUTER_BASE_URL}/v1/chat/completions`
- The function attempts to parse OpenAI-style chat completions (`choices[0].message.content`) and returns a dict `{"text": <extracted>, "raw": <full JSON>}`
- If the endpoint returns non-2xx or non-JSON responses, a `PuterError` is raised with diagnostics

Notes
- Puter is primarily designed for client-side use with Puter.js. Server-side endpoints and billing model may differ — check Puter docs and terms before using in production.
- If `PUTER_BASE_URL` is not set, the adapter will raise a `PuterError`.

Integration with orchestrator
To use Puter as a fallback or a provider in the orchestrator, call `call_puter_chat` when constructing responses or inside `_get_chat_llm` fallback path. Example:

```py
from src.providers.puter_provider import call_puter_chat, PuterError

try:
    res = call_puter_chat(prompt, model="qwen/qwen3.7-plus")
    text = res.get('text') or str(res.get('raw'))
except PuterError as e:
    # fallback to Ollama or other model
    pass
```

Security
- Use server-side adapter only if you understand Puter's billing and data policies
- Keep `PUTER_API_KEY` secret

Running tests
```bash
cd backend
python test_puter_provider.py
```

If the call fails with a 404 HTML page (Vercel/Next), the configured base URL is likely incorrect or the service is not exposing an OpenAI-compatible route. In that case prefer client-side Puter.js usage or contact Puter support.
