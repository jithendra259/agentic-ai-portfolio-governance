Puter integration (frontend)

This project includes a lightweight Puter integration that lets the browser call Puter.js (Puter's client) directly so users can use Qwen models without server-side keys.

Files added
- `frontend/src/lib/puterModels.js` - exported `PUTER_MODELS` array containing Qwen model ids to show in UI
- `frontend/src/lib/puterClient.js` - small runtime loader for the Puter CDN script and helper functions `initPuter`, `chat`, `txt2img`

How to use
1. In a browser-only chat UI (React component), import the client:

```js
import { chat } from '../lib/puterClient';
import { PUTER_MODELS } from '../lib/puterModels';

// call
const response = await chat('Say hello', { model: PUTER_MODELS[0] });
console.log(response);
```

2. The Puter client is loaded from the CDN so no `npm install` is required. If you prefer bundling, install `@heyputer/puter.js` and import it instead.

Notes & security
- Puter uses a "user-pays" model and is designed to run client-side — no server API key is required.
- Do not call Puter from server-side code unless you confirm their server-side API and usage policy.
- This integration is intended as a frontend provider/fallback. If you want a server-side provider, I can research Puter REST endpoints and add a backend adapter.

Next steps (optional)
- Add a provider selector UI component to let users pick between `ashnaai`, `ollama`, `puter`.
- Add usage and cost warnings in the UI when Puter is selected.
- Add a small example Chat component demonstrating streaming and code formatting.
