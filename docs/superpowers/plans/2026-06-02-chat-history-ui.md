# Chat History UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add usable new-chat and older-conversation navigation to the Portfolio Assistant while preserving the working Supabase-backed memory, streaming, and inline chart behavior.

**Architecture:** Add a small session-summary read API on top of the existing `chat_messages` persistence layer. Keep MUI X Chat as the message engine and wrap it with a custom React shell for sidebar/history controls and layout polish.

**Tech Stack:** Python FastAPI, `unittest`, MongoDB/Supabase Postgres memory layer, React 19, Vite, MUI, MUI X Chat, lucide-react.

---

## File Structure

- Modify `backend/src/memory/mongodb_memory_layer.py`: add title normalization and `list_chat_sessions(limit=50)` with Supabase and MongoDB implementations.
- Modify `backend/test/test_chat_memory_supabase.py`: add failing tests for Supabase session summaries and title fallback.
- Modify `backend/api/main.py`: add response models and `GET /chat/sessions`.
- Modify `frontend/src/components/ChatInterface.jsx`: replace single-session-only state with active-session state, session list fetching, new-chat action, refresh action, and shell layout around `ChatBox`.
- Modify `frontend/src/index.css`: add global chat layout safety styles for markdown, long content, chart containers, and mobile sizing.

---

### Task 1: Backend Session Summary Memory API

**Files:**
- Modify: `backend/test/test_chat_memory_supabase.py`
- Modify: `backend/src/memory/mongodb_memory_layer.py`

- [ ] **Step 1: Write the failing Supabase session listing test**

Append these tests to `SupabaseChatMemoryTests` in `backend/test/test_chat_memory_supabase.py`:

```python
    def test_list_chat_sessions_returns_recent_session_summaries(self):
        first_created = datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)
        second_created = datetime(2026, 6, 2, 10, 30, tzinfo=timezone.utc)
        cursor = FakeCursor(
            rows=[
                ("session-2", "Compare AAPL and MSFT performance over a long horizon", 4, second_created, second_created),
                ("session-1", "Plot TSLA", 2, first_created, first_created),
            ]
        )
        manager = MongoMemoryManager(mongo_uri="", postgres_url="")
        manager.pg_pool = FakePool(cursor)

        sessions = manager.list_chat_sessions(limit=20)

        sql, params = cursor.statements[-1]
        self.assertIn("GROUP BY session_id", sql)
        self.assertEqual(params[0], 20)
        self.assertEqual(
            sessions,
            [
                {
                    "session_id": "session-2",
                    "title": "Compare AAPL and MSFT performance over a long horizon",
                    "message_count": 4,
                    "created_at": "2026-06-02T10:30:00+00:00",
                    "updated_at": "2026-06-02T10:30:00+00:00",
                },
                {
                    "session_id": "session-1",
                    "title": "Plot TSLA",
                    "message_count": 2,
                    "created_at": "2026-06-01T09:00:00+00:00",
                    "updated_at": "2026-06-01T09:00:00+00:00",
                },
            ],
        )

    def test_list_chat_sessions_uses_new_chat_for_missing_user_title(self):
        created_at = datetime(2026, 6, 2, tzinfo=timezone.utc)
        cursor = FakeCursor(rows=[("session-empty", None, 1, created_at, created_at)])
        manager = MongoMemoryManager(mongo_uri="", postgres_url="")
        manager.pg_pool = FakePool(cursor)

        sessions = manager.list_chat_sessions()

        self.assertEqual(sessions[0]["title"], "New chat")
```

- [ ] **Step 2: Run the backend test and verify it fails**

Run:

```powershell
python -m unittest backend.test.test_chat_memory_supabase -v
```

Expected: FAIL with `AttributeError: 'MongoMemoryManager' object has no attribute 'list_chat_sessions'`.

- [ ] **Step 3: Implement title normalization and Supabase session listing**

In `backend/src/memory/mongodb_memory_layer.py`, add this helper near `append_chat_message`:

```python
    def _format_chat_session_title(self, title: Any) -> str:
        normalized = " ".join(str(title or "").split())
        if not normalized:
            return "New chat"
        if len(normalized) <= 64:
            return normalized
        return f"{normalized[:61].rstrip()}..."
```

Then add this method after `list_chat_messages`:

```python
    def list_chat_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return persisted chat sessions sorted by most recent activity."""
        safe_limit = max(1, min(int(limit or 50), 100))

        if self.pg_pool:
            try:
                with self.pg_pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            WITH ranked_messages AS (
                                SELECT
                                    session_id,
                                    role,
                                    content,
                                    created_at,
                                    ROW_NUMBER() OVER (
                                        PARTITION BY session_id
                                        ORDER BY CASE WHEN role = 'user' THEN 0 ELSE 1 END, created_at ASC, id ASC
                                    ) AS title_rank
                                FROM chat_messages
                            )
                            SELECT
                                session_id,
                                MAX(CASE WHEN title_rank = 1 AND role = 'user' THEN content ELSE NULL END) AS title,
                                COUNT(*) AS message_count,
                                MIN(created_at) AS created_at,
                                MAX(created_at) AS updated_at
                            FROM ranked_messages
                            GROUP BY session_id
                            ORDER BY updated_at DESC
                            LIMIT %s;
                            """,
                            (safe_limit,),
                        )
                        rows = cur.fetchall()

                return [
                    {
                        "session_id": str(row[0]),
                        "title": self._format_chat_session_title(row[1]),
                        "message_count": int(row[2] or 0),
                        "created_at": row[3].isoformat() if hasattr(row[3], "isoformat") else str(row[3] or ""),
                        "updated_at": row[4].isoformat() if hasattr(row[4], "isoformat") else str(row[4] or ""),
                    }
                    for row in rows
                ]
            except Exception as exc:
                logger.warning("Postgres list_chat_sessions failed: %s. Falling back to Mongo...", exc)

        if not self.is_available:
            return []

        try:
            chat_col = self._collection("chat_messages")
            if chat_col is None:
                return []
            pipeline = [
                {"$sort": {"session_id": 1, "created_at": 1}},
                {
                    "$group": {
                        "_id": "$session_id",
                        "first_user": {
                            "$first": {
                                "$cond": [{"$eq": ["$role", "user"]}, "$content", None]
                            }
                        },
                        "message_count": {"$sum": 1},
                        "created_at": {"$min": "$created_at"},
                        "updated_at": {"$max": "$created_at"},
                    }
                },
                {"$sort": {"updated_at": -1}},
                {"$limit": safe_limit},
            ]
            sessions = []
            for row in chat_col.aggregate(pipeline):
                created_at = row.get("created_at")
                updated_at = row.get("updated_at")
                sessions.append(
                    {
                        "session_id": str(row.get("_id")),
                        "title": self._format_chat_session_title(row.get("first_user")),
                        "message_count": int(row.get("message_count") or 0),
                        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or ""),
                        "updated_at": updated_at.isoformat() if hasattr(updated_at, "isoformat") else str(updated_at or ""),
                    }
                )
            return sessions
        except PyMongoError as exc:
            logger.warning("Failed to list chat sessions: %s", exc)
            return []
```

- [ ] **Step 4: Run the backend test and verify it passes**

Run:

```powershell
python -m unittest backend.test.test_chat_memory_supabase -v
```

Expected: PASS for all tests in `SupabaseChatMemoryTests`.

---

### Task 2: FastAPI Session List Endpoint

**Files:**
- Modify: `backend/api/main.py`

- [ ] **Step 1: Write the failing API surface check**

Create `backend/test/test_chat_sessions_api.py`:

```python
import os
import sys
import unittest

from fastapi.testclient import TestClient


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app


class FakeMemoryManager:
    pg_pool = object()

    def list_chat_sessions(self, limit=50):
        return [
            {
                "session_id": "session-1",
                "title": "Plot AAPL",
                "message_count": 2,
                "created_at": "2026-06-02T00:00:00+00:00",
                "updated_at": "2026-06-02T00:05:00+00:00",
            }
        ]


class ChatSessionsApiTests(unittest.TestCase):
    def test_chat_sessions_endpoint_returns_summaries(self):
        import api.main as main

        original_memory_manager = main.memory_manager
        main.memory_manager = FakeMemoryManager()
        try:
            client = TestClient(app)
            response = client.get("/chat/sessions?limit=25")
        finally:
            main.memory_manager = original_memory_manager

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "sessions": [
                    {
                        "session_id": "session-1",
                        "title": "Plot AAPL",
                        "message_count": 2,
                        "created_at": "2026-06-02T00:00:00+00:00",
                        "updated_at": "2026-06-02T00:05:00+00:00",
                    }
                ]
            },
        )
```

- [ ] **Step 2: Run the API test and verify it fails**

Run:

```powershell
python -m unittest backend.test.test_chat_sessions_api -v
```

Expected: FAIL with HTTP 404 for `/chat/sessions`.

- [ ] **Step 3: Implement response models and endpoint**

In `backend/api/main.py`, add these models after `ChatHistoryResponse`:

```python
class ChatSessionResponse(BaseModel):
    session_id: str
    title: str
    message_count: int
    created_at: str
    updated_at: str


class ChatSessionsResponse(BaseModel):
    sessions: list[ChatSessionResponse]
```

Add this route above `@app.get("/chat/{session_id}/messages", ...)` so the literal `/chat/sessions` path is matched before the parameterized route:

```python
@app.get("/chat/sessions", response_model=ChatSessionsResponse)
def chat_sessions(limit: int = 50) -> ChatSessionsResponse:
    safe_limit = max(1, min(int(limit or 50), 100))
    sessions = memory_manager.list_chat_sessions(limit=safe_limit)
    return ChatSessionsResponse(sessions=sessions)
```

- [ ] **Step 4: Run backend API and memory tests**

Run:

```powershell
python -m unittest backend.test.test_chat_sessions_api backend.test.test_chat_memory_supabase -v
```

Expected: PASS.

---

### Task 3: Frontend Chat Shell and Session Controls

**Files:**
- Modify: `frontend/src/components/ChatInterface.jsx`
- Modify: `frontend/src/index.css`

- [ ] **Step 1: Run current frontend build as the baseline**

Run:

```powershell
pnpm --dir frontend build
```

Expected: PASS before editing the frontend. If it fails, record the existing failure before changing code.

- [ ] **Step 2: Change `sessionId` state to active session state**

In `ChatInterface.jsx`, replace the read-only session state:

```jsx
  const [sessionId] = useState(() => {
```

with:

```jsx
  const [sessionId, setSessionId] = useState(() => {
```

Add helper functions above `export default function ChatInterface()`:

```jsx
function createSessionId() {
  const randomPart = window.crypto?.randomUUID?.() || Math.random().toString(36).slice(2);
  return `portfolio-chat-${randomPart}`;
}

function formatSessionDate(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}
```

Update the initializer to use `createSessionId()`:

```jsx
    const nextSessionId = createSessionId();
```

- [ ] **Step 3: Add session list state and loaders**

Inside `ChatInterface`, add:

```jsx
  const [chatSessions, setChatSessions] = useState([]);
  const [sessionsLoaded, setSessionsLoaded] = useState(false);
  const [sessionsError, setSessionsError] = useState('');
```

Add this function after the model loading effect:

```jsx
  const refreshSessions = useCallback(() => {
    setSessionsError('');
    return fetch(`${BACKEND_BASE}/chat/sessions?limit=50`)
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch chat sessions');
        return res.json();
      })
      .then((data) => {
        setChatSessions(data?.sessions || []);
        setSessionsLoaded(true);
      })
      .catch((err) => {
        console.error('Failed to load chat sessions:', err);
        setChatSessions([]);
        setSessionsError('Could not load older chats');
        setSessionsLoaded(true);
      });
  }, []);
```

Update the React import to include `useCallback`.

Add:

```jsx
  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);
```

- [ ] **Step 4: Add new/open chat handlers**

Add below `initialMessages`:

```jsx
  const activeSession = chatSessions.find((item) => item.session_id === sessionId);
  const activeTitle = activeSession?.title || 'Portfolio Assistant';

  const handleNewChat = useCallback(() => {
    const nextSessionId = createSessionId();
    window.localStorage.setItem(SESSION_STORAGE_KEY, nextSessionId);
    setSessionId(nextSessionId);
    setPersistedMessages([]);
    setHistoryLoaded(true);
  }, []);

  const handleOpenSession = useCallback((nextSessionId) => {
    if (!nextSessionId || nextSessionId === sessionId) return;
    window.localStorage.setItem(SESSION_STORAGE_KEY, nextSessionId);
    setSessionId(nextSessionId);
  }, [sessionId]);
```

In `adapter.sendMessage`, after `const ndjsonStream = await parseNDJSONStream(response, signal);`, trigger a refresh after the stream settles:

```jsx
      const refreshAfterStream = new TransformStream({
        flush() {
          refreshSessions();
        },
      });
```

Then return:

```jsx
      return ndjsonStream.pipeThrough(transformStream).pipeThrough(refreshAfterStream);
```

Update the adapter dependency array from `[sessionId]` to `[refreshSessions, sessionId]`.

- [ ] **Step 5: Wrap `ChatBox` in the shell layout**

Replace the top-level `return` wrapper with a shell containing:

```jsx
    <Box className="chat-app-shell">
      <Box className="chat-sidebar">
        <Box className="chat-brand">
          <Box className="chat-brand-icon"><Bot size={22} /></Box>
          <Box>
            <Typography className="chat-brand-title">Portfolio Assistant</Typography>
            <Typography className="chat-brand-subtitle">Supabase memory</Typography>
          </Box>
        </Box>
        <Box className="chat-sidebar-actions">
          <button className="chat-action-button primary" type="button" onClick={handleNewChat}>
            New Chat
          </button>
          <button className="chat-action-button" type="button" onClick={refreshSessions}>
            Refresh
          </button>
        </Box>
        <Box className="chat-history-list">
          {!sessionsLoaded ? (
            <Typography className="chat-history-empty">Loading chats...</Typography>
          ) : sessionsError ? (
            <Typography className="chat-history-empty">{sessionsError}</Typography>
          ) : chatSessions.length === 0 ? (
            <Typography className="chat-history-empty">No older chats yet</Typography>
          ) : (
            chatSessions.map((item) => (
              <button
                key={item.session_id}
                className={`chat-history-item ${item.session_id === sessionId ? 'active' : ''}`}
                type="button"
                onClick={() => handleOpenSession(item.session_id)}
                title={item.title}
              >
                <span className="chat-history-title">{item.title}</span>
                <span className="chat-history-meta">
                  {item.message_count} messages · {formatSessionDate(item.updated_at)}
                </span>
              </button>
            ))
          )}
        </Box>
      </Box>
      <Box className="chat-main">
        <Box className="chat-topbar">
          <Box>
            <Typography className="chat-topbar-title">{activeTitle}</Typography>
            <Typography className="chat-topbar-subtitle">
              {historyLoaded ? 'Memory loaded' : 'Loading memory...'}
            </Typography>
          </Box>
        </Box>
        <Box className="chatbox-wrap">
          <ChatBox ... />
        </Box>
      </Box>
    </Box>
```

Keep all existing `ChatBox` props. Change the `ChatBox` `sx` to:

```jsx
        sx={{
          flex: 1,
          minHeight: 0,
          height: '100%',
          border: 'none',
          borderRadius: 0,
          backgroundColor: '#0D0D0D',
          color: '#ECECEC',
        }}
```

- [ ] **Step 6: Add layout CSS**

Append to `frontend/src/index.css`:

```css
.chat-app-shell {
  display: grid;
  grid-template-columns: 304px minmax(0, 1fr);
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  background: #0d0d0d;
  color: #ececec;
}

.chat-sidebar {
  min-width: 0;
  border-right: 1px solid #323232;
  background: #171717;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.chat-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 18px 18px 14px;
  border-bottom: 1px solid #2d2d2d;
}

.chat-brand-icon {
  width: 38px;
  height: 38px;
  border: 1px solid #484848;
  border-radius: 8px;
  display: grid;
  place-items: center;
  color: #f3f3f3;
  background: #242424;
  flex: 0 0 auto;
}

.chat-brand-title,
.chat-topbar-title {
  color: #f5f5f5 !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  line-height: 1.25 !important;
}

.chat-brand-subtitle,
.chat-topbar-subtitle,
.chat-history-meta,
.chat-history-empty {
  color: #a9a9a9 !important;
  font-size: 0.78rem !important;
  line-height: 1.35 !important;
}

.chat-sidebar-actions {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 8px;
  padding: 12px;
}

.chat-action-button,
.chat-history-item {
  font: inherit;
  border: 1px solid #373737;
  color: #e7e7e7;
  background: #202020;
  cursor: pointer;
}

.chat-action-button {
  min-height: 36px;
  border-radius: 8px;
  padding: 0 12px;
  font-weight: 650;
}

.chat-action-button.primary {
  background: #f1f1f1;
  color: #111;
  border-color: #f1f1f1;
}

.chat-action-button:hover,
.chat-history-item:hover {
  background: #2a2a2a;
}

.chat-action-button.primary:hover {
  background: #fff;
}

.chat-history-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 0 10px 12px;
  overflow-y: auto;
  min-height: 0;
}

.chat-history-item {
  width: 100%;
  text-align: left;
  border-radius: 8px;
  padding: 10px 11px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.chat-history-item.active {
  background: #303030;
  border-color: #5d5d5d;
}

.chat-history-title {
  color: #eeeeee;
  font-weight: 620;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.chat-main {
  min-width: 0;
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

.chat-topbar {
  min-height: 70px;
  border-bottom: 1px solid #303030;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  background: #121212;
  flex: 0 0 auto;
}

.chatbox-wrap {
  min-height: 0;
  flex: 1;
  display: flex;
  overflow: hidden;
}

.chatbox-wrap * {
  min-width: 0;
}

.chatbox-wrap p,
.chatbox-wrap li {
  overflow-wrap: anywhere;
}

.chatbox-wrap pre,
.markdown-table-container {
  max-width: 100%;
  overflow-x: auto;
}

@media (max-width: 780px) {
  .chat-app-shell {
    grid-template-columns: 1fr;
    grid-template-rows: auto minmax(0, 1fr);
  }

  .chat-sidebar {
    border-right: none;
    border-bottom: 1px solid #323232;
    max-height: 190px;
  }

  .chat-brand {
    padding: 12px 14px 8px;
  }

  .chat-sidebar-actions {
    padding: 8px 10px;
  }

  .chat-history-list {
    flex-direction: row;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 0 10px 10px;
  }

  .chat-history-item {
    min-width: 210px;
  }

  .chat-main {
    height: auto;
    min-height: 0;
  }

  .chat-topbar {
    min-height: 56px;
    padding: 0 16px;
  }
}
```

- [ ] **Step 7: Run frontend build**

Run:

```powershell
pnpm --dir frontend build
```

Expected: PASS.

---

### Task 4: Final Verification

**Files:**
- Verify only.

- [ ] **Step 1: Run backend tests**

Run:

```powershell
python -m unittest backend.test.test_chat_sessions_api backend.test.test_chat_memory_supabase -v
```

Expected: PASS.

- [ ] **Step 2: Run frontend build**

Run:

```powershell
pnpm --dir frontend build
```

Expected: PASS.

- [ ] **Step 3: Start backend server**

Run:

```powershell
Start-Process -WindowStyle Hidden -FilePath python -ArgumentList "-m", "uvicorn", "backend.api.main:app", "--host", "127.0.0.1", "--port", "8000"
```

Expected: backend serves `http://127.0.0.1:8000/health`.

- [ ] **Step 4: Start frontend dev server**

Run:

```powershell
Start-Process -WindowStyle Hidden -WorkingDirectory frontend -FilePath pnpm -ArgumentList "dev", "--host", "127.0.0.1"
```

Expected: Vite prints a localhost URL, usually `http://127.0.0.1:5173/`.

- [ ] **Step 5: Manual browser verification**

Open the Vite URL and verify:

- The sidebar shows New Chat, Refresh, and older conversations if `/chat/sessions` returns data.
- Clicking New Chat changes the active conversation and shows the welcome state.
- Clicking an older conversation loads its previous messages.
- A long response or chart response does not hide behind the composer and does not overflow off the right edge.

---

## Self-Review

- Spec coverage: backend session summaries are covered by Tasks 1 and 2; frontend shell/new/open/refresh actions and layout fixes are covered by Task 3; verification is covered by Task 4.
- Placeholder scan: no task uses TBD/TODO or asks for unspecified tests.
- Type consistency: backend response fields are `session_id`, `title`, `message_count`, `created_at`, `updated_at`; frontend uses the same property names.
