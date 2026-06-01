# Chat History UI Design

## Goal

Improve the Portfolio Assistant chat UI now that Supabase-backed chat memory is working. The user should be able to start a fresh conversation, return to older Supabase conversations, and use the chart/chat experience without message content being clipped or hidden by the composer.

## Scope

This design covers:

- A backend endpoint for listing persisted chat sessions.
- A frontend shell around the existing MUI X Chat component.
- New chat, refresh history, and open old chat actions.
- Layout fixes for the sidebar, header, message area, chart bubbles, and composer.

This design does not include rename or delete conversation actions. Those require a separate backend data-removal contract and should be added after the basic thread navigation is stable.

## Backend Design

Add `MongoMemoryManager.list_chat_sessions(limit=50)`.

For Supabase/Postgres, query `chat_messages` grouped by `session_id` and return:

- `session_id`
- `title`
- `message_count`
- `created_at`
- `updated_at`

The title should be derived from the earliest user message in the session. It should be trimmed, whitespace-normalized, and shortened for display. If no user message exists, use `New chat`.

For MongoDB fallback, aggregate the `chat_messages` collection with the same shape. If neither Supabase nor MongoDB is available, return an empty list.

Add `GET /chat/sessions?limit=50` in `backend/api/main.py`. The endpoint validates the limit and returns the session summaries sorted by most recently updated first.

## Frontend Design

Keep `ChatBox` as the message and streaming engine. Wrap it in a custom shell implemented in `frontend/src/components/ChatInterface.jsx`.

The shell should include:

- A fixed-width desktop sidebar with the assistant identity, a new chat button, a refresh button, and recent conversation rows.
- An active conversation highlight.
- A responsive mobile layout where the sidebar collapses into a compact horizontal/top area or hides without blocking chat use.
- A top header for the active chat title and status.
- A main chat area where messages have a readable max width and long assistant/chart content can wrap or scroll horizontally where needed.

Session state should work as follows:

- On load, read the active session id from localStorage.
- Fetch `/chat/sessions` and `/chat/{session_id}/messages`.
- Clicking an older chat sets it as active, stores it in localStorage, and reloads messages.
- New chat creates a fresh `portfolio-chat-<uuid>` id, stores it as active, clears persisted messages, and refreshes the session list after the first successful response.

## Error Handling

If `/chat/sessions` fails, the UI should still load the active local session and show a non-blocking empty/error state in the history list.

If loading messages for a session fails, show the normal welcome message and log the error. Chat sending should remain available.

If the user creates a new chat while a response is streaming, the current request should be allowed to abort through the existing `ChatBox` signal behavior.

## Testing

Backend tests should cover:

- Supabase session listing returns grouped session summaries.
- Titles are derived from the first user message.
- Empty or unavailable stores return an empty list.

Frontend verification should cover:

- Production build succeeds.
- New chat changes the active session id.
- Older chat selection reloads persisted messages.
- Long assistant responses and inline chart output are not clipped by the viewport or composer.
