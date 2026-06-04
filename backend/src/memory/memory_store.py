from __future__ import annotations

from threading import RLock
from typing import Any

from src.memory.session_state import clone_state, default_session_state, utc_now_iso


class InProcessSessionMemoryStore:
    """Small structured memory store for one backend process.

    The existing project already persists raw chat messages through Mongo/Postgres.
    This store keeps the compact state that routing and tools should trust.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._states: dict[str, dict[str, Any]] = {}
        self._messages: dict[str, list[dict[str, Any]]] = {}

    def get_state(self, session_id: str) -> dict[str, Any]:
        clean_session_id = str(session_id or "").strip()
        with self._lock:
            if clean_session_id not in self._states:
                self._states[clean_session_id] = default_session_state(clean_session_id)
            return clone_state(self._states[clean_session_id])

    def save_state(self, session_id: str, state: dict[str, Any]) -> dict[str, Any]:
        clean_session_id = str(session_id or "").strip()
        next_state = clone_state(state)
        next_state["session_id"] = clean_session_id
        next_state["updated_at"] = utc_now_iso()
        with self._lock:
            self._states[clean_session_id] = next_state
            return clone_state(next_state)

    def update_state(self, session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        state = self.get_state(session_id)
        _deep_update(state, updates)
        return self.save_state(session_id, state)

    def append_message(self, session_id: str, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        clean_session_id = str(session_id or "").strip()
        if not clean_session_id:
            return
        row = {
            "role": str(role or "").strip().lower(),
            "content": str(content or ""),
            "metadata": metadata or {},
            "created_at": utc_now_iso(),
        }
        with self._lock:
            rows = self._messages.setdefault(clean_session_id, [])
            rows.append(row)
            del rows[:-25]

    def get_last_messages(self, session_id: str, limit: int = 25) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit or 25), 25))
        with self._lock:
            return [dict(row) for row in self._messages.get(str(session_id or "").strip(), [])[-safe_limit:]]

    def hydrate_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        clean_session_id = str(session_id or "").strip()
        with self._lock:
            self._messages[clean_session_id] = [dict(row) for row in (messages or [])[-25:]]

    def delete_session(self, session_id: str) -> None:
        clean_session_id = str(session_id or "").strip()
        if not clean_session_id:
            return
        with self._lock:
            self._states.pop(clean_session_id, None)
            self._messages.pop(clean_session_id, None)


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
