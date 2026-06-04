from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock
from typing import Any


class AuditLogger:
    def __init__(self) -> None:
        self._lock = RLock()
        self._records: list[dict[str, Any]] = []

    def log(self, record: dict[str, Any]) -> dict[str, Any]:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": record.get("session_id"),
            "user_message": record.get("user_message"),
            "intent": record.get("intent"),
            "resolved_universe": record.get("resolved_universe"),
            "tool_called": record.get("tool_called"),
            "plot_id": record.get("plot_id"),
            "data_source": record.get("data_source"),
            "status": record.get("status"),
            "failure_reason": record.get("failure_reason"),
        }
        with self._lock:
            self._records.append(row)
            del self._records[:-1000]
        return row

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit or 100), 1000))
        with self._lock:
            return [dict(row) for row in self._records[-safe_limit:]]


GLOBAL_AUDIT_LOGGER = AuditLogger()
