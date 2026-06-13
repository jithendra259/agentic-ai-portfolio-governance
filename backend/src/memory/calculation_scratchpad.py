from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any


class CalculationScratchpad:
    """Durable calculation ledger for exact formulas and numeric results."""

    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self._records: list[dict[str, Any]] = []
        self._next_index = 1
        for record in records or []:
            self._records.append(dict(record))
            self._next_index = max(self._next_index, self._index_after(record.get("calculation_id")))

    def record(
        self,
        *,
        label: str,
        formula: str,
        inputs: dict[str, Any] | None = None,
        result: Any = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = {
            "calculation_id": f"calc_{self._next_index:03d}",
            "label": str(label),
            "formula": str(formula),
            "inputs": self._json_safe(inputs or {}),
            "result": self._json_safe(result),
            "source": source,
            "metadata": self._json_safe(metadata or {}),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._next_index += 1
        self._records.append(row)
        return dict(row)

    def get(self, calculation_id: str) -> dict[str, Any] | None:
        for record in self._records:
            if record.get("calculation_id") == calculation_id:
                return dict(record)
        return None

    def compact_summary(self, limit: int = 8) -> list[dict[str, Any]]:
        safe_limit = max(1, int(limit or 8))
        return [
            {
                "calculation_id": record.get("calculation_id"),
                "label": record.get("label"),
                "formula": record.get("formula"),
                "result": record.get("result"),
                "source": record.get("source"),
            }
            for record in self._records[-safe_limit:]
        ]

    def to_list(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._records]

    @classmethod
    def from_list(cls, records: list[dict[str, Any]] | None) -> "CalculationScratchpad":
        return cls(records or [])

    @staticmethod
    def _index_after(calculation_id: Any) -> int:
        if not isinstance(calculation_id, str) or not calculation_id.startswith("calc_"):
            return 1
        try:
            return int(calculation_id.split("_", 1)[1]) + 1
        except (IndexError, ValueError):
            return 1

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [cls._json_safe(item) for item in value]
        return value
