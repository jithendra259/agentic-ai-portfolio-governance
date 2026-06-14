from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is installed in production/test envs
    np = None

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is installed in production/test envs
    pd = None


def sanitize_for_mongodb(obj: Any) -> Any:
    """Convert compute-heavy Python objects into JSON/BSON-safe state values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if pd is not None and isinstance(obj, pd.DataFrame):
        return sanitize_for_mongodb(obj.to_dict(orient="records"))
    if pd is not None and isinstance(obj, pd.Series):
        return sanitize_for_mongodb(obj.to_dict())
    if np is not None and isinstance(obj, np.ndarray):
        return sanitize_for_mongodb(obj.tolist())
    if np is not None and isinstance(obj, np.generic):
        return sanitize_for_mongodb(obj.item())
    if is_dataclass(obj):
        return sanitize_for_mongodb(asdict(obj))
    if isinstance(obj, dict):
        return {str(key): sanitize_for_mongodb(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_mongodb(value) for value in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(sanitize_for_mongodb(value) for value in obj)
    if hasattr(obj, "model_dump"):
        return sanitize_for_mongodb(obj.model_dump())
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return sanitize_for_mongodb(obj.to_dict())
        except TypeError:
            pass
    return str(obj)


class SanitizingCheckpointer:
    """Gateway wrapper that keeps LangGraph checkpoint payloads JSON/BSON-safe."""

    def __init__(self, inner: Any):
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @staticmethod
    def _sanitize_checkpoint_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args:
            return args
        config, *payload = args
        return (config, *(sanitize_for_mongodb(arg) for arg in payload))

    @staticmethod
    def _sanitize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value if key == "config" else sanitize_for_mongodb(value)
            for key, value in kwargs.items()
        }

    def put(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.put(
            *self._sanitize_checkpoint_args(args),
            **self._sanitize_kwargs(kwargs),
        )

    async def aput(self, *args: Any, **kwargs: Any) -> Any:
        return await self.inner.aput(
            *self._sanitize_checkpoint_args(args),
            **self._sanitize_kwargs(kwargs),
        )

    def put_writes(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.put_writes(
            *self._sanitize_checkpoint_args(args),
            **self._sanitize_kwargs(kwargs),
        )

    async def aput_writes(self, *args: Any, **kwargs: Any) -> Any:
        return await self.inner.aput_writes(
            *self._sanitize_checkpoint_args(args),
            **self._sanitize_kwargs(kwargs),
        )


def wrap_checkpointer_for_mongodb(checkpointer: Any) -> Any:
    if checkpointer is None or isinstance(checkpointer, SanitizingCheckpointer):
        return checkpointer
    return SanitizingCheckpointer(checkpointer)
