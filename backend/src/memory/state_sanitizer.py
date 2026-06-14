from __future__ import annotations

from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Mapping, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
    DeltaChannelHistory,
)

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
    if isinstance(obj, (list, tuple, deque)):
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


def reconstruct_from_mongodb(obj: Any) -> Any:
    """Best-effort reconstruction for values loaded from sanitized Mongo state."""
    if isinstance(obj, dict):
        marker = obj.get("__type__")
        if marker == "DataFrame" and pd is not None:
            return pd.DataFrame(obj.get("data", []))
        if marker == "Series" and pd is not None:
            return pd.Series(obj.get("data", {}), name=obj.get("name"))
        if marker == "Decimal":
            return Decimal(str(obj.get("value", "0")))
        if marker == "datetime":
            return datetime.fromisoformat(str(obj.get("value")))
        if marker == "date":
            return date.fromisoformat(str(obj.get("value")))
        return {key: reconstruct_from_mongodb(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [reconstruct_from_mongodb(value) for value in obj]
    return obj


class SanitizingCheckpointer(BaseCheckpointSaver):
    """Gateway wrapper that keeps LangGraph checkpoint payloads JSON/BSON-safe."""

    def __init__(self, inner: BaseCheckpointSaver):
        super().__init__(serde=getattr(inner, "serde", None))
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def get_next_version(self, current: str | None, channel: Any) -> str:
        return self.inner.get_next_version(current, channel)

    @property
    def config_specs(self) -> list:
        return self.inner.config_specs

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

    def get(self, config: RunnableConfig) -> Checkpoint | None:
        return self.inner.get(config)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.inner.get_tuple(config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        return self.inner.list(config, filter=filter, before=before, limit=limit)

    def put(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.put(
            *self._sanitize_checkpoint_args(args),
            **self._sanitize_kwargs(kwargs),
        )

    def put_writes(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.put_writes(
            *self._sanitize_checkpoint_args(args),
            **self._sanitize_kwargs(kwargs),
        )

    def delete_thread(self, thread_id: str) -> None:
        return self.inner.delete_thread(thread_id)

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        return self.inner.delete_for_runs(run_ids)

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        return self.inner.copy_thread(source_thread_id, target_thread_id)

    def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        return self.inner.prune(thread_ids, strategy=strategy)

    async def aget(self, config: RunnableConfig) -> Checkpoint | None:
        return await self.inner.aget(config)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await self.inner.aget_tuple(config)

    def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        return self.inner.alist(config, filter=filter, before=before, limit=limit)

    async def aput(self, *args: Any, **kwargs: Any) -> Any:
        return await self.inner.aput(
            *self._sanitize_checkpoint_args(args),
            **self._sanitize_kwargs(kwargs),
        )

    async def aput_writes(self, *args: Any, **kwargs: Any) -> Any:
        return await self.inner.aput_writes(
            *self._sanitize_checkpoint_args(args),
            **self._sanitize_kwargs(kwargs),
        )

    async def adelete_thread(self, thread_id: str) -> None:
        return await self.inner.adelete_thread(thread_id)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        return await self.inner.adelete_for_runs(run_ids)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        return await self.inner.acopy_thread(source_thread_id, target_thread_id)

    async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        return await self.inner.aprune(thread_ids, strategy=strategy)

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        return self.inner.get_delta_channel_history(config=config, channels=channels)

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        return await self.inner.aget_delta_channel_history(config=config, channels=channels)


def wrap_checkpointer_for_mongodb(checkpointer: Any) -> Any:
    if checkpointer is None or isinstance(checkpointer, SanitizingCheckpointer):
        return checkpointer
    return SanitizingCheckpointer(checkpointer)
