from __future__ import annotations

from copy import deepcopy
from typing import Any


def append_list(existing: list[Any] | None, new: list[Any] | None) -> list[Any]:
    return [*(existing or []), *(new or [])]


def merge_unique_values(existing: list[Any] | None, new: list[Any] | None) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for value in [*(existing or []), *(new or [])]:
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        merged.append(value)
    return merged


def merge_dict_state(existing: dict[str, Any] | None, new: dict[str, Any] | None) -> dict[str, Any]:
    if not existing:
        return deepcopy(new or {})
    if not new:
        return deepcopy(existing)

    merged = deepcopy(existing)
    for key, value in new.items():
        old_value = merged.get(key)
        if isinstance(old_value, dict) and isinstance(value, dict):
            merged[key] = merge_dict_state(old_value, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def merge_records_by_id(
    existing: list[dict[str, Any]] | None,
    new: list[dict[str, Any]] | None,
    *,
    id_key: str,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    index: dict[Any, int] = {}
    for record in [*(existing or []), *(new or [])]:
        if not isinstance(record, dict):
            continue
        key = record.get(id_key)
        if key is None or key not in index:
            index[key] = len(merged)
            merged.append(deepcopy(record))
            continue
        merged[index[key]] = merge_dict_state(merged[index[key]], record)
    return merged


def merge_blackboard_state(
    existing: dict[str, Any] | None,
    new: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = merge_dict_state(existing, new)
    existing = existing or {}
    new = new or {}

    for key in ("completed_tasks", "failed_tasks", "skipped_tasks", "task_order"):
        if key in existing or key in new:
            merged[key] = merge_unique_values(existing.get(key, []), new.get(key, []))

    for key in ("react_history", "action_history", "loop_events", "pending_human_inputs"):
        if key in existing or key in new:
            merged[key] = append_list(existing.get(key, []), new.get(key, []))

    if "calculation_scratchpad" in existing or "calculation_scratchpad" in new:
        merged["calculation_scratchpad"] = merge_records_by_id(
            existing.get("calculation_scratchpad", []),
            new.get("calculation_scratchpad", []),
            id_key="calculation_id",
        )

    if "tasks" in existing or "tasks" in new:
        merged["tasks"] = merge_dict_state(existing.get("tasks", {}), new.get("tasks", {}))

    return merged
