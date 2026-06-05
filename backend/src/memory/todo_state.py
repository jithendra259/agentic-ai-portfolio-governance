from __future__ import annotations

from copy import deepcopy
from typing import Any


TODO_STATUS_VALUES = (
    "pending",
    "done",
    "blocked",
    "computed_by_fallback",
    "requires_user_input",
    "skipped_by_intent_lock",
    "failed_validation",
)

TODO_STATUS_SET = set(TODO_STATUS_VALUES)


def build_todo(
    step: int,
    name: str,
    *,
    status: str = "pending",
    required_data: list[str] | None = None,
    resolved_data: dict[str, Any] | None = None,
    can_fallback: bool = False,
    fallback_method: str | None = None,
    reason: str | None = None,
    user_options: list[str] | None = None,
) -> dict[str, Any]:
    if status not in TODO_STATUS_SET:
        raise ValueError(f"Unsupported todo status: {status}")
    return {
        "step": step,
        "name": name,
        "status": status,
        "required_data": list(required_data or []),
        "resolved_data": dict(resolved_data or {}),
        "can_fallback": bool(can_fallback),
        "fallback_method": fallback_method,
        "reason": reason,
        "user_options": list(user_options or []),
    }


def set_todo_status(
    plan: dict[str, Any],
    name: str,
    status: str,
    *,
    resolved_data: dict[str, Any] | None = None,
    fallback_method: str | None = None,
    reason: str | None = None,
    user_options: list[str] | None = None,
) -> dict[str, Any]:
    if status not in TODO_STATUS_SET:
        raise ValueError(f"Unsupported todo status: {status}")

    updated = deepcopy(plan)
    for todo in updated.get("todos", []):
        if todo.get("name") != name:
            continue
        todo["status"] = status
        if resolved_data is not None:
            todo["resolved_data"] = dict(resolved_data)
        if fallback_method is not None:
            todo["fallback_method"] = fallback_method
        if reason is not None:
            todo["reason"] = reason
        if user_options is not None:
            todo["user_options"] = list(user_options)
        break
    return updated


def todo_status(plan: dict[str, Any], name: str) -> str | None:
    for todo in plan.get("todos", []):
        if todo.get("name") == name:
            return todo.get("status")
    return None
