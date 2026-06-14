from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class PlanValidationError(ValueError):
    """Raised when a generated execution plan is syntactically valid but unsafe."""


class PlannedTask(BaseModel):
    task_id: str = Field(description="Unique task id")
    description: str = Field(description="Task description")
    dependencies: list[str] = Field(default_factory=list)
    name: str | None = None
    tool: str | None = None


class ExecutionPlan(BaseModel):
    goal: str
    nodes: list[PlannedTask]


def parse_execution_plan(raw: Any) -> ExecutionPlan:
    if isinstance(raw, ExecutionPlan):
        return raw
    if isinstance(raw, dict):
        return _validate_model(raw)
    if hasattr(raw, "model_dump"):
        return _validate_model(raw.model_dump())
    if hasattr(raw, "dict"):
        return _validate_model(raw.dict())

    text = _extract_json_text(str(raw or ""))
    data = json.loads(text)
    return _validate_model(data)


def validate_execution_plan(plan: ExecutionPlan) -> ExecutionPlan:
    if not plan.nodes:
        raise PlanValidationError("Execution plan must contain at least one node")

    task_ids = [node.task_id for node in plan.nodes]
    if len(task_ids) != len(set(task_ids)):
        raise PlanValidationError("Duplicate task_id detected in execution plan")

    task_id_set = set(task_ids)
    for node in plan.nodes:
        missing = [dep for dep in node.dependencies if dep not in task_id_set]
        if missing:
            raise PlanValidationError(
                f"Task {node.task_id} depends on missing task ids: {', '.join(missing)}"
            )

    adjacency: dict[str, list[str]] = {task_id: [] for task_id in task_ids}
    for node in plan.nodes:
        for dep in node.dependencies:
            adjacency[dep].append(node.task_id)

    visited: set[str] = set()
    active: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in active:
            raise PlanValidationError("Circular dependency detected in execution plan")
        if task_id in visited:
            return
        active.add(task_id)
        for neighbor in adjacency.get(task_id, []):
            visit(neighbor)
        active.remove(task_id)
        visited.add(task_id)

    for task_id in task_ids:
        visit(task_id)
    return plan


def build_sequential_fallback_plan(goal: str, steps: list[str] | None = None) -> ExecutionPlan:
    clean_steps = [str(step).strip(" -\t") for step in (steps or []) if str(step).strip(" -\t")]
    if not clean_steps:
        clean_steps = [
            "Clarify the portfolio governance objective and required data",
            "Gather or resolve the required portfolio inputs",
            "Compute the requested metrics and preserve exact results",
            "Summarize findings with limitations and next actions",
        ]

    nodes = []
    previous_id: str | None = None
    for index, step in enumerate(clean_steps[:6], start=1):
        task_id = f"task_{index}"
        nodes.append(
            PlannedTask(
                task_id=task_id,
                description=step,
                dependencies=[previous_id] if previous_id else [],
                name=_slugify(step) or f"step_{index}",
                tool="generic_executor",
            )
        )
        previous_id = task_id
    return ExecutionPlan(goal=goal, nodes=nodes)


def extract_task_descriptions(raw: Any) -> list[str]:
    try:
        if isinstance(raw, ExecutionPlan):
            return [node.description for node in raw.nodes if node.description]
        if isinstance(raw, dict):
            nodes = raw.get("nodes", [])
        elif hasattr(raw, "model_dump"):
            nodes = raw.model_dump().get("nodes", [])
        elif hasattr(raw, "dict"):
            nodes = raw.dict().get("nodes", [])
        else:
            text = _extract_json_text(str(raw or ""))
            nodes = json.loads(text).get("nodes", [])
    except Exception:
        return []

    descriptions = []
    for node in nodes:
        if isinstance(node, PlannedTask):
            description = node.description
        elif isinstance(node, dict):
            description = node.get("description") or node.get("name")
        else:
            description = None
        if description and str(description).strip():
            descriptions.append(str(description).strip())
    return descriptions


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]

    # Tolerate common LLM JSON slips.
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _validate_model(data: dict[str, Any]) -> ExecutionPlan:
    try:
        if hasattr(ExecutionPlan, "model_validate"):
            return ExecutionPlan.model_validate(data)
        return ExecutionPlan.parse_obj(data)
    except ValidationError as exc:
        raise PlanValidationError(str(exc)) from exc


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:40]
