from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from src.decision.decision_orchestrator import DecisionOrchestrator
from src.decision.decision_schema import DecisionRequest, ValidationRequest


router = APIRouter(prefix="/api/governance/decision", tags=["governance-decision"])
orchestrator = DecisionOrchestrator()


@router.post("")
def governance_decision(request: DecisionRequest) -> dict[str, Any]:
    return orchestrator.decide(_dump(request))


@router.post("/portfolio-health")
def portfolio_health(request: DecisionRequest) -> dict[str, Any]:
    return orchestrator.portfolio_health(_dump(request))


@router.post("/advisory-allocation")
def advisory_allocation(request: DecisionRequest) -> dict[str, Any]:
    return orchestrator.advisory_allocation(_dump(request))


@router.post("/plots")
def decision_plots(request: DecisionRequest) -> dict[str, Any]:
    return orchestrator.select_plots(_dump(request))


@router.post("/validate")
def validate_decision(request: ValidationRequest) -> dict[str, Any]:
    return orchestrator.validate(_dump(request))


def _dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
