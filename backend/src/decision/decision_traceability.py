from __future__ import annotations

import uuid
from typing import Any

from src.decision.decision_schema import DecisionTrace


class DecisionTraceability:
    def build_trace(
        self,
        *,
        decision_type: str,
        regime: Any,
        method: Any,
        hitl: Any,
        validation: Any,
        limitations: list[str],
    ) -> DecisionTrace:
        claims = [
            {
                "claim_id": "regime_classification",
                "claim": f"Regime classified as {getattr(regime, 'label', 'Unknown')}.",
                "source": "governance_policy.classify_regime",
                "status": "validated" if getattr(regime, "label", "Unknown") != "Unknown" else "limited",
            },
            {
                "claim_id": "method_selection",
                "claim": f"Allocation method selected: {getattr(method, 'method', 'not_applicable')}.",
                "source": "governance_policy.select_allocation_method",
                "status": "validated",
            },
            {
                "claim_id": "hitl_policy",
                "claim": "HITL policy evaluated from regime, turnover, concentration, and graph availability.",
                "source": "governance_policy.evaluate_hitl",
                "status": "validated",
            },
            {
                "claim_id": "response_validation",
                "claim": "Advisory-language guardrails and weight checks were evaluated.",
                "source": "governance_policy.validation",
                "status": "validated" if getattr(validation, "valid", False) else "failed",
            },
        ]
        return DecisionTrace(
            decision_id=str(uuid.uuid4()),
            claims=claims,
            source_modules=[
                "intent_router",
                "governance_policy",
                "governance_decision_agent",
                "decision_traceability",
            ],
            limitations=limitations,
        )
