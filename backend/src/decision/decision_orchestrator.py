from __future__ import annotations

from typing import Any

from src.decision.governance_decision_agent import GovernanceDecisionAgent
from src.decision.governance_policy import GovernancePolicy


class DecisionOrchestrator:
    def __init__(self, agent: GovernanceDecisionAgent | None = None):
        self.agent = agent or GovernanceDecisionAgent()
        self.policy: GovernancePolicy = self.agent.policy

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.agent.decide(payload)

    def portfolio_health(self, payload: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(payload or {})
        enriched.setdefault("decision_context", {})["decision_type"] = "portfolio_health_decision"
        return self.agent.decide(enriched)

    def advisory_allocation(self, payload: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(payload or {})
        message = str(enriched.get("message") or "")
        if "advisory allocation" not in message.lower():
            enriched["message"] = f"advisory allocation {message}".strip()
        return self.agent.decide(enriched)

    def select_plots(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.agent.select_plots(payload or {})

    def validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        response_text = str((payload or {}).get("response_text") or "")
        decision = (payload or {}).get("decision") or {}
        text_validation = self.policy.validate_response_text(response_text)
        issues = list(text_validation.issues)
        warnings = list(text_validation.warnings)

        trace = decision.get("traceability") or {}
        if not trace.get("claims"):
            issues.append("traceability_missing")
        embedded_validation = decision.get("validation") or {}
        if embedded_validation and not embedded_validation.get("valid", True):
            issues.extend(embedded_validation.get("issues", ["embedded_validation_failed"]))

        return {
            "valid": not issues,
            "issues": issues,
            "warnings": warnings,
            "checks": ["advisory_language", "traceability", "embedded_validation"],
        }
