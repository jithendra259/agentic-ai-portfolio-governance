from __future__ import annotations

import re
from typing import Any

from src.decision.decision_schema import HitlAction, MethodSelection, RegimeResult, ValidationResult


FORBIDDEN_TRADING_PATTERNS = (
    r"\bbuy\b",
    r"\bsell\b",
    r"\bhold\b",
    r"\bexecute\b",
    r"\border\b",
    r"\btrade\b",
    r"\btarget price\b",
)


class GovernancePolicy:
    """Deterministic advisory policy for the thesis decision layer."""

    calm_max = 0.50
    crisis_min = 0.85
    turnover_threshold = 0.40
    concentration_threshold = 0.35
    advisory_asset_cap = 0.45
    weight_sum_tolerance = 0.02

    def classify_regime(self, instability_index: float | None) -> RegimeResult:
        if instability_index is None:
            return RegimeResult(label="Unknown", threshold="missing_instability_index", confidence=0.25)
        if instability_index < self.calm_max:
            return RegimeResult(
                label="Calm",
                instability_index=instability_index,
                threshold="I_t < 0.50",
                confidence=0.9,
            )
        if instability_index < self.crisis_min:
            return RegimeResult(
                label="Elevated",
                instability_index=instability_index,
                threshold="0.50 <= I_t < 0.85",
                confidence=0.88,
            )
        return RegimeResult(
            label="Crisis",
            instability_index=instability_index,
            threshold="I_t >= 0.85",
            confidence=0.92,
        )

    def select_allocation_method(
        self,
        regime_label: str,
        graph_available: bool,
        solver_failed: bool = False,
    ) -> MethodSelection:
        if solver_failed:
            return MethodSelection(
                method="fallback_regularized_allocation",
                reason_codes=["optimizer_solver_failed", "advisory_fallback_required"],
                confidence=0.65,
            )
        if regime_label == "Crisis" and graph_available:
            return MethodSelection(
                method="graph_regularized_cvar_allocation",
                reason_codes=["crisis_regime", "graph_context_available"],
                confidence=0.88,
            )
        if regime_label in {"Elevated", "Crisis"}:
            codes = [f"{regime_label.lower()}_regime", "tail_risk_constraints_preferred"]
            if regime_label == "Crisis" and not graph_available:
                codes.append("graph_data_missing")
            return MethodSelection(method="cvar_risk_constrained_allocation", reason_codes=codes, confidence=0.82)
        return MethodSelection(
            method="mean_variance_or_equal_weight_baseline",
            reason_codes=["calm_regime", "baseline_method_sufficient"],
            confidence=0.78,
        )

    def evaluate_hitl(
        self,
        regime_label: str,
        metrics: dict[str, Any],
        graph_required: bool,
        graph_available: bool,
    ) -> HitlAction:
        reason_codes: list[str] = []
        if regime_label == "Crisis":
            reason_codes.append("crisis_regime")
        if float(metrics.get("turnover") or 0.0) > self.turnover_threshold:
            reason_codes.append("turnover_above_policy")
        if float(metrics.get("max_weight") or 0.0) > self.concentration_threshold:
            reason_codes.append("concentration_above_policy")
        if graph_required and not graph_available:
            reason_codes.append("graph_data_missing")
        if bool(metrics.get("validation_failed")):
            reason_codes.append("validation_failed")

        required = bool(reason_codes)
        level = "mandatory_review" if required else "standard_advisory"
        message = (
            "Human review is required before presenting this advisory governance conclusion."
            if required
            else "No human review trigger was activated by the current policy."
        )
        return HitlAction(required=required, level=level, reason_codes=reason_codes, message=message)

    def validate_weights(self, weights: dict[str, float]) -> ValidationResult:
        issues: list[str] = []
        warnings: list[str] = []
        if not weights:
            return ValidationResult(valid=False, issues=["weights_missing"])
        total = sum(float(value) for value in weights.values())
        if any(float(value) < 0 for value in weights.values()):
            issues.append("negative_weight")
        if abs(total - 1.0) > self.weight_sum_tolerance:
            issues.append("weights_do_not_sum_to_one")
        if any(float(value) > self.advisory_asset_cap for value in weights.values()):
            warnings.append("single_asset_weight_above_advisory_cap")
        return ValidationResult(valid=not issues, issues=issues, warnings=warnings)

    def validate_response_text(self, response_text: str) -> ValidationResult:
        issues = []
        normalized = response_text or ""
        for pattern in FORBIDDEN_TRADING_PATTERNS:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                issues.append("forbidden_trading_language")
                break
        warnings = [] if "advisory" in normalized.lower() else ["advisory_disclaimer_missing"]
        return ValidationResult(valid=not issues, issues=issues, warnings=warnings)
