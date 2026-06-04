from __future__ import annotations

from typing import Any

from src.decision.decision_schema import DecisionRequest
from src.decision.decision_traceability import DecisionTraceability
from src.decision.governance_policy import GovernancePolicy
from src.decision.plot_registry import registry_audit, select_registry_plots
from src.intent.intent_router import IntentRouter


DECISION_TYPE_BY_SUB_INTENT = {
    "instability_regime": "instability_decision",
    "diversification": "diversification_decision",
    "risk_governance": "risk_governance_decision",
    "advisory_allocation": "advisory_allocation_decision",
    "graph_contagion": "graph_contagion_decision",
    "hitl_governance": "hitl_governance_decision",
    "smart_plot_selection": "plot_decision",
    "full_plot_coverage": "plot_decision",
    "response_validation": "validation_decision",
}


class GovernanceDecisionAgent:
    def __init__(
        self,
        policy: GovernancePolicy | None = None,
        intent_router: IntentRouter | None = None,
        traceability: DecisionTraceability | None = None,
    ):
        self.policy = policy or GovernancePolicy()
        self.intent_router = intent_router or IntentRouter()
        self.traceability = traceability or DecisionTraceability()

    def decide(self, payload: dict[str, Any] | DecisionRequest) -> dict[str, Any]:
        request = payload if isinstance(payload, DecisionRequest) else DecisionRequest(**(payload or {}))
        plan = self.intent_router.build_execution_plan(request.message, previous_analysis=request.previous_analysis)
        decision_type = self._decision_type(plan, request.message)
        tickers = request.tickers or plan.get("entities", {}).get("tickers", [])
        current_weights = request.current_weights or plan.get("entities", {}).get("current_weights", {})
        metrics = self._derive_metrics(request.metrics, current_weights, request.use_sample_data_when_missing)
        graph_available = bool(request.graph.get("available")) if request.graph else False
        graph_required = decision_type in {"graph_contagion_decision", "instability_decision", "advisory_allocation_decision"}

        regime = self.policy.classify_regime(metrics.get("instability_index"))
        method = self.policy.select_allocation_method(
            regime.label,
            graph_available=graph_available,
            solver_failed=bool(metrics.get("optimizer_solver_failed")),
        )
        advisory_weights = self._advisory_weights(tickers, current_weights)
        weight_validation = self.policy.validate_weights(advisory_weights) if advisory_weights else self.policy.validate_weights(current_weights)
        hitl = self.policy.evaluate_hitl(regime.label, metrics, graph_required, graph_available)
        draft_response = self._draft_response(decision_type, regime.label, method.method, hitl.required)
        text_validation = self.policy.validate_response_text(draft_response)
        validation = self._combine_validation(weight_validation, text_validation)
        limitations = self._limitations(request, metrics, graph_required, graph_available, advisory_weights)
        trace = self.traceability.build_trace(
            decision_type=decision_type,
            regime=regime,
            method=method,
            hitl=hitl,
            validation=validation,
            limitations=limitations,
        )

        return {
            "decision_type": decision_type,
            "intent_plan": plan,
            "regime": self._model_dump(regime),
            "method_selection": self._model_dump(method),
            "metrics": metrics,
            "hitl": self._model_dump(hitl),
            "advisory_allocation": {
                "method": method.method,
                "weights": advisory_weights,
                "current_weights": current_weights,
                "weight_validation": self._model_dump(weight_validation),
            },
            "smart_plots": self.select_plots(
                {
                    "message": request.message,
                    "decision_context": {"decision_type": decision_type},
                    "metrics": metrics,
                }
            ),
            "chatbot_plan": {
                "response_mode": plan.get("response", {}).get("mode", "standard"),
                "must_include": ["reason_codes", "limitations", "confidence", "advisory_only_disclaimer"],
                "must_avoid": ["buy", "sell", "hold", "execute", "order", "trade"],
                "draft_response": draft_response,
            },
            "validation": self._model_dump(validation),
            "traceability": self._model_dump(trace),
            "limitations": limitations,
        }

    def select_plots(self, payload: dict[str, Any]) -> dict[str, Any]:
        message = str(payload.get("message") or "")
        decision_context = payload.get("decision_context") or {}
        decision_type = decision_context.get("decision_type") or self._decision_type(
            self.intent_router.build_execution_plan(message), message
        )
        normalized = message.lower()
        negated_full_request = any(
            phrase in normalized
            for phrase in (
                "do not show all 88",
                "do not return all 88",
                "don't show all 88",
                "don't return all 88",
                "unless i explicitly ask for full analytics",
                "unless full analytics view",
            )
        )
        explicit_full_request = any(
            phrase in normalized
            for phrase in (
                "audit the full analytics",
                "full analytics plot registry",
                "render all 88",
                "show all 88",
                "return all 88",
                "check all 88",
                "list all 88",
            )
        )
        if explicit_full_request and not negated_full_request:
            audit = registry_audit()
            return {
                "plot_mode": "full_analytics",
                "max_plots": 88,
                "plots": [],
                "registry": audit["registry"],
                "audit": {
                    "total_registered": audit["total_registered"],
                    "duplicate_plot_ids": audit["duplicate_plot_ids"],
                    "tabs": audit["tabs"],
                },
                "reason": "Explicit full analytics registry audit request; Smart View rendering remains disabled.",
            }

        library = {
            "portfolio_health_decision": [
                "plot_19_composite_instability_index_plot",
                "plot_39_hhi_concentration_index",
                "plot_49_portfolio_drawdown_plot",
                "plot_51_cvar_comparison_plot",
                "plot_29_current_vs_advisory_allocation_by_ticker",
                "plot_55_risk_contribution_by_ticker",
                "plot_11_return_correlation_heatmap",
            ],
            "advisory_allocation_decision": [
                "plot_29_current_vs_advisory_allocation_by_ticker",
                "plot_32_allocation_change_by_ticker",
                "plot_37_allocation_constraint_boundary_plot",
                "plot_55_risk_contribution_by_ticker",
                "plot_77_turnover_alert_plot",
            ],
            "instability_decision": [
                "plot_19_composite_instability_index_plot",
                "plot_20_regime_classification_timeline",
                "plot_22_volatility_spike_component_plot",
                "plot_23_correlation_spike_component_plot",
                "plot_24_maximum_drawdown_component_plot",
                "plot_11_return_correlation_heatmap",
                "plot_49_portfolio_drawdown_plot",
            ],
        }
        selected = library.get(decision_type, library["portfolio_health_decision"])
        return {
            "plot_mode": "smart_view",
            "default_tab": "Smart View",
            "max_plots": len(selected),
            "plots": select_registry_plots(selected[:8]),
        }

    def _decision_type(self, plan: dict[str, Any], message: str) -> str:
        sub_intent = plan.get("sub_intent", "unknown")
        if sub_intent in DECISION_TYPE_BY_SUB_INTENT:
            return DECISION_TYPE_BY_SUB_INTENT[sub_intent]
        normalized = message.lower()
        if "health" in normalized or "portfolio" in normalized:
            return "portfolio_health_decision"
        return "portfolio_health_decision"

    def _derive_metrics(
        self,
        supplied: dict[str, float],
        current_weights: dict[str, float],
        use_sample_data_when_missing: bool,
    ) -> dict[str, float | bool]:
        metrics: dict[str, float | bool] = dict(supplied or {})
        if "max_weight" not in metrics and current_weights:
            metrics["max_weight"] = max(float(value) for value in current_weights.values())
        if "hhi" not in metrics and current_weights:
            metrics["hhi"] = sum(float(value) ** 2 for value in current_weights.values())
        if "turnover" not in metrics:
            metrics["turnover"] = 0.0
        if "instability_index" not in metrics and use_sample_data_when_missing:
            metrics["instability_index"] = 0.55
            metrics["sample_data_used"] = True
        return metrics

    def _advisory_weights(self, tickers: list[str], current_weights: dict[str, float]) -> dict[str, float]:
        names = tickers or list(current_weights.keys())
        if not names:
            return {}
        equal = round(1.0 / len(names), 6)
        weights = {ticker: equal for ticker in names}
        residual = round(1.0 - sum(weights.values()), 6)
        if residual:
            weights[names[0]] = round(weights[names[0]] + residual, 6)
        return weights

    def _draft_response(self, decision_type: str, regime: str, method: str, hitl_required: bool) -> str:
        review = " Human review is required by policy." if hitl_required else ""
        return (
            f"This advisory governance review classifies the decision as {decision_type}, "
            f"with regime {regime} and method {method}.{review} "
            "It is informational and should be reviewed with the documented limitations."
        )

    def _combine_validation(self, *results: Any) -> Any:
        issues: list[str] = []
        warnings: list[str] = []
        for result in results:
            issues.extend(getattr(result, "issues", []))
            warnings.extend(getattr(result, "warnings", []))
        return type(results[0])(valid=not issues, issues=issues, warnings=warnings)

    def _limitations(
        self,
        request: DecisionRequest,
        metrics: dict[str, Any],
        graph_required: bool,
        graph_available: bool,
        advisory_weights: dict[str, float],
    ) -> list[str]:
        limitations: list[str] = []
        if metrics.get("sample_data_used"):
            limitations.append("Sample governance metrics were used because live analytics were not supplied.")
        if graph_required and not graph_available:
            limitations.append("Graph contagion data was not available; graph-dependent conclusions are limited.")
        if advisory_weights and not request.metrics.get("optimizer_solver_status"):
            limitations.append("Decision-layer v1 used deterministic advisory reference weights unless optimizer output is supplied.")
        return limitations

    def _model_dump(self, value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        return dict(value)
