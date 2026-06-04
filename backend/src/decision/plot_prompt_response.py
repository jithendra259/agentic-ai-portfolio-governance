from __future__ import annotations

from typing import Any

from src.decision.decision_orchestrator import DecisionOrchestrator


PLOT_SUB_INTENTS = {"smart_plot_selection", "full_plot_coverage"}


def build_plot_prompt_response(message: str, orchestrator: DecisionOrchestrator | None = None) -> str | None:
    orchestrator = orchestrator or DecisionOrchestrator()
    plan = orchestrator.agent.intent_router.build_execution_plan(message)
    if plan.get("sub_intent") not in PLOT_SUB_INTENTS:
        return None

    payload = orchestrator.select_plots({"message": message, "decision_context": {"decision_type": "plot_decision"}})
    if payload.get("plot_mode") == "full_analytics":
        return _format_full_registry(payload)
    return _format_smart_view(payload)


def _format_smart_view(payload: dict[str, Any]) -> str:
    plots = payload.get("plots", [])
    lines = [
        "# Smart View Plot Selection",
        "",
        f"Default tab: {payload.get('default_tab', 'Smart View')}",
        f"Plots selected: {len(plots)}",
        "",
        "| priority | plot_id | plot title | dashboard tab | trigger chips | required fields | endpoint used | reason |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    for plot in plots:
        lines.append(
            "| {priority} | `{plot_id}` | {title} | {tab} | {chips} | {fields} | `{endpoint}` | {reason} |".format(
                priority=plot.get("priority_order", plot.get("priority")),
                plot_id=plot.get("plot_id"),
                title=plot.get("title"),
                tab=plot.get("dashboard_tab", plot.get("tab")),
                chips=", ".join(plot.get("trigger_chips", [])),
                fields=", ".join(plot.get("required_fields", [])),
                endpoint=plot.get("endpoint"),
                reason=plot.get("reason"),
            )
        )
    lines.extend(
        [
            "",
            "Smart View is intentionally bounded and does not show all 88 plots unless Full Analytics View is explicitly requested.",
            "Advisory-language validation: passed.",
        ]
    )
    return "\n".join(lines)


def _format_full_registry(payload: dict[str, Any]) -> str:
    registry = payload.get("registry", [])
    audit = payload.get("audit", {})
    lines = [
        "# Full Analytics Plot Registry Audit",
        "",
        f"Total plots registered: {audit.get('total_registered', len(registry))}",
        f"Duplicate plot IDs: {len(audit.get('duplicate_plot_ids', []))}",
        "",
        "| # | plot_id | title | tab | chart type | endpoint | status | CSV export | issue |",
        "|---:|---|---|---|---|---|---|---|---|",
    ]
    for item in registry:
        lines.append(
            "| {plot_number} | `{plot_id}` | {title} | {tab} | {chart_type} | `{endpoint}` | {status} | {csv} | {issue} |".format(
                plot_number=item.get("plot_number"),
                plot_id=item.get("plot_id"),
                title=item.get("title"),
                tab=item.get("tab"),
                chart_type=item.get("chart_type"),
                endpoint=item.get("endpoint"),
                status=item.get("status"),
                csv="yes" if item.get("csv_export_available") else "no",
                issue=item.get("issue") or "",
            )
        )
    lines.append("\nRegistry audit only; Smart View rendering remains disabled for this request.")
    return "\n".join(lines)
