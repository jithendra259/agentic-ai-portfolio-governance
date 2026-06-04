from __future__ import annotations

import re
from typing import Any


TAB_NAMES = [
    "Data EDA",
    "Correlation & Covariance EDA",
    "Instability Monitor",
    "Advisory Diversification",
    "Diversification Diagnostics",
    "Risk Governance",
    "Contagion Graph Analysis",
    "Agent Governance and Explainability",
    "Evaluation and Backtesting",
]


PLOT_GROUPS = [
    (
        "Data EDA",
        "/api/analytics/eda",
        [
            ("Adjusted Close Price Trend", "line", ["date", "ticker", "close"]),
            ("Normalized Price Movement", "line", ["date", "ticker", "normalized_price"]),
            ("Daily Log Returns Plot", "line", ["date", "ticker", "log_return"]),
            ("Return Distribution Plot", "bar", ["ticker", "bin", "frequency"]),
            ("Boxplot of Daily Returns by Ticker", "boxplot", ["ticker", "min", "q1", "median", "q3", "max"]),
            ("Rolling Volatility Plot", "line", ["date", "ticker", "rolling_volatility"]),
            ("Rolling Mean Return Plot", "line", ["date", "ticker", "rolling_mean_return"]),
            ("Cumulative Return Plot", "line", ["date", "ticker", "cumulative_return"]),
            ("Missing Data Heatmap", "heatmap", ["date", "ticker", "missing_flag"]),
            ("Outlier Return Detection Plot", "scatter", ["date", "ticker", "log_return", "z_score"]),
        ],
    ),
    (
        "Correlation & Covariance EDA",
        "/api/analytics/eda",
        [
            ("Return Correlation Heatmap", "heatmap", ["tickerX", "tickerY", "correlation"]),
            ("Rolling Average Correlation Plot", "line", ["date", "averageCorrelation"]),
            ("Covariance Matrix Heatmap", "heatmap", ["tickerX", "tickerY", "covariance"]),
            ("Covariance Drift Plot", "line", ["date", "covarianceDrift"]),
            ("Correlation Stress Plot", "line", ["date", "correlationStress"]),
            ("Eigenvalue Spectrum Plot", "bar", ["component", "eigenvalue"]),
            ("PCA Explained Variance Plot", "bar", ["component", "explainedVariancePercent"]),
            ("Pairwise Return Scatter Matrix", "scatter", ["tickerX", "tickerY", "returnX", "returnY"]),
        ],
    ),
    (
        "Instability Monitor",
        "/api/analytics/instability",
        [
            ("Composite Instability Index Plot", "gauge", ["date", "instabilityIndex", "regime"]),
            ("Regime Classification Timeline", "line", ["date", "regime", "threshold"]),
            ("Market Stress Index Plot", "line", ["date", "marketStressIndex"]),
            ("Volatility Spike Component Plot", "line", ["date", "volatilitySpike"]),
            ("Correlation Spike Component Plot", "line", ["date", "correlationSpike"]),
            ("Maximum Drawdown Component Plot", "line", ["date", "maxDrawdown"]),
            ("Instability Component Contribution Plot", "stacked_bar", ["component", "contribution"]),
            ("Crisis Window Activation Plot", "bar", ["date", "crisisWindowActive"]),
            ("Regime Frequency Plot", "pie", ["regime", "count"]),
            ("Threshold Sensitivity Plot", "line", ["threshold", "activationCount"]),
        ],
    ),
    (
        "Advisory Diversification",
        "/api/analytics/advisory-allocation",
        [
            ("Current vs Advisory Allocation by Ticker", "bar", ["ticker", "currentWeight", "advisoryWeight"]),
            ("Current vs Advisory Sector Allocation", "bar", ["sector", "currentWeight", "advisoryWeight"]),
            ("Advisory Allocation Pie Chart", "pie", ["ticker", "advisoryWeight"]),
            ("Allocation Change by Ticker", "bar", ["ticker", "weightChange", "reasonCode"]),
            ("Allocation Adaptation Over Time", "line", ["date", "ticker", "advisoryWeight"]),
            ("Critical Condition Allocation Shift", "waterfall", ["ticker", "beforeWeight", "afterWeight"]),
            ("Ticker Exposure Waterfall", "waterfall", ["ticker", "exposureContribution"]),
            ("Cash/Defensive Buffer Plot", "gauge", ["bufferType", "weight"]),
            ("Allocation Constraint Boundary Plot", "line", ["ticker", "weight", "constraint"]),
            ("Before vs After Diversification Map", "radar", ["metric", "before", "after"]),
        ],
    ),
    (
        "Diversification Diagnostics",
        "/api/analytics/diversification",
        [
            ("HHI Concentration Index", "gauge", ["hhi", "threshold"]),
            ("Effective Number of Holdings", "bar", ["effectiveHoldings", "tickerCount"]),
            ("Diversification Score Before vs After", "bar", ["metric", "before", "after"]),
            ("Ticker Concentration Plot", "bar", ["ticker", "weight"]),
            ("Sector Concentration Plot", "bar", ["sector", "weight"]),
            ("Concentration Threshold Breach Plot", "bar", ["entity", "actual", "threshold"]),
            ("Diversification Ratio Plot", "line", ["date", "diversificationRatio"]),
            ("Top Holdings Concentration Plot", "bar", ["ticker", "cumulativeWeight"]),
            ("Portfolio Weight Dispersion Plot", "boxplot", ["ticker", "weight"]),
            ("Equal Weight Distance Plot", "bar", ["ticker", "distanceFromEqualWeight"]),
        ],
    ),
    (
        "Risk Governance",
        "/api/analytics/risk-governance",
        [
            ("Portfolio Drawdown Plot", "line", ["date", "drawdown"]),
            ("Maximum Drawdown Comparison", "bar", ["strategy", "maxDrawdown"]),
            ("CVaR Comparison Plot", "bar", ["strategy", "cvar"]),
            ("VaR and CVaR Tail Loss Plot", "line", ["quantile", "loss", "metric"]),
            ("Rolling CVaR Plot", "line", ["date", "rollingCvar"]),
            ("Rolling Volatility Comparison", "line", ["date", "strategy", "rollingVolatility"]),
            ("Risk Contribution by Ticker", "bar", ["ticker", "riskContribution"]),
            ("Allocation vs Risk Contribution Plot", "scatter", ["ticker", "allocationShare", "riskContribution"]),
            ("Sharpe Ratio Comparison", "bar", ["strategy", "sharpe"]),
            ("Sortino Ratio Comparison", "bar", ["strategy", "sortino"]),
        ],
    ),
    (
        "Contagion Graph Analysis",
        "/api/analytics/contagion",
        [
            ("Institution-Asset Bipartite Graph", "network", ["institution", "ticker", "exposure"]),
            ("Ticker Co-Ownership Graph", "network", ["tickerX", "tickerY", "coOwnership"]),
            ("Eigenvector Centrality by Ticker", "bar", ["ticker", "centrality"]),
            ("Contagion Penalty Score Plot", "bar", ["ticker", "penaltyScore"]),
            ("Centrality vs Advisory Weight Plot", "scatter", ["ticker", "centrality", "advisoryWeight"]),
            ("Centrality vs Weight Change Plot", "scatter", ["ticker", "centrality", "weightChange"]),
            ("Graph-Regularized CVaR Penalty Plot", "line", ["lambda", "penalty"]),
            ("Sigmoid Trust Function Plot", "line", ["instabilityIndex", "lambda"]),
            ("Co-Ownership Density Plot", "gauge", ["density", "threshold"]),
            ("Top Institutional Holder Exposure Plot", "bar", ["institution", "exposurePercent"]),
        ],
    ),
    (
        "Agent Governance and Explainability",
        "/api/analytics/agent-governance",
        [
            ("Five-Agent Pipeline Status", "timeline", ["agentName", "status", "startedAt", "completedAt"]),
            ("Blackboard Audit Trail Timeline", "timeline", ["timestamp", "agentName", "action"]),
            ("HITL Trigger Timeline", "timeline", ["date", "triggerType", "reasonCode"]),
            ("Governance Decision Log", "table", ["timestamp", "windowId", "action", "reason"]),
            ("Agent Explanation Panel", "markdown", ["narrative", "topRiskDrivers"]),
            ("Numerical Claim Traceability Table", "table", ["claim", "value", "sourceCollection", "sourceKey"]),
            ("Trigger Reason Breakdown", "bar", ["reason", "count"]),
            ("Before-HITL vs After-HITL Allocation", "bar", ["ticker", "beforeHitlAllocation", "afterHitlAllocation"]),
            ("Turnover Alert Plot", "line", ["date", "turnover", "turnoverThreshold"]),
            ("Governance Rule Compliance Matrix", "table", ["ruleName", "status", "currentValue", "threshold"]),
        ],
    ),
    (
        "Evaluation and Backtesting",
        "/api/analytics/backtesting",
        [
            ("Advisory Portfolio vs Equal Weight Equity Curve", "line", ["date", "advisoryPortfolioValue", "equalWeightValue"]),
            ("Advisory Portfolio vs Standard CVaR Equity Curve", "line", ["date", "advisoryPortfolioValue", "standardCvarValue"]),
            ("Strategy Performance Comparison", "table", ["strategy", "annualReturn", "volatility", "sharpe", "cvar95"]),
            ("Crisis-Regime Drawdown Comparison", "bar", ["strategy", "crisisDrawdown"]),
            ("CVaR Reduction by Sector Universe", "bar", ["sector", "cvarReductionPercent"]),
            ("Rolling Sharpe Ratio Plot", "line", ["date", "rollingSharpe"]),
            ("Ablation Study Comparison", "table", ["model", "annualReturn", "volatility", "sharpe"]),
            ("Transaction Cost Impact Plot", "bar", ["strategy", "costDrag"]),
            ("Turnover Comparison Plot", "bar", ["strategy", "averageTurnover"]),
            ("IS vs OOS Sharpe Plot", "bar", ["universe", "inSampleSharpe", "outOfSampleSharpe"]),
        ],
    ),
]


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_plot_registry() -> list[dict[str, Any]]:
    registry: list[dict[str, Any]] = []
    plot_number = 1
    for tab, endpoint, plots in PLOT_GROUPS:
        for title, chart_type, required_fields in plots:
            status = "missing_data" if tab == "Contagion Graph Analysis" else "ready"
            registry.append(
                {
                    "plot_number": plot_number,
                    "plot_id": f"plot_{plot_number:02d}_{slugify(title)}",
                    "title": title,
                    "tab": tab,
                    "chart_type": chart_type,
                    "endpoint": endpoint,
                    "required_fields": required_fields,
                    "status": status,
                    "explanation_present": True,
                    "advisory_interpretation_present": True,
                    "csv_export_available": chart_type not in {"network", "markdown"},
                    "issue": "Requires institutional holdings data; fallback empty state is allowed." if status == "missing_data" else None,
                }
            )
            plot_number += 1
    return registry


FULL_PLOT_REGISTRY = build_plot_registry()
PLOT_BY_ID = {item["plot_id"]: item for item in FULL_PLOT_REGISTRY}


def registry_audit() -> dict[str, Any]:
    plot_ids = [item["plot_id"] for item in FULL_PLOT_REGISTRY]
    duplicate_ids = sorted({plot_id for plot_id in plot_ids if plot_ids.count(plot_id) > 1})
    return {
        "total_registered": len(FULL_PLOT_REGISTRY),
        "duplicate_plot_ids": duplicate_ids,
        "tabs": [{"tab": tab, "count": sum(1 for item in FULL_PLOT_REGISTRY if item["tab"] == tab)} for tab in TAB_NAMES],
        "registry": FULL_PLOT_REGISTRY,
    }


def select_registry_plots(plot_ids: list[str]) -> list[dict[str, Any]]:
    selected = []
    for priority, plot_id in enumerate(plot_ids, start=1):
        entry = PLOT_BY_ID[plot_id]
        selected.append(
            {
                "plot_id": entry["plot_id"],
                "title": entry["title"],
                "dashboard_tab": entry["tab"],
                "tab": entry["tab"],
                "priority": priority,
                "priority_order": priority,
                "trigger_chips": trigger_chips_for(entry),
                "reason": reason_for(entry),
                "required_fields": entry["required_fields"],
                "endpoint": entry["endpoint"],
                "status": entry["status"],
            }
        )
    return selected


def trigger_chips_for(entry: dict[str, Any]) -> list[str]:
    tab = entry["tab"]
    if tab == "Instability Monitor":
        return ["regime", "threshold", "stress"]
    if tab == "Advisory Diversification":
        return ["weights", "constraints", "reason_codes"]
    if tab == "Diversification Diagnostics":
        return ["hhi", "concentration", "effective_holdings"]
    if tab == "Risk Governance":
        return ["cvar", "drawdown", "traceability"]
    if tab == "Contagion Graph Analysis":
        return ["graph_data", "centrality", "fallback_safe"]
    if tab == "Agent Governance and Explainability":
        return ["hitl", "audit_trail", "claim_sources"]
    if tab == "Evaluation and Backtesting":
        return ["benchmark", "costs", "crisis_period"]
    if tab == "Correlation & Covariance EDA":
        return ["correlation", "covariance", "finite_numeric"]
    return ["data_quality", "returns", "sample_data_chip"]


def reason_for(entry: dict[str, Any]) -> str:
    return f"Shown because {entry['title']} supports advisory governance review in {entry['tab']}."
