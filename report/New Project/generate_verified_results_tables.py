from collections import defaultdict
import csv
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess


REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parents[1]
TABLE_DIR = ROOT / "notebook" / "tables_universe_analysis"
FIGURE_DIR = ROOT / "notebook" / "figures_universe_analysis"
OUTPUT = REPORT_DIR / "generated_verified_results_tables.tex"


PRIMARY_STRATEGIES = {
    "cvar_optimized": "Standard CVaR",
    "graph_cvar_optimized": "Static G-CVaR",
    "adaptive_graph_cvar": "Adaptive G-CVaR",
}


def read_csv(name: str):
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required evidence file is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    value = row.get(key, "")
    if value in (None, ""):
        raise ValueError(f"Missing required metric {key!r} in row {row}")
    return float(value)


def mean(values):
    values = list(values)
    if not values:
        raise ValueError("Cannot calculate a mean from an empty sequence")
    return sum(values) / len(values)


def tex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def strategy_means(rows):
    grouped = defaultdict(list)
    for row in rows:
        strategy = row.get("strategy")
        if strategy in PRIMARY_STRATEGIES:
            grouped[strategy].append(row)
    if set(grouped) != set(PRIMARY_STRATEGIES):
        raise ValueError("Authoritative primary strategy rows are incomplete")
    if any(len(rows_for_strategy) != 11 for rows_for_strategy in grouped.values()):
        raise ValueError("Expected 11 authoritative rows per primary strategy")
    metrics = [
        "annual_return",
        "sharpe_ratio",
        "historical_cvar_loss_95",
        "max_drawdown_magnitude",
        "graph_exposure",
        "turnover",
    ]
    return {
        strategy: {
            metric: mean(number(row, metric) for row in rows_for_strategy)
            for metric in metrics
        }
        for strategy, rows_for_strategy in grouped.items()
    }


def rows_by_universe(rows, strategy):
    selected = {
        row["universe"]: row for row in rows if row.get("strategy") == strategy
    }
    if len(selected) != 11:
        raise ValueError(f"Expected 11 universes for {strategy}, found {len(selected)}")
    return selected


def pairwise_wins(rows, left_strategy, right_strategy):
    left = rows_by_universe(rows, left_strategy)
    right = rows_by_universe(rows, right_strategy)
    definitions = [
        ("Annual return", "annual_return", True),
        ("Sharpe ratio", "sharpe_ratio", True),
        ("CVaR loss", "historical_cvar_loss_95", False),
        ("Maximum drawdown", "max_drawdown_magnitude", False),
    ]
    result = []
    for label, metric, higher_is_better in definitions:
        wins = 0
        for universe in left:
            left_value = number(left[universe], metric)
            right_value = number(right[universe], metric)
            wins += int(
                left_value > right_value
                if higher_is_better
                else left_value < right_value
            )
        result.append((label, wins))
    return result


def build_tables() -> str:
    coverage = read_csv("universe_data_coverage_audit_2014_2025.csv")
    boundary = read_csv("calibration_vs_test_boundary_audit.csv")
    primary = read_csv("gcvar_test_governance_ranking.csv")
    gate = read_csv("gcvar_adaptive_gate_audit_test.csv")
    v2_results = read_csv("adaptive_graph_cvar_v2_results.csv")
    v2_activation = read_csv("adaptive_graph_cvar_v2_activation_summary.csv")
    checks = read_csv("final_technical_validation_checks.csv")
    rejections = read_csv("nan_safe_governance_rejections.csv")
    core_ranking = read_csv("nan_safe_core_governance_ranking.csv")
    supplemental_ranking = read_csv("nan_safe_supplemental_governance_ranking.csv")
    hitl_ranking = read_csv("nan_safe_hitl_simulation_ranking.csv")
    plot_qa = read_csv("plot_quality_audit.csv")

    if len(coverage) != 11:
        raise ValueError(f"Expected 11 coverage rows, found {len(coverage)}")
    requested_assets = sum(int(row["requested_tickers"]) for row in coverage)
    available_assets = sum(int(row["available_tickers"]) for row in coverage)
    if requested_assets != 197 or available_assets != 197:
        raise ValueError("Verified coverage must contain all 197 requested assets")
    if any(str(row["whether_test_used_in_calibration"]).lower() != "false" for row in boundary):
        raise ValueError("Calibration/test boundary audit reports test-data use")
    if len(v2_results) != 11 or len(gate) != 132:
        raise ValueError("Primary/V2 universe or decision coverage is incomplete")
    if sum(int(row["rebalance_count"]) for row in v2_activation) != 132:
        raise ValueError("V2 activation summary does not cover 132 decisions")
    if len(core_ranking) != 110 or len(supplemental_ranking) != 11:
        raise ValueError("NaN-safe ranking counts do not match the completed run")
    if hitl_ranking:
        raise ValueError("HITL ranking should be empty when governance metrics are incomplete")
    if len(rejections) != 22:
        raise ValueError("Expected 22 explicit NaN-safe ranking rejections")
    if len(plot_qa) != 121 or any(row["qa_status"] != "pass" for row in plot_qa):
        raise ValueError("All 121 plot-quality rows must pass")
    if len(list(FIGURE_DIR.rglob("*.png"))) != 121:
        raise ValueError("The complete result figure directory must contain 121 PNG files")

    check_map = {
        row["check"]: str(row["passed"]).lower() == "true" for row in checks
    }
    expected_false = {"v2_activation_nonzero"}
    actual_false = {name for name, passed in check_map.items() if not passed}
    if actual_false != expected_false:
        raise ValueError(f"Unexpected technical validation failures: {actual_false}")

    means = strategy_means(primary)
    adaptive_rows = rows_by_universe(primary, "adaptive_graph_cvar")
    adaptive_vs_cvar = pairwise_wins(
        primary, "adaptive_graph_cvar", "cvar_optimized"
    )
    adaptive_vs_static = pairwise_wins(
        primary, "adaptive_graph_cvar", "graph_cvar_optimized"
    )
    primary_active = sum(number(row, "active_graph_lambda") > 0 for row in gate)
    active_universes = sorted(
        {row["universe"] for row in gate if number(row, "active_graph_lambda") > 0},
        key=lambda value: int(value[1:]),
    )
    active_decisions_by_universe = {
        universe: sum(
            number(row, "active_graph_lambda") > 0
            for row in gate
            if row["universe"] == universe
        )
        for universe in adaptive_rows
    }

    v2_metrics = {
        metric: mean(number(row, metric) for row in v2_results)
        for metric in [
            "annual_return",
            "sharpe_ratio",
            "historical_cvar_loss_95",
            "max_drawdown_magnitude",
            "turnover",
            "active_frequency",
        ]
    }

    rejection_counts = defaultdict(int)
    for row in rejections:
        rejection_counts[row["strategy"]] += 1

    lines = [
        "% Generated by generate_verified_results_tables.py; do not edit manually.",
        r"\subsection{Executive empirical result summary}",
        r"\label{sec:executive-empirical-summary}",
        (
            "The completed evidence supports a selective rather than universal conclusion. "
            "Adaptive G-CVaR delivered stronger return and Sharpe outcomes than Standard CVaR "
            "in eight of eleven universes, while Standard CVaR retained better CVaR-loss and "
            "maximum-drawdown outcomes in most universes. The primary adaptive gate activated "
            f"in {primary_active} of 132 decisions across {len(active_universes)} universes. "
            "Supplemental V2 completed every decision without solver fallback, but its "
            "pre-registered activation threshold was not crossed in the untouched test."
        ),
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Experiment-lane interpretation hierarchy}",
        r"\label{tab:experiment-lanes}",
        r"\begin{tabular}{L{3.0cm}L{4.0cm}L{6.0cm}}",
        r"\toprule",
        r"\textbf{Lane} & \textbf{Strategies} & \textbf{Permitted interpretation} \\",
        r"\midrule",
        r"Authoritative primary & Standard CVaR, Static G-CVaR, Adaptive G-CVaR & Main untouched-test claims under the quadratic graph protocol \\",
        r"Core comparison & Ten complete-metric strategies & NaN-safe within-universe governance comparison \\",
        r"Supplemental & Linear-centrality Adaptive G-CVaR V2 & Separate extension; not a replacement for the primary objective \\",
        r"HITL simulation & Sample governed allocation adjustments & Operational simulation, not a mathematical optimizer ranking \\",
        r"Legacy tournament & Earlier broad algorithm comparison & Historical context only; never merged with authoritative estimates \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Verified data coverage and walk-forward boundaries}",
        r"\label{tab:verified-data-boundaries}",
        r"\begin{tabular}{L{4.0cm}L{3.0cm}L{6.0cm}}",
        r"\toprule",
        r"\textbf{Item} & \textbf{Verified value} & \textbf{Use} \\",
        r"\midrule",
        f"Assets and universes & {available_assets}/{requested_assets} assets; 11 universes & Complete requested panel " + r"\\",
        r"Price/return coverage & 2014-01-02 to 2025-12-31 & 3,018 price rows and 3,017 return rows per universe \\",
        r"Training & 2014--2019 & Model and indicator estimation \\",
        r"Validation & 2020--2022 & Parameter and gate calibration \\",
        r"Untouched test & 2023--2025 & Final reporting only \\",
        r"Boundary audit & Test used in calibration: false & Verified for all universes \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Authoritative untouched-test primary-protocol means across 11 universes}",
        r"\label{tab:authoritative-primary-means}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\textbf{Strategy} & \textbf{Return} & \textbf{Sharpe} & \textbf{CVaR loss} & \textbf{MaxDD} & \textbf{Graph exp.} & \textbf{Turnover} \\",
        r"\midrule",
    ]
    for strategy in ["cvar_optimized", "graph_cvar_optimized", "adaptive_graph_cvar"]:
        values = means[strategy]
        ann_ret = f"{values['annual_return']:.2%}".replace("%", r"\%")
        cvar_l = f"{values['historical_cvar_loss_95']:.3%}".replace("%", r"\%")
        max_dd = f"{values['max_drawdown_magnitude']:.2%}".replace("%", r"\%")
        lines.append(
            f"{PRIMARY_STRATEGIES[strategy]} & {ann_ret} & "
            f"{values['sharpe_ratio']:.3f} & {cvar_l} & "
            f"{max_dd} & {values['graph_exposure']:.3f} & "
            f"{values['turnover']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            (
                r"\noindent\textit{Source:} "
                r"\texttt{gcvar\_test\_governance\_ranking.csv}."
            ),
            r"\begin{table}[htbp]",
            r"\centering",
            r"\scriptsize",
            r"\caption{Adaptive G-CVaR untouched-test results by universe}",
            r"\label{tab:adaptive-by-universe}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"\textbf{Universe} & \textbf{Return} & \textbf{Sharpe} & \textbf{CVaR loss} & \textbf{MaxDD} & \textbf{Graph exp.} & \textbf{Turnover} & \textbf{Active decisions} " + r"\\",
            r"\midrule",
        ]
    )
    for universe in sorted(adaptive_rows, key=lambda value: int(value[1:])):
        row = adaptive_rows[universe]
        ann_ret = f"{number(row, 'annual_return'):.2%}".replace("%", r"\%")
        cvar_l = f"{number(row, 'historical_cvar_loss_95'):.3%}".replace("%", r"\%")
        max_dd = f"{number(row, 'max_drawdown_magnitude'):.2%}".replace("%", r"\%")
        lines.append(
            f"{universe} & {ann_ret} & "
            f"{number(row, 'sharpe_ratio'):.3f} & "
            f"{cvar_l} & "
            f"{max_dd} & "
            f"{number(row, 'graph_exposure'):.3f} & "
            f"{number(row, 'turnover'):.3f} & "
            f"{active_decisions_by_universe[universe]}/12 " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Adaptive G-CVaR pairwise universe wins in the authoritative protocol}",
            r"\label{tab:adaptive-pairwise-wins}",
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"\textbf{Metric} & \textbf{vs Standard CVaR} & \textbf{vs Static G-CVaR} \\",
            r"\midrule",
        ]
    )
    static_lookup = dict(adaptive_vs_static)
    for metric, win_count in adaptive_vs_cvar:
        lines.append(
            f"{metric} & {win_count}/11 & {static_lookup[metric]}/11 " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Supplemental Linear-Centrality Adaptive G-CVaR V2 means}",
            r"\label{tab:v2-means}",
            r"\begin{tabular}{rrrrrr}",
            r"\toprule",
            r"\textbf{Return} & \textbf{Sharpe} & \textbf{CVaR loss} & \textbf{MaxDD} & \textbf{Turnover} & \textbf{Activation} \\",
            r"\midrule",
            (
                f"{v2_metrics['annual_return']:.2%}".replace("%", r"\%") + " & " +
                f"{v2_metrics['sharpe_ratio']:.3f} & " +
                f"{v2_metrics['historical_cvar_loss_95']:.3%}".replace("%", r"\%") + " & " +
                f"{v2_metrics['max_drawdown_magnitude']:.2%}".replace("%", r"\%") + " & " +
                f"{v2_metrics['turnover']:.3f} & " +
                f"{v2_metrics['active_frequency']:.1%}".replace("%", r"\%") + r" \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            (
                r"\noindent\textit{Source:} "
                r"\texttt{adaptive\_graph\_cvar\_v2\_results.csv}; "
                r"activation from \texttt{adaptive\_graph\_cvar\_v2\_activation\_summary.csv}."
            ),
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{NaN-safe ranking eligibility and explicit rejection audit}",
            r"\label{tab:nan-safe-ranking-audit}",
            r"\begin{tabular}{L{5.5cm}rrL{4.5cm}}",
            r"\toprule",
            r"\textbf{Family/strategy} & \textbf{Eligible} & \textbf{Rejected} & \textbf{Interpretation} \\",
            r"\midrule",
            f"Core strategies & {len(core_ranking)} & 0 & Complete governance metrics " + r"\\",
            f"Supplemental V2 & {len(supplemental_ranking)} & 0 & Ranked only in supplemental family " + r"\\",
            f"Fixed Quarterly G-CVaR & 0 & {rejection_counts['fixed_quarterly_graph_cvar']} & Missing HHI, effective N, graph exposure " + r"\\",
            f"Sample HITL G-CVaR & 0 & {rejection_counts['sample_hitl_governed_adaptive_gcvar']} & Missing turnover, HHI, effective N, graph exposure " + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Reproducibility and execution environment}",
            r"\label{tab:reproducibility-environment}",
            r"\begin{tabular}{L{4.2cm}L{8.8cm}}",
            r"\toprule",
            r"\textbf{Item} & \textbf{Recorded value} \\",
            r"\midrule",
            f"Python & {tex_escape(platform.python_version())} " + r"\\",
            f"pandas / NumPy & {tex_escape(package_version('pandas'))} / {tex_escape(package_version('numpy'))} " + r"\\",
            f"CVXPY / CLARABEL & {tex_escape(package_version('cvxpy'))} / {tex_escape(package_version('clarabel'))} " + r"\\",
            f"yfinance / Matplotlib & {tex_escape(package_version('yfinance'))} / {tex_escape(package_version('matplotlib'))} " + r"\\",
            "Git commit at report generation & " + rf"\texttt{{{tex_escape(git_commit())}}} " + r"\\",
            r"Analysis entry point & \path{notebook/agentic_ai_portfolio_governance_final_repaired_full_pro (1).py} \\",
            r"Data window & 2014-01-02 through 2025-12-31 \\",
            r"Result images & 121 PNG files; all plot-quality audit rows passed \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            r"\noindent\textbf{Technical validation audit.} Nine of ten boolean checks passed. The sole false row is \path{v2_activation_nonzero}, which records the observed zero-activation result rather than an implementation failure. Source: \path{final_technical_validation_checks.csv}.",
            r"\noindent\textbf{Complete visual evidence.} The indexed annex contains 121 figures and is generated from \path{plot_quality_audit.csv} plus the complete result-figure directory.",
            r"\medskip",
            r"\noindent\textit{Primary evidence files:} \path{universe_data_coverage_audit_2014_2025.csv}, \path{calibration_vs_test_boundary_audit.csv}, \path{gcvar_test_governance_ranking.csv}, \path{adaptive_graph_cvar_v2_results.csv}, \path{nan_safe_governance_rejections.csv}, and \path{final_technical_validation_checks.csv}.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUTPUT.write_text(build_tables(), encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT}")
