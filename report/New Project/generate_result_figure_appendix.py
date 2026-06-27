from pathlib import Path
from collections import Counter
import re


REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parents[1]
FIGURE_ROOT = ROOT / "notebook" / "figures_universe_analysis"
OUTPUT = REPORT_DIR / "generated_result_figures.tex"
EXPECTED_FIGURES = 24

ALLOWED_STEMS = {
    "adaptive_gcvar_evidence_triangle",
    "adaptive_lambda_diagnostics_grid",
    "gcvar_implementation_audit",
    "terminal_value_vs_cvar_tradeoff",
    "all_universes_gcvar_frontier_summary",
    "computed_ablation_summary",
    "cvar_drawdown_improvement_vs_equal_weight",
    "risk_return_governance_map",
    "adaptive_gcvar_rank_heatmap",
    "adaptive_gcvar_valid_superiority_check",
    "sample_hitl_100000_terminal_value_comparison",
    "sample_hitl_decision_distribution_and_event_impact",
    "sample_hitl_ticker_network_risk_vs_adopted_allocation",
    "aggregate_cumulative_performance_log_scale",
    "bootstrap_sharpe_confidence_intervals",
    "computed_component_contribution_to_sharpe",
    "monthly_return_distribution_cvar_vs_adaptive_gcvar",
    "ablation_composite_score_waterfall",
    "U1_performance_diagnostics",
    "U1_stress_overlay",
    "U2_performance_diagnostics",
    "U2_stress_overlay",
    "crisis_only_governance_comparison",
    "time_varying_graph_exposure",
}

REPORT_LOCATIONS = {
    "adaptive_gcvar_evidence_triangle": "Research Methodology -- walk-forward G-CVaR protocol",
    "gcvar_implementation_audit": "Research Methodology -- validation and reproducibility controls",
    "terminal_value_vs_cvar_tradeoff": "Analytical Flow and Decision Logic -- G-CVaR objective and constraints",
    "time_varying_graph_exposure": "Analytical Flow and Decision Logic -- adaptive graph penalties",
    "cvar_drawdown_improvement_vs_equal_weight": "Results and Discussion -- downside-risk interpretation",
    "computed_component_contribution_to_sharpe": "Results and Discussion -- comparative evaluation and ablation",
    "ablation_composite_score_waterfall": "Results and Discussion -- comparative evaluation and ablation",
    "U1_performance_diagnostics": "Results and Discussion -- U1 universe case study",
    "U1_stress_overlay": "Results and Discussion -- U1 universe case study",
    "U2_performance_diagnostics": "Results and Discussion -- U2 contrasting-universe case study",
    "U2_stress_overlay": "Results and Discussion -- U2 contrasting-universe case study",
    "crisis_only_governance_comparison": "Results and Discussion -- robustness and crisis regimes",
    "sample_hitl_100000_terminal_value_comparison": "HITL and Governance Evaluation -- outcome interpretation",
    "sample_hitl_decision_distribution_and_event_impact": "HITL and Governance Evaluation -- intervention frequency and event effects",
    "sample_hitl_ticker_network_risk_vs_adopted_allocation": "HITL and Governance Evaluation -- graph-aware allocation review",
    "adaptive_lambda_diagnostics_grid": "Results and Discussion -- Adaptive Gate Behaviour and Crisis Evidence",
    "all_universes_gcvar_frontier_summary": "Results and Discussion -- Graph-Penalty Frontier and Structural Trade-off",
    "computed_ablation_summary": "Results and Discussion -- Comprehensive Comparative Evaluation and Ablation Studies",
    "risk_return_governance_map": "Results and Discussion -- Comprehensive Comparative Evaluation and Ablation Studies",
    "adaptive_gcvar_rank_heatmap": "Results and Discussion -- Interpretation of the Authoritative Primary Protocol",
    "adaptive_gcvar_valid_superiority_check": "Results and Discussion -- Interpretation of the Authoritative Primary Protocol",
    "aggregate_cumulative_performance_log_scale": "Results and Discussion -- Comprehensive Comparative Evaluation and Ablation Studies",
    "bootstrap_sharpe_confidence_intervals": "Results and Discussion -- Comprehensive Comparative Evaluation and Ablation Studies",
    "monthly_return_distribution_cvar_vs_adaptive_gcvar": "Results and Discussion -- Comprehensive Comparative Evaluation and Ablation Studies",
}


def natural_key(value: str):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_\allowbreak{}",
        "$": r"\$",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def label_from_path(path: Path) -> str:
    clean = re.sub(
        r"[^a-z0-9]+", "-", path.with_suffix("").as_posix().lower()
    ).strip("-")
    return f"fig:results:{clean}"


def figure_source(path: Path) -> str:
    return (
        "../../notebook/figures_universe_analysis/"
        + path.relative_to(FIGURE_ROOT).as_posix()
    )


def title_from_name(value: str) -> str:
    if value == "root":
        return "Universe-level diagnostics"
    cleaned = re.sub(r"[_-]+", " ", Path(value).stem).strip()
    return cleaned[:1].upper() + cleaned[1:]


def build_annex() -> str:
    pngs = [
        png for png in sorted(
            FIGURE_ROOT.rglob("*.png"),
            key=lambda path: natural_key(path.relative_to(FIGURE_ROOT).as_posix()),
        )
        if png.stem in ALLOWED_STEMS
    ]
    stem_counts = Counter(png.stem for png in pngs)
    discovered_stems = set(stem_counts)
    missing = sorted(ALLOWED_STEMS - discovered_stems, key=natural_key)
    unexpected = sorted(discovered_stems - ALLOWED_STEMS, key=natural_key)
    duplicates = sorted(
        (stem for stem, count in stem_counts.items() if count > 1),
        key=natural_key,
    )
    if (
        len(pngs) != EXPECTED_FIGURES
        or missing
        or unexpected
        or duplicates
    ):
        raise RuntimeError(
            f"Expected exactly {EXPECTED_FIGURES} curated result figures; "
            f"found {len(pngs)}. Missing stems: {missing or 'none'}; "
            f"unexpected stems: {unexpected or 'none'}; "
            f"duplicate stems: {duplicates or 'none'}"
        )
    if set(REPORT_LOCATIONS) != ALLOWED_STEMS:
        raise RuntimeError("Report placement metadata must cover every managed figure")

    lines = [
        "% Generated by generate_result_figure_appendix.py; do not edit manually.",
        r"\section{Managed Result Figure Manifest}",
        r"\label{sec:complete-result-figures}",
        (
            "This manifest inventories 24 curated, managed result figures from the "
            "completed analysis run and records where each is interpreted in "
            "the report. It is not an inventory of every embedded report image."
        ),
        r"\subsubsection*{Indexed Figure Manifest}",
        (
            "The manifest provides a compact reproducibility inventory of evidence "
            "families, exact figure stems, and final report locations."
        ),
        r"\small",
        r"\begin{longtable}{rL{3.2cm}L{4.6cm}L{5.2cm}}",
        r"\toprule",
        r"\textbf{No.} & \textbf{Evidence family} & \textbf{Figure filename/stem} & \textbf{Report location} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{No.} & \textbf{Evidence family} & \textbf{Figure filename/stem} & \textbf{Report location} \\",
        r"\midrule",
        r"\endhead",
    ]

    for number, png in enumerate(pngs, start=1):
        rel = png.relative_to(FIGURE_ROOT)
        group = rel.parts[0] if len(rel.parts) > 1 else "root"
        lines.append(
            f"{number} & {tex_escape(title_from_name(group))} & "
            f"{tex_escape(png.stem)} & {tex_escape(REPORT_LOCATIONS[png.stem])} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{longtable}", r"\addtocounter{table}{-1}", r"\normalsize"])

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUTPUT.write_text(build_annex(), encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT} with {EXPECTED_FIGURES} figures")

