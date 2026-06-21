# Report Figure Redistribution Design

## Objective

Move the additional verified result figures out of the detached full-page compendium and place them beside the methodology, analytical argument, result, or governance discussion they directly support. Keep a compact reproducibility manifest, but remove duplicate full-page copies from the appendix.

## Scope

- Update `report/New Project/main.tex` with contextual figure placements and short interpretive transitions.
- Update `report/New Project/generate_result_figure_appendix.py` so it generates a placement manifest rather than a second copy of figures already used in the body.
- Regenerate `report/New Project/generated_result_figures.tex`.
- Preserve the existing quantitative claims, tables, citations, labels, and evidence-family boundaries.
- Do not alter analytical CSV data or regenerate source plots.

## Placement Map

### Research Methodology

- `adaptive_gcvar_evidence_triangle.png`: place with the walk-forward G-CVaR protocol to summarize the relationship between evidence, activation, and untouched-test interpretation.
- `gcvar_implementation_audit.png`: place with validation and reproducibility controls to show the implementation audit chain.

### Analytical Flow and Decision Logic

- `terminal_value_vs_cvar_tradeoff.png`: place with the G-CVaR objective and constraint discussion.
- `time_varying_graph_exposure.png`: place with adaptive graph penalties and threshold-calibration logic.

### Results and Discussion

- `cvar_drawdown_improvement_vs_equal_weight.png`: place with downside-risk interpretation.
- `computed_component_contribution_to_sharpe.png`: place with comparative evaluation and ablation findings.
- `ablation_composite_score_waterfall.png`: pair with component contribution to show incremental system value.
- `U1_performance_diagnostics.png` and `U1_stress_overlay.png`: place as a paired universe case-study figure.
- `U2_performance_diagnostics.png` and `U2_stress_overlay.png`: place as a paired contrasting-universe figure.
- `crisis_only_governance_comparison.png`: place with robustness and crisis-regime interpretation.

### HITL and Governance Evaluation

- `sample_hitl_100000_terminal_value_comparison.png`: place with HITL outcome interpretation.
- `sample_hitl_decision_distribution_and_event_impact.png`: place with intervention frequency and event effects.
- `sample_hitl_ticker_network_risk_vs_adopted_allocation.png`: place with graph-aware allocation review.

## Layout Rules

- Use a single full-width figure when labels need the full text width.
- Use two side-by-side panels only for directly comparable U1/U2 diagnostic pairs or tightly related HITL outputs.
- Keep every plot large enough for axes, legends, and annotations to remain readable in the compiled PDF.
- Add a concise caption identifying the evidence family, analytical purpose, and completed 2014--2025 run.
- Add one short interpretive paragraph around each placement; do not leave figures as decorative inserts.
- Prefer normal floating placement and avoid orphaned subsection headings or figure-only pages.

## Manifest and Appendix Behavior

The generated manifest will list each verified figure, evidence family, and final report location. Figures moved into the body will not be rendered again in the appendix. Any genuinely supplemental figure retained outside the body may still be rendered once after the manifest.

## Verification

1. Run the appendix generator and its report contract tests.
2. Compile `main.tex` from `report/New Project` using the available Tectonic runtime.
3. Confirm there are no missing assets, duplicate LaTeX labels, unresolved references, or unexpected figure-count failures.
4. Render the changed chapter ranges and appendix to images.
5. Visually inspect caption proximity, font size, whitespace, page breaks, and chart legibility.
6. Confirm the final PDF contains one rendered copy of each redistributed figure and a manifest entry pointing to its body location.

## Acceptance Criteria

- All fifteen additional figures have a defensible contextual placement.
- The manifest remains reproducible and names the final placement for every managed figure.
- No redistributed figure is duplicated in the appendix.
- The report compiles successfully and the changed pages have no clipping, overlap, unreadable chart text, or empty heading-only pages caused by the redistribution.
