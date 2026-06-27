# Contextual Report Figure Redistribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Place the fifteen additional verified figures beside the arguments they support, retain a reproducible location manifest, and eliminate duplicate full-page appendix copies.

**Architecture:** `main.tex` owns contextual narrative and body placement. `generate_result_figure_appendix.py` owns the managed-figure inventory and emits a compact manifest that records each figure's destination without rendering body figures again. Existing contract tests enforce complete inventory coverage, unique rendered copies, valid labels, and successful regeneration.

**Tech Stack:** Python 3.13, `unittest`, LaTeX/XeLaTeX-compatible syntax, Tectonic, PyMuPDF visual verification.

---

## File Structure

- Modify `report/New Project/main.tex`: add contextual figure blocks and interpretation in methodology, analytical logic, results, and HITL sections; replace obsolete appendix language.
- Modify `report/New Project/generate_result_figure_appendix.py`: define the placement map and generate the compact manifest without duplicate `\ResultFigure` calls.
- Regenerate `report/New Project/generated_result_figures.tex`: generated placement manifest included by `main.tex`.
- Modify `report/New Project/tests/test_report_contract.py`: verify all managed figures are inventoried, all fifteen redistributed figures appear exactly once in the body, labels are unique, and the generated appendix contains no duplicate renderings.
- Regenerate `report/New Project/main.pdf`: compiled deliverable for visual QA.

### Task 1: Lock the redistribution contract with failing tests

**Files:**
- Modify: `report/New Project/tests/test_report_contract.py`
- Test: `report/New Project/tests/test_report_contract.py`

- [ ] **Step 1: Replace the annex-only figure assertion with inventory and placement assertions**

Add constants for the fifteen redistributed stems and tests equivalent to:

```python
REDISTRIBUTED_STEMS = {
    "adaptive_gcvar_evidence_triangle",
    "gcvar_implementation_audit",
    "terminal_value_vs_cvar_tradeoff",
    "time_varying_graph_exposure",
    "cvar_drawdown_improvement_vs_equal_weight",
    "computed_component_contribution_to_sharpe",
    "ablation_composite_score_waterfall",
    "U1_performance_diagnostics",
    "U1_stress_overlay",
    "U2_performance_diagnostics",
    "U2_stress_overlay",
    "crisis_only_governance_comparison",
    "sample_hitl_100000_terminal_value_comparison",
    "sample_hitl_decision_distribution_and_event_impact",
    "sample_hitl_ticker_network_risk_vs_adopted_allocation",
}

def test_manifest_inventories_every_managed_figure(self):
    annex = ANNEX.read_text(encoding="utf-8")
    for stem in REDISTRIBUTED_STEMS:
        with self.subTest(stem=stem):
            self.assertIn(stem.replace("_", r"\_"), annex)

def test_redistributed_figures_render_once_in_body_and_not_annex(self):
    annex = ANNEX.read_text(encoding="utf-8")
    for stem in REDISTRIBUTED_STEMS:
        with self.subTest(stem=stem):
            self.assertEqual(1, self.text.count(stem + ".png"))
            self.assertNotIn(stem + ".png", annex)

def test_figure_labels_are_unique(self):
    labels = re.findall(r"\\label\{(fig:[^}]+)\}", self.text)
    self.assertEqual(len(labels), len(set(labels)))
```

- [ ] **Step 2: Run the targeted tests and confirm the old structure fails**

Run:

```powershell
python -m unittest report/New` Project/tests/test_report_contract.py -v
```

Expected: the new manifest/placement tests fail because the appendix still renders the managed figures and the body does not yet contain all fifteen.

- [ ] **Step 3: Commit the test contract**

```powershell
git add -- 'report/New Project/tests/test_report_contract.py'
git commit -m 'test: require contextual report figure placement'
```

### Task 2: Generate a compact placement manifest

**Files:**
- Modify: `report/New Project/generate_result_figure_appendix.py`
- Regenerate: `report/New Project/generated_result_figures.tex`
- Test: `report/New Project/tests/test_report_contract.py`

- [ ] **Step 1: Define explicit placement metadata**

Replace the annex render loop with a mapping whose values are stable report destinations:

```python
PLACEMENTS = {
    "adaptive_gcvar_evidence_triangle": "Section 4.8 - Walk-Forward Evaluation Protocol",
    "gcvar_implementation_audit": "Section 4.8 - Walk-Forward Evaluation Protocol",
    "terminal_value_vs_cvar_tradeoff": "Section 6.4 - G-CVaR Formulation",
    "time_varying_graph_exposure": "Section 6.4 - Threshold Calibration",
    "cvar_drawdown_improvement_vs_equal_weight": "Section 9.2 - Primary Protocol",
    "computed_component_contribution_to_sharpe": "Section 9.6 - Comparative Ablation",
    "ablation_composite_score_waterfall": "Section 9.6 - Comparative Ablation",
    "U1_performance_diagnostics": "Section 9.3 - Universe-Specific Results",
    "U1_stress_overlay": "Section 9.3 - Universe-Specific Results",
    "U2_performance_diagnostics": "Section 9.3 - Universe-Specific Results",
    "U2_stress_overlay": "Section 9.3 - Universe-Specific Results",
    "crisis_only_governance_comparison": "Section 9.4 - Crisis Evidence",
    "sample_hitl_100000_terminal_value_comparison": "Section 9.6 - HITL Evaluation",
    "sample_hitl_decision_distribution_and_event_impact": "Section 9.6 - HITL Evaluation",
    "sample_hitl_ticker_network_risk_vs_adopted_allocation": "Section 9.6 - HITL Evaluation",
}
```

- [ ] **Step 2: Emit a four-column manifest and stop rendering body figures**

Generate `No.`, `Evidence family`, `Figure`, and `Report location` columns. Preserve repository-relative discovery and the expected managed-figure count, but remove the `\clearpage`, subsection, and `\ResultFigure` emission loop.

- [ ] **Step 3: Regenerate the manifest**

Run:

```powershell
python 'report/New Project/generate_result_figure_appendix.py'
```

Expected: `generated_result_figures.tex` is rewritten with a manifest and zero `\ResultFigure` commands.

- [ ] **Step 4: Run the manifest tests**

Run:

```powershell
python -m unittest report/New` Project/tests/test_report_contract.py -v
```

Expected: inventory assertions pass; body-placement assertions still fail.

- [ ] **Step 5: Commit the generator and generated output**

```powershell
git add -- 'report/New Project/generate_result_figure_appendix.py' 'report/New Project/generated_result_figures.tex'
git commit -m 'refactor: generate report figure placement manifest'
```

### Task 3: Place methodology and analytical-logic figures

**Files:**
- Modify: `report/New Project/main.tex:883-905`
- Modify: `report/New Project/main.tex:1379-1468`

- [ ] **Step 1: Add the methodology evidence pair after the protocol-boundary discussion**

Insert two readable figure blocks for `adaptive_gcvar_evidence_triangle.png` and `gcvar_implementation_audit.png`, each followed by prose that distinguishes frozen protocol evidence from implementation validation. Use labels `fig:methodology-adaptive-evidence-triangle` and `fig:methodology-gcvar-implementation-audit`.

- [ ] **Step 2: Add the optimisation trade-off figure after the primary objective**

Insert `publication_final_plots/terminal_value_vs_cvar_tradeoff.png` after the paragraph separating the primary and supplemental objectives. Explain that the plot visualizes a return/tail-loss trade-off rather than universal dominance. Use label `fig:analysis-terminal-value-cvar-tradeoff`.

- [ ] **Step 3: Add time-varying exposure after threshold calibration**

Insert `walk_forward_governance_gcvar/time_varying_graph_exposure.png` with label `fig:analysis-time-varying-graph-exposure` and connect it explicitly to the frozen adaptive gate.

- [ ] **Step 4: Run report tests**

```powershell
python -m unittest report/New` Project/tests/test_report_contract.py -v
```

Expected: the four methodology/analysis stems pass their once-in-body assertions; remaining body-placement tests fail.

- [ ] **Step 5: Commit the first body placements**

```powershell
git add -- 'report/New Project/main.tex'
git commit -m 'docs: place methodology and analytical evidence figures'
```

### Task 4: Place results, robustness, and HITL figures

**Files:**
- Modify: `report/New Project/main.tex:2041-2268`

- [ ] **Step 1: Add downside-risk comparison to the primary-protocol interpretation**

Insert `evaluation_style/cvar_drawdown_improvement_vs_equal_weight.png` after the primary win-count discussion. Caption it as a downside comparison against Equal Weight without claiming universal superiority. Use label `fig:results-cvar-drawdown-improvement`.

- [ ] **Step 2: Add U1 and U2 diagnostic pairs to universe-specific analysis**

Create two `figure` environments, each with `minipage` panels at `0.49\textwidth`: U1 performance plus stress overlay, then U2 performance plus stress overlay. Use one caption and one label per universe: `fig:results-u1-diagnostics` and `fig:results-u2-diagnostics`.

- [ ] **Step 3: Add crisis-only governance comparison**

Insert `walk_forward_governance_gcvar/crisis_only_governance_comparison.png` after the adaptive-gate diagnostics. State that it is conditional mechanism evidence, not a replacement for untouched-test metrics. Use label `fig:results-crisis-only-governance`.

- [ ] **Step 4: Add the ablation pair**

Place `publication_final_plots/computed_component_contribution_to_sharpe.png` and `ten_algorithm_ablation/ablation_composite_score_waterfall.png` as a two-panel figure after Table `tab:ablation-metrics-detail`. Use label `fig:results-ablation-contribution-pair` and explain how metric-specific and composite views differ.

- [ ] **Step 5: Add the HITL evidence set**

Create one full-width terminal-value comparison and one paired decision-impact/network-risk figure using the three `hitl_sample_analysis` files. Explicitly label them as simulated governance evidence. Use labels `fig:results-hitl-terminal-value` and `fig:results-hitl-decision-network`.

- [ ] **Step 6: Replace obsolete annex catalogue wording**

Change “appear exactly once in the indexed result-figure annex” to explain that figures appear once at their analytical point of use and are indexed by the placement manifest.

- [ ] **Step 7: Run the full contract test**

```powershell
python -m unittest report/New` Project/tests/test_report_contract.py -v
```

Expected: all report contract tests pass.

- [ ] **Step 8: Commit the results placements**

```powershell
git add -- 'report/New Project/main.tex'
git commit -m 'docs: integrate result and governance figures contextually'
```

### Task 5: Compile and visually verify the final report

**Files:**
- Regenerate: `report/New Project/main.pdf`
- Inspect: `report/New Project/.latex-build/main.log`
- Inspect: `tmp/pdfs/report-figure-redistribution/`

- [ ] **Step 1: Regenerate managed LaTeX inputs and run tests**

```powershell
python 'report/New Project/generate_result_figure_appendix.py'
python -m unittest report/New` Project/tests/test_report_contract.py -v
```

Expected: generator reports the managed figure count and all tests pass.

- [ ] **Step 2: Compile from the report directory**

```powershell
& 'C:\Users\jithe\.codex\.tmp\bundled-marketplaces\openai-bundled\plugins\latex\bin\tectonic.exe' -X compile main.tex --outdir .latex-build --keep-logs --keep-intermediates
Copy-Item -LiteralPath '.latex-build\main.pdf' -Destination 'main.pdf' -Force
```

Run with working directory `report/New Project`. Expected: exit code 0 and an updated `main.pdf`.

- [ ] **Step 3: Check compile diagnostics**

```powershell
rg -n "error|undefined|multiply defined|overfull|underfull" '.latex-build/main.log'
```

Expected: no errors, undefined references, or multiply defined labels; inspect any box warnings near changed figures.

- [ ] **Step 4: Render changed pages for visual inspection**

Use PyMuPDF to render the methodology, analytical-logic, results, HITL, and manifest page ranges to `tmp/pdfs/report-figure-redistribution/`, then assemble contact sheets. Confirm axes and legends are readable, captions remain with figures, paired panels are aligned, and no heading-only page was introduced.

- [ ] **Step 5: Verify single-copy rendering in the PDF**

Extract PDF text and confirm every redistributed caption appears once. Compare the final page count with the 143-page baseline and confirm the removal of duplicate appendix pages reduced or did not inflate the report.

- [ ] **Step 6: Commit the verified report changes**

```powershell
git add -- 'report/New Project/main.tex' 'report/New Project/generate_result_figure_appendix.py' 'report/New Project/generated_result_figures.tex' 'report/New Project/tests/test_report_contract.py' 'report/New Project/main.pdf'
git commit -m 'docs: redistribute verified figures through thesis report'
```
