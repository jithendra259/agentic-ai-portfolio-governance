# Thesis Report Evidence and Implementation Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the thesis into a compiled, professional, code-audited, evidence-backed report that embeds all 121 generated result figures and accurately distinguishes the authoritative quadratic G-CVaR protocol from legacy, HITL, and supplemental V2 evidence.

**Architecture:** Preserve the existing single-document thesis structure while replacing stale claims in place. Add a deterministic report-local appendix generator that converts the current result-figure tree into a stable LaTeX include file, and add contract tests that verify dates, protocol separation, implementation coverage, removed claims, figure completeness, and report-local path portability before compilation.

**Tech Stack:** LaTeX/Tectonic, Python 3 standard library, `unittest`, `pypdf`, Poppler when available, existing notebook CSV/PNG artifacts.

---

## File structure

- Modify: `report/New Project/main.tex` — authoritative thesis source, narrative, equations, tables, implementation audit, conclusions, and appendix inclusion.
- Create: `report/New Project/generate_result_figure_appendix.py` — deterministic scanner and LaTeX annex generator for all result PNGs.
- Create: `report/New Project/generated_result_figures.tex` — generated grouped annex referenced by `main.tex`.
- Create: `report/New Project/generate_verified_results_tables.py` — deterministic CSV-to-LaTeX evidence-table generator.
- Create: `report/New Project/generated_verified_results_tables.tex` — generated executive results, lane, reproducibility, and audit tables.
- Create: `report/New Project/tests/test_report_contract.py` — report evidence, path, and figure-completeness checks.
- Generate: `report/New Project/.latex-build/main.pdf` — compiled PDF artifact.

### Task 1: Add report contract tests

**Files:**
- Create: `report/New Project/tests/test_report_contract.py`
- Test: `report/New Project/tests/test_report_contract.py`

- [ ] **Step 1: Write the failing report contract tests**

Create tests that read `main.tex`, the generated figure include, and the notebook artifact tree:

```python
from pathlib import Path
import re
import unittest


REPORT_DIR = Path(__file__).resolve().parents[1]
ROOT = REPORT_DIR.parents[1]
MAIN = REPORT_DIR / "main.tex"
ANNEX = REPORT_DIR / "generated_result_figures.tex"
FIGURE_ROOT = ROOT / "notebook" / "figures_universe_analysis"


class ThesisReportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = MAIN.read_text(encoding="utf-8")

    def test_verified_study_period_and_protocol_boundaries_are_present(self):
        self.assertIn("2014--2025", self.text)
        self.assertIn("2014--2019", self.text)
        self.assertIn("2020--2022", self.text)
        self.assertIn("2023--2025", self.text)
        self.assertNotIn("Study Period: 2012--2025", self.text)
        self.assertNotIn("2005--2025", self.text)

    def test_primary_and_supplemental_objectives_are_distinct(self):
        self.assertIn("Static G-CVaR: fixed graph-aware tail-risk optimizer", self.text)
        self.assertIn("Adaptive G-CVaR: instability-gated graph-aware tail-risk optimizer", self.text)
        self.assertIn("Supplemental Linear-Centrality Adaptive G-CVaR", self.text)
        self.assertRegex(self.text, r"w\^\\top A_t w")
        self.assertRegex(self.text, r"c_t\^\\top w")

    def test_verified_implementation_components_are_documented(self):
        required = [
            "React 19", "Vite 8", "MUI X Chat", "FastAPI", "LangGraph",
            "MongoDB", "Supabase PostgreSQL", "NDJSON", "advisory-only",
            "/chat/stream", "/api/analytics", "/api/governance/decision",
        ]
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, self.text)

    def test_unsupported_fixed_claims_are_removed(self):
        self.assertNotIn("38.1\\%", self.text)
        self.assertNotIn("25.9\\% CVaR", self.text)
        self.assertNotIn("statistically significant reductions in crisis-period tail risk", self.text)

    def test_report_uses_portable_graphic_paths(self):
        self.assertNotRegex(self.text, r"[A-Z]:/")
        self.assertNotRegex(self.text, r"[A-Z]:\\\\")

    def test_every_result_png_is_referenced_once(self):
        self.assertTrue(ANNEX.exists())
        annex = ANNEX.read_text(encoding="utf-8")
        pngs = sorted(FIGURE_ROOT.rglob("*.png"))
        self.assertEqual(121, len(pngs))
        references = re.findall(r"\\ResultFigure\{([^}]+)\}", annex)
        self.assertEqual(len(pngs), len(references))
        self.assertEqual(len(references), len(set(references)))
        expected = {
            "../../notebook/figures_universe_analysis/" + p.relative_to(FIGURE_ROOT).as_posix()
            for p in pngs
        }
        self.assertEqual(expected, set(references))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the contract tests and verify failure**

Run:

```powershell
& '.\venv\Scripts\python.exe' -m unittest discover -s 'report\New Project\tests' -v
```

Expected: failures for stale dates/claims, missing implementation terms, non-portable path, and absent generated annex.

- [ ] **Step 3: Commit the failing contract tests**

```powershell
git add -- 'report/New Project/tests/test_report_contract.py'
git commit -m "test: define thesis report evidence contract"
```

### Task 2: Generate the complete result-figure annex

**Files:**
- Create: `report/New Project/generate_result_figure_appendix.py`
- Create: `report/New Project/generated_result_figures.tex`
- Modify: `report/New Project/main.tex:1-90`
- Test: `report/New Project/tests/test_report_contract.py`

- [ ] **Step 1: Add the deterministic annex generator**

Implement a standard-library script with these exact responsibilities:

```python
from pathlib import Path
import re


REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parents[1]
FIGURE_ROOT = ROOT / "notebook" / "figures_universe_analysis"
OUTPUT = REPORT_DIR / "generated_result_figures.tex"


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", value)]


def tex_escape(value: str) -> str:
    replacements = {
        "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def title_from_name(value: str) -> str:
    cleaned = re.sub(r"[_-]+", " ", Path(value).stem).strip()
    return cleaned[:1].upper() + cleaned[1:]


def label_from_path(path: Path) -> str:
    return "fig:results:" + re.sub(r"[^a-z0-9]+", "-", path.as_posix().lower()).strip("-")


def build_annex() -> str:
    pngs = sorted(FIGURE_ROOT.rglob("*.png"), key=lambda p: natural_key(p.relative_to(FIGURE_ROOT).as_posix()))
    if len(pngs) != 121:
        raise RuntimeError(f"Expected 121 result figures, found {len(pngs)}")
    lines = [
        "% Generated by generate_result_figure_appendix.py; do not edit manually.",
        r"\section{Complete Generated Result Figure Compendium}",
        r"\label{app:complete-result-figures}",
        r"This annex embeds every PNG produced by the verified analysis run. Automated image-quality validation passed for all 121 files; economic interpretation remains tied to the corresponding CSV evidence and discussion in the Results chapter.",
    ]
    current_group = None
    for png in pngs:
        rel = png.relative_to(FIGURE_ROOT)
        group = rel.parts[0] if len(rel.parts) > 1 else "root"
        if group != current_group:
            lines.extend([r"\clearpage", rf"\subsection{{{tex_escape(title_from_name(group))}}}"])
            current_group = group
        source = "../../notebook/figures_universe_analysis/" + rel.as_posix()
        caption = title_from_name(rel.name) + ". Source: completed 2014--2025 analysis run."
        lines.append(rf"\ResultFigure{{{source}}}{{{tex_escape(caption)}}}{{{label_from_path(rel)}}}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUTPUT.write_text(build_annex(), encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT}")
```

- [ ] **Step 2: Add the reusable figure macro and portable graphics paths**

In `main.tex` preamble, remove machine-specific `D:/...` and `C:/...` paths. Keep report-local paths and add:

```tex
\graphicspath{{./}{images/}{images/uploads/}{../../notebook/figures_universe_analysis/}}

\newcommand{\ResultFigure}[3]{%
  \begin{figure}[p]
    \centering
    \includegraphics[width=0.96\textwidth,height=0.82\textheight,keepaspectratio]{#1}
    \caption{#2}
    \label{#3}
  \end{figure}
  \clearpage
}
```

- [ ] **Step 3: Run the generator**

```powershell
& '.\venv\Scripts\python.exe' 'report\New Project\generate_result_figure_appendix.py'
```

Expected: `generated_result_figures.tex` contains 121 unique `\ResultFigure` calls.

- [ ] **Step 4: Add the annex include near the end of `main.tex`**

Insert before References:

```tex
\clearpage
\appendix
\input{generated_result_figures.tex}
```

- [ ] **Step 5: Run the figure-completeness test**

```powershell
& '.\venv\Scripts\python.exe' -m unittest 'report\New Project\tests\test_report_contract.py' -v
```

Expected: the figure-reference test passes; narrative contract tests remain failing.

- [ ] **Step 6: Commit the annex tooling and generated source**

```powershell
git add -- 'report/New Project/generate_result_figure_appendix.py' 'report/New Project/generated_result_figures.tex' 'report/New Project/main.tex'
git commit -m "docs: add complete result figure annex"
```

### Task 3: Correct front matter and methodology

**Files:**
- Modify: `report/New Project/main.tex:90-190`
- Modify: `report/New Project/main.tex:790-853`
- Modify: `report/New Project/main.tex:1188-1415`
- Create: `report/New Project/generate_verified_results_tables.py`
- Create: `report/New Project/generated_verified_results_tables.tex`
- Test: `report/New Project/tests/test_report_contract.py`

- [ ] **Step 1: Correct the front matter**

Replace the title-page study period with `2014--2025`. Rewrite the Abstract to state:

- 197 assets and 11 universes;
- training 2014--2019, validation 2020--2022, untouched testing 2023--2025;
- quadratic Static and Adaptive G-CVaR as primary models;
- linear-centrality V2 as supplemental;
- stronger return/Sharpe behavior in many universes without universal CVaR/drawdown dominance;
- 7/132 primary adaptive activations and 0/132 V2 threshold activations;
- frontend/backend governance architecture and advisory-only posture.

Remove the unverified 38.1% and statistical-significance statements.

Replace the manually typed Table of Contents, List of Figures, and List of Tables---whose page numbers will become stale after the complete annex is added---with LaTeX-generated navigation:

```tex
\clearpage
\tableofcontents
\clearpage
\listoffigures
\addcontentsline{toc}{section}{List of Figures}
\clearpage
\listoftables
\addcontentsline{toc}{section}{List of Tables}
```

Retain the nomenclature and implementation-artifact material after the automatic lists.

- [ ] **Step 2: Add the evidence hierarchy and date-boundary table**

In `Research Methodology`, insert a subsection titled `Walk-Forward Governance G-CVaR Evaluation Protocol` with this table:

```tex
\begin{table}[htbp]
\centering
\caption{Strict estimation, calibration, and untouched-test boundaries}
\label{tab:protocol-boundaries}
\begin{tabular}{lll}
\toprule
\textbf{Lane} & \textbf{Dates} & \textbf{Permitted use} \\
\midrule
Training & 2014--2019 & Estimation of historical models and indicators \\
Validation & 2020--2022 & Parameter and threshold selection \\
Untouched test & 2023--2025 & Final reporting only \\
\bottomrule
\end{tabular}
\end{table}
```

State that `whether_test_used_in_calibration=False` for all universes.

- [ ] **Step 3: Correct the mathematical model definitions**

Present the primary graph term exactly as `\lambda_t w^\top A_t w`. Define:

```tex
\noindent\textbf{Static G-CVaR: fixed graph-aware tail-risk optimizer.}

\noindent\textbf{Adaptive G-CVaR: instability-gated graph-aware tail-risk optimizer.}
```

Add a separate V2 paragraph and equation with `\lambda_t c_t^\top w`, labelled supplemental and correlation-proxy-based. State that activation is evaluated as the multiplier exceeding 0.5, not the effective lambda exceeding an impossible absolute value.

- [ ] **Step 4: Add data coverage and ranking eligibility**

Document 197/197 available assets, 3,018 price rows, 3,017 return rows, all 11 universes, NaN-safe family separation, and explicit rejection of incomplete fixed-quarterly/HITL rows from governance ranking.

Add `generate_verified_results_tables.py` so data coverage, experiment lanes, authoritative means, pairwise wins, primary/V2 activation, technical checks, rejection counts, and reproducibility metadata are read directly from the completed CSV artifacts and emitted into `generated_verified_results_tables.tex`. The generated source must identify each CSV input and expose `\input{generated_verified_results_tables.tex}` blocks for the methodology and Results chapter. It must fail if required files, universes, metrics, or validation checks are absent instead of silently emitting incomplete evidence.

The generated evidence must include:

- an executive result summary separating strengths, limitations, and non-activation findings;
- an experiment-lane table distinguishing authoritative, core, supplemental, HITL, and legacy outputs;
- a reproducibility table containing Python/package versions, CVXPY/CLARABEL, the active Git commit, data dates, and the exact analysis entry point;
- an indexed manifest introducing the complete 121-figure appendix.

- [ ] **Step 5: Run report contract tests**

```powershell
& '.\venv\Scripts\python.exe' 'report\New Project\generate_verified_results_tables.py'
& '.\venv\Scripts\python.exe' -m unittest discover -s 'report\New Project\tests' -v
```

Expected: study-period, protocol, objective, and unsupported-claim tests pass.

- [ ] **Step 6: Commit front matter and methodology**

```powershell
git add -- 'report/New Project/main.tex' 'report/New Project/generate_verified_results_tables.py' 'report/New Project/generated_verified_results_tables.tex' 'report/New Project/tests/test_report_contract.py'
git commit -m "docs: align thesis methodology with verified protocol"
```

### Task 4: Replace conceptual implementation claims with the code-audited system

**Files:**
- Modify: `report/New Project/main.tex:1569-1810`
- Test: `report/New Project/tests/test_report_contract.py`

- [ ] **Step 1: Add the frontend implementation subsection**

Document React 19, Vite 8, Material UI, MUI X Charts, MUI X Chat, Toolpad, authenticated application shell, chat/history UX, NDJSON streaming, analytics tabs, PlotSpec tokens, chart registries/validators, responsive rendering, premium fallback behavior, and fixture-gallery verification.

Add a component table mapping `App.jsx`, `AuthContext.jsx`, `ChatInterface.jsx`, `AnalyticsDashboard.jsx`, and `InlineChart.jsx` to their implemented responsibilities.

- [ ] **Step 2: Add the backend implementation subsection**

Document FastAPI, CORS, static outputs, authentication/analytics/governance routers, chat/session/run/event/plot routes, deterministic fast paths, intent lock, missing-data resolver, response contract, LangGraph orchestration, audit records, and advisory-only behavior.

Add an API-family table with `/health`, `/chat`, `/chat/stream`, `/chat/sessions`, `/api/auth`, `/api/analytics`, `/api/governance/decision`, and `/api/plots/{plot_id}`.

- [ ] **Step 3: Correct persistence and model-runtime descriptions**

Replace the SQLite-only claim with the implemented hybrid description: MongoDB plus Supabase PostgreSQL support, in-process session fallback where applicable, stored chat messages, session ownership, legacy claiming, and visualization persistence.

Describe Ollama primary/fallback selection, configured default-LLM fallback, Ashna API option, and `/health` degraded status when data/model dependencies are unavailable. State that availability is environment-dependent.

- [ ] **Step 4: Run the implementation contract test**

```powershell
& '.\venv\Scripts\python.exe' -m unittest discover -s 'report\New Project\tests' -v
```

Expected: implementation-component contract passes.

- [ ] **Step 5: Commit the implementation chapter**

```powershell
git add -- 'report/New Project/main.tex'
git commit -m "docs: document implemented frontend and backend"
```

### Task 5: Replace stale results with verified evidence

**Files:**
- Modify: `report/New Project/main.tex:1812-2418`
- Test: `report/New Project/tests/test_report_contract.py`

- [ ] **Step 1: Add the authoritative primary-results table**

Use the verified means:

```tex
\begin{table}[htbp]
\centering
\caption{Authoritative untouched-test primary-protocol means across 11 universes}
\label{tab:authoritative-primary-means}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Strategy} & \textbf{Return} & \textbf{Sharpe} & \textbf{CVaR loss} & \textbf{MaxDD} & \textbf{Graph exposure} \\
\midrule
Standard CVaR & 12.17\% & 0.767 & 2.209\% & 16.89\% & 0.178 \\
Static G-CVaR & 22.16\% & 1.081 & 2.753\% & 21.16\% & 0.147 \\
Adaptive G-CVaR & 23.47\% & 1.120 & 2.778\% & 21.62\% & 0.187 \\
\bottomrule
\end{tabular}
\end{table}
```

Identify `gcvar_test_governance_ranking.csv` as the source.

- [ ] **Step 2: Add pairwise win and activation evidence**

State that Adaptive beat Standard CVaR in 8/11 universes on return and Sharpe, 1/11 on CVaR loss, and 0/11 on maximum drawdown. Against Static G-CVaR, Adaptive won 7/11 on return, 8/11 on Sharpe, 4/11 on CVaR loss, and 3/11 on maximum drawdown.

State that the primary gate activated 7/132 decisions in U2, U4, U8, U9, and U10; crisis graph exposure was lower than calm exposure in four of five universes with crisis observations.

- [ ] **Step 3: Add the V2 supplemental table and interpretation**

Report V2 means: 12.01% annual return, 0.647 Sharpe, 2.153% CVaR loss, 16.91% maximum drawdown, and 22.57% mean turnover. State that all 132 decisions used the correlation proxy, no solver fallback occurred, and activation frequency was zero.

Explicitly state that a within-family rank of one is not a cross-strategy superiority claim.

- [ ] **Step 4: Separate legacy, core, supplemental, and HITL rankings**

Label the earlier tournament tables as a separate lane. Explain that the NaN-safe final ranking contains 110 eligible core rows and 11 eligible supplemental V2 rows, while 22 incomplete fixed-quarterly/HITL rows are exported with rejection reasons and receive no governance score.

- [ ] **Step 5: Replace unsupported result prose**

Remove categorical claims of guaranteed neutralisation, universal drawdown improvement, and universal G-CVaR superiority. Replace them with the observed selective-strength conclusion: strong return and Sharpe behavior in many universes, sparse crisis-aware activation, and weaker downside dominance than Standard CVaR in the authoritative primary protocol.

- [ ] **Step 6: Run contract tests and claim scan**

```powershell
& '.\venv\Scripts\python.exe' -m unittest discover -s 'report\New Project\tests' -v
rg -n '38\.1\\%|25\.9\\% CVaR|statistically significant reductions in crisis-period tail risk|always outperforms|universally superior' 'report\New Project\main.tex'
```

Expected: tests pass and `rg` returns no unsupported claims.

- [ ] **Step 7: Commit the verified results rewrite**

```powershell
git add -- 'report/New Project/main.tex'
git commit -m "docs: replace thesis claims with verified results"
```

### Task 6: Update conclusion, limitations, and artifact inventory

**Files:**
- Modify: `report/New Project/main.tex:2419-2551`
- Test: `report/New Project/tests/test_report_contract.py`

- [ ] **Step 1: Rewrite the conclusion**

Conclude that the contribution is a governed decision-support architecture and transparent evaluation protocol, not a universally dominant optimizer. Preserve the claim that deterministic analytics constrain conversational behavior, while qualifying economic performance by metric and universe.

- [ ] **Step 2: Expand limitations**

Include correlation-proxy dependence, absent SEC 13F input, sparse primary activation, zero V2 threshold activation, transaction-cost simplification, Yahoo Finance dependence, 2023--2025 test length, family-separated ranking, environment-dependent persistence/model services, and distinction between automated plot QA and economic validation.

- [ ] **Step 3: Add an artifact inventory**

List the full analysis script, `gcvar_protocol.py`, `gcvar_v2.py`, data-boundary audit, primary ranking, V2 audit/results/activation summary, NaN-safe rankings/rejections, technical checks, and plot-quality audit using repository-relative paths.

- [ ] **Step 4: Run all report contract tests**

```powershell
& '.\venv\Scripts\python.exe' -m unittest discover -s 'report\New Project\tests' -v
```

Expected: all report contract tests pass.

- [ ] **Step 5: Commit conclusion and limitations**

```powershell
git add -- 'report/New Project/main.tex'
git commit -m "docs: add thesis limitations and artifact audit"
```

### Task 7: Compile and visually verify the complete report

**Files:**
- Verify: `report/New Project/main.tex`
- Verify: `report/New Project/generated_result_figures.tex`
- Generate: `report/New Project/.latex-build/main.pdf`

- [ ] **Step 1: Regenerate the annex and rerun all contracts**

```powershell
& '.\venv\Scripts\python.exe' 'report\New Project\generate_result_figure_appendix.py'
& '.\venv\Scripts\python.exe' 'report\New Project\generate_verified_results_tables.py'
& '.\venv\Scripts\python.exe' -m unittest discover -s 'report\New Project\tests' -v
```

Expected: generator reports one annex; all report tests pass.

- [ ] **Step 2: Run the LaTeX doctor**

From `C:\Users\jithe\.codex\plugins\cache\openai-bundled\latex\0.2.3` run:

```powershell
python scripts\latex_doctor.py --json
```

Expected: status `ready` or `existing-usable`, with a successful Tectonic or TeX Live smoke compile.

- [ ] **Step 3: Compile through the LaTeX plugin workflow**

```powershell
python scripts\compile_latex.py 'D:\projects\agentic-ai-portfolio-governance\report\New Project\main.tex' --output-directory 'D:\projects\agentic-ai-portfolio-governance\report\New Project\.latex-build' --json
```

Expected: exit code 0 and `main.pdf` in `.latex-build`.

- [ ] **Step 4: Check the compile log**

```powershell
rg -n 'LaTeX Error|Undefined control sequence|File .* not found|undefined references|multiply defined|Overfull \\hbox' 'report\New Project\.latex-build' -g '*.log'
```

Expected: no fatal errors, missing files, or undefined references. Review and fix materially clipped overfull boxes.

- [ ] **Step 5: Verify PDF structure and embedded figure count**

Use `pypdf` to check that the document opens, has metadata title `Multi-Agent Agentic AI Framework for Portfolio Governance`, includes extractable text for `Supplemental Linear-Centrality Adaptive G-CVaR`, and has a page count consistent with the 121-figure annex.

- [ ] **Step 6: Render representative pages for visual QA**

Use the PDF skill to render and inspect the title, abstract, methodology boundary table, frontend/backend implementation tables, primary results table, V2 discussion, conclusion, first appendix group, middle appendix group, and final appendix page. Fix blank pages, clipped tables/captions, unreadable figures, and broken headings.

- [ ] **Step 7: Run final verification after any visual repairs**

Repeat report tests, annex generation, compile, log scan, PDF metadata/text checks, and representative page renders. Do not claim completion from an earlier build.

- [ ] **Step 8: Commit the verified report sources**

```powershell
git add -- 'report/New Project/main.tex' 'report/New Project/generated_result_figures.tex' 'report/New Project/generate_result_figure_appendix.py' 'report/New Project/generated_verified_results_tables.tex' 'report/New Project/generate_verified_results_tables.py' 'report/New Project/tests/test_report_contract.py'
git commit -m "docs: complete verified thesis report upgrade"
```

Do not commit `.latex-build`, notebook CSV outputs, notebook PNG outputs, or unrelated dirty-worktree files.
