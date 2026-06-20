# Thesis Report Evidence and Implementation Upgrade Design

## Goal

Upgrade `report/New Project/main.tex` into a professional, internally consistent, and evidence-backed thesis that accurately documents the implemented frontend and backend, the verified 2014--2025 quantitative protocol, the supplemental linear-centrality G-CVaR V2 experiment, and every generated result figure.

## Scope

The upgrade is limited to the thesis source and report-local supporting files. It does not change the mathematical objectives, model parameters, application code, generated experiment results, or ranking weights. The report must describe the repository and experimental evidence as they exist after the completed notebook-script run.

The report will cover:

- the complete React/Vite/MUI frontend implementation;
- FastAPI routes, deterministic request handling, LangGraph orchestration, memory, persistence, authentication, analytics, governance endpoints, streaming, plotting, health reporting, and model fallback behavior;
- the verified 197-asset, 11-universe data panel from 2014-01-02 through 2025-12-31;
- the strict training, validation, and untouched-test boundaries;
- the frozen quadratic Static and Adaptive G-CVaR protocol;
- the separate supplemental linear-centrality Adaptive G-CVaR V2 experiment;
- NaN-safe, family-separated ranking rules and rejection handling;
- all 121 generated result figures plus the report's existing architectural figures;
- observed strengths, weaknesses, non-activation outcomes, and limitations.

## Evidence hierarchy

The report will use an explicit hierarchy so different experimental lanes cannot be mistaken for interchangeable estimates.

1. **Authoritative primary protocol:** the walk-forward quadratic G-CVaR evaluation using training 2014--2019, validation 2020--2022, and untouched testing 2023--2025.
2. **Core comparison lane:** the complete-strategy comparison used by the NaN-safe core governance ranking.
3. **Supplemental experiments:** the linear-centrality Adaptive G-CVaR V2 and fixed-quarterly extensions.
4. **HITL simulation:** reported separately from mathematical portfolio strategies.
5. **Legacy tournament outputs:** retained only as historical or supplemental context and labelled so they cannot override the authoritative protocol.

Where two files use the same strategy label but arise from different lanes, the report will state the lane and evaluation method next to the value. It will not combine incompatible estimates into one comparison table.

## Quantitative protocol to document

### Data coverage

The verified market panel contains all 197 requested assets across 11 thematic universes. Prices cover 2014-01-02 to 2025-12-31, producing 3,018 price rows and 3,017 return rows per universe. The report will replace stale references to 2005 or 2012 start dates with the verified dates.

### Experimental boundaries

The primary evaluation protocol is:

- training: 2014-01-02 through 2019-12-31;
- validation and parameter selection: 2020-01-01 through 2022-12-31;
- untouched test: 2023-01-01 through 2025-12-31.

The calibration-versus-test audit must be described and the report must state that the test interval was not used for calibration.

### Frozen primary mathematical model

Static G-CVaR remains the fixed graph-aware optimizer and Adaptive G-CVaR remains the instability-gated graph-aware optimizer. Their graph-risk term is

\[
\lambda_t w^\top A_t w.
\]

The report must explain that this quadratic term preserves pairwise graph-risk interaction. It must not redefine the completed primary experiment.

### Supplemental V2 model

Adaptive G-CVaR V2 is labelled **Supplemental Linear-Centrality Adaptive G-CVaR** and uses

\[
\operatorname{CVaR}_\alpha(w)-\eta\mu^\top w+\lambda_t c_t^\top w.
\]

The report must state that V2 uses a correlation-network proxy because a publication-date-safe local SEC 13F holdings file was unavailable. It must also state that V2 completed all 132 quarterly decisions with no solver fallback or look-ahead violation, but the pre-registered activation condition was not crossed in the untouched test window. Zero activation is an empirical result, not a defect to be hidden or a threshold to be retuned after test inspection.

## Result interpretation rules

The report will remove or qualify unsupported statements such as universal superiority, guaranteed contagion neutralisation, unverified statistical significance, and stale fixed percentages including the current unqualified 38.1% drawdown and 25.9% crisis-CVaR claims.

The authoritative primary results will be described as follows:

- Adaptive G-CVaR produced strong return and Sharpe outcomes relative to Standard CVaR in many universes;
- it did not dominate Standard CVaR on CVaR loss or maximum drawdown;
- relative to Static G-CVaR, Adaptive G-CVaR more often improved return and Sharpe than downside metrics;
- the primary adaptive gate activated in 7 of 132 untouched-test decisions across five universes;
- crisis graph exposure was lower than calm exposure in four of the five universes with crisis observations;
- the result supports selective regime behavior, not universal downside dominance.

V2 will be described as technically valid and broadly competitive with Standard CVaR, but not as a superior adaptive strategy because its activation frequency was zero. A rank of one inside the supplemental family must not be interpreted as a cross-strategy victory when V2 is the only eligible strategy in that family.

The report will distinguish automated plot-integrity validation from substantive economic interpretation. All 121 PNG files passed the implemented plot-quality audit, but that fact alone is not evidence that every economic claim is correct.

## Frontend implementation section

The revised implementation chapter will describe the repository's actual frontend:

- React 19 application built with Vite 8;
- Material UI, MUI X Charts, MUI X Chat, and Toolpad components;
- authenticated application shell and login/signup/OAuth flows;
- ChatGPT-style chat interface with persistent session identifiers and conversation navigation;
- NDJSON streaming through `/chat/stream`;
- session listing, hydration, legacy-session claiming, deletion, and new-chat behavior;
- Markdown, mathematical notation, and inline PlotSpec rendering;
- analytics dashboard with data/EDA, risk/governance, correlation/covariance, and backtesting tabs;
- responsive chart renderers and registries for line, bar, pie, scatter, box, heatmap, network, candlestick, Sankey, funnel, radar, gauge, radial, and sparkline families;
- explicit plot validation, premium-chart fallback behavior, and deterministic fixture-gallery testing.

The report will avoid claiming that every chart is generated by the LLM. It will explain that structured PlotSpecs, deterministic resolvers, validation, and frontend renderers constrain visualization behavior.

## Backend implementation section

The revised implementation chapter will describe the actual backend:

- FastAPI application with CORS configuration and static output serving;
- authentication router for login, signup, session verification, logout, and OAuth providers;
- analytics router under `/api/analytics`;
- governance-decision router under `/api/governance/decision`;
- synchronous chat, streaming chat, background-run, event-stream, session-history, audit, health, and plot-data routes;
- intent locking, deterministic request fast paths, missing-data resolution, response contracts, and planner state;
- LangGraph orchestration and checkpointer behavior;
- structured agent and decision modules rather than unrestricted LLM tool selection;
- MongoDB-backed historical data and memory with Supabase PostgreSQL support and documented fallback paths;
- persistent chat messages, session ownership, legacy-history claiming, and stored visualization payloads;
- Ollama primary/fallback model configuration, configured default-LLM fallback, and degraded health reporting when required services are unavailable;
- advisory-only operating posture and explicit audit records.

The report will state that persistence and model availability are environment-dependent. It will not present an optional external service as always available.

## Report structure

The existing thesis structure will be preserved to minimize numbering and reference breakage. The principal changes are:

1. **Front matter:** correct the study period and rewrite the abstract using verified results.
2. **Methodology:** add data coverage, strict date boundaries, frozen primary objective, V2 objective, graph-source fallback, activation rule, and ranking eligibility.
3. **System design and implementation:** replace conceptual-only descriptions with the code-audited frontend/backend architecture and runtime behavior.
4. **Experimental setup:** document the authoritative evidence hierarchy and prevent cross-lane comparisons.
5. **Results and discussion:** replace unsupported claims with verified metrics, win counts, gate behavior, solver evidence, and limitations.
6. **Conclusion:** state selective strengths and honest limitations rather than universal superiority.
7. **Appendices:** add a complete indexed figure compendium for all generated result PNGs and a compact artifact/audit inventory.

## Complete figure inclusion

Every PNG under `notebook/figures_universe_analysis` at implementation time will be included in the compiled report. Key figures will appear in the Results chapter. The complete collection will appear in a grouped visual annex organized by source subdirectory and natural filename order.

To keep `main.tex` maintainable, the full figure list will live in a report-local generated include file referenced by `main.tex`. Each entry will contain:

- a stable label derived from its relative path;
- a concise caption based on the figure name and evidence family;
- the source relative path;
- a continuation-friendly page layout;
- a short note that detailed metric values must be read from the corresponding CSV table.

The report build will use repository-relative paths so the source remains reproducible on another machine. Hard-coded `D:` or `C:` paths will be removed from the final report configuration where practical.

## Tables and audit evidence

The main report will include compact tables for:

- data coverage and date boundaries;
- authoritative primary mean outcomes;
- Adaptive G-CVaR win counts versus Standard CVaR and Static G-CVaR;
- primary and V2 activation behavior;
- solver, feasibility, and look-ahead checks;
- NaN-safe ranking eligibility and rejection reasons;
- frontend and backend component-to-responsibility mapping;
- API route families and persistence/fallback behavior.

The report will cite the exact generated CSV filename beneath each empirical table. Values will be transcribed from the completed run without post-test optimization.

## Professional presentation requirements

- Preserve consistent typography, captions, labels, cross-references, and table formatting.
- Avoid duplicated captions, stale page-number tables, unsupported superlatives, and unexplained acronyms.
- Use `booktabs`, `longtable`, and existing report macros consistently.
- Keep key figures readable at normal page scale; use landscape pages only where materially useful.
- Ensure all figures have source-aware captions and are referenced or clearly identified as appendix evidence.
- Keep the report's advisory-only scope explicit.
- Describe limitations alongside the relevant result, not only in the final chapter.

## Verification

The completed upgrade must pass the following checks:

1. scan for stale study periods and unsupported fixed claims;
2. scan for the required primary/V2 equations and protocol labels;
3. verify that all generated PNG paths are referenced exactly once in the report sources;
4. verify that all references and included files exist;
5. run the LaTeX plugin's environment doctor;
6. compile with the plugin's compile workflow into a report-local build directory;
7. inspect the compile log for fatal errors, missing files, undefined references, and serious overfull boxes;
8. confirm PDF metadata, page count, and extractable text;
9. render representative front matter, methodology, implementation, results, conclusion, and appendix pages for visual inspection;
10. confirm that unrelated repository changes were not modified.

## Acceptance criteria

The task is complete when the PDF compiles successfully, all 121 generated result figures are embedded, the implementation description matches the current frontend/backend code, quantitative claims match the completed experimental outputs, the primary and supplemental protocols are clearly separated, and the conclusions do not claim universal G-CVaR superiority.
