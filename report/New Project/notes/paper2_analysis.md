# Paper 2 Analysis

## Source
- PDF: `prism-uploads/cas-sc-template (9).pdf`
- Title: *Multi-Agent Portfolio Governance with Graph-Regularized CVaR and Adaptive Contagion Penalization*
- Length observed: 24 PDF pages

## High-Level Summary
This paper addresses a different but complementary weakness in conventional portfolio risk systems: they measure tail risk within an asset-return window, but they do not directly account for cross-asset contagion transmitted through common institutional ownership.

The paper proposes the Graph-Regularized CVaR Agentic Portfolio Governance System (`G-CVaR-APGS`), which combines:
- a CVaR optimisation objective
- a graph-based penalty derived from institutional co-ownership centrality
- an adaptive sigmoid gate `lambda_t` that activates the penalty only when instability is sufficiently high
- a five-agent blackboard architecture with auditability and HITL governance

## Core Problem the Paper Solves
The central argument is that standard CVaR and related convex risk models are still incomplete because they ignore the ownership network through which crisis propagation can spread.

The paper therefore solves two linked problems:
1. systemic-risk blindness in tail-risk optimisation
2. governance and auditability needs in real institutional portfolio systems

This makes Paper 2 more explicitly network-aware and crisis-focused than Paper 1.

## Main Research Questions and Hypotheses
From the early sections, the paper states four major questions / hypotheses.

### RQ1 / H1
Whether institutional co-ownership structure can produce a meaningful concentration-penalty signal.

### RQ2 / H2
Whether sigmoid-gated graph penalisation improves CVaR@95% and crisis drawdown across 11 GICS universes.

### RQ3 / H3
Whether the five-agent blackboard architecture provides structural governance advantages over a monolithic optimiser.

### RQ4 / H4
Whether crisis-period gains can specifically be attributed to the adaptive graph-penalty mechanism rather than incidental effects.

## Methodology
### 1. Instability Index
The paper defines an instability index `I_t` over rolling windows using three bounded components:
- realised volatility spike
- correlation spike
- rolling maximum drawdown

The reported formula on the extracted page is:
- `I_t = 0.4 * sigma_spike + 0.3 * rho_spike + 0.3 * MDD_t`

Unlike Paper 1, this version of the instability index appears explicitly bounded in `[0,1]` and is used to activate the trust gate.

### 2. Regime Classification
The paper uses deterministic thresholds:
- Calm: `I_t < 0.50`
- Elevated: `0.50 <= I_t < 0.85`
- Crisis: `I_t >= 0.85`

This threshold structure is important because it controls when the graph penalty becomes active.

### 3. Graph-Regularized CVaR
The proposed optimisation framework embeds a graph penalty into CVaR using institutional co-ownership information.

The paper highlights:
- bipartite holdings network
- stock-level centrality signal
- adaptive sigmoid gating via `lambda_t`
- a formal joint-convexity proof

The centrality signal is based on eigenvector centrality, which links the method directly to network contagion interpretation rather than correlation alone.

### 4. Multi-Agent Governance Architecture
The paper repeatedly emphasizes a five-agent blackboard design with:
- data ingestion
- instability calculation
- contagion graph construction
- optimisation
- audit / explanation / governance support

This is one of the strongest links between the mathematical method and the implementation architecture.

## Experimental Design
From the abstract and later extracted pages, the main evaluation setup includes:
- 11 GICS sector universes
- 218 equities
- 552 rolling windows
- time span: 2005–2025
- three major crisis episodes included

The paper also reports a crisis-only evaluation restricted to the windows where the adaptive penalty is actually active.

## Key Empirical Results
The extracted pages provide several headline results:

### Aggregate performance versus Equal Weight
- mean CVaR@95% reduction: `25.9%`
- crisis-regime drawdown reduction: `32.5 pp`
- mean Sharpe ratio: `0.646`
- pooled Sharpe improvement versus Equal Weight significant at `p = 0.0065`

### Crisis-active windows versus Standard CVaR
In the `22` windows where `lambda_t > 0.5`:
- CVaR@95% improvement significant at `p = 0.0031`
- maximum drawdown improvement significant at `p = 0.0068`
- pooled Sharpe gap versus Standard CVaR is near zero / marginal (`p = 0.052`, `beta_hat = 0.31`), which the paper interprets as architecturally expected because the penalty is inactive most of the time

### Regime-stratified result highlights
From the extracted table:
- Calm regime CVaR reduction: `26.0%`
- Elevated regime CVaR reduction: `24.9%`
- Crisis regime CVaR reduction: `20.7%`
- Calm-regime non-significance is explicitly described as expected because `lambda_t ~= 0` by design

### Ablation evidence
The paper presents a four-condition ablation and reports a clear crisis-window ordering:
- Full G-CVaR
- No Graph
- Static-lambda
- No Both / Standard CVaR

This is presented as direct support for the hypothesis that the adaptive graph component is causally responsible for crisis-window gains.

## Strongest Contribution Claims
The paper’s strongest thesis-relevant claims are:
1. graph-regularized CVaR using institutional co-ownership centrality
2. adaptive penalty activation tied to instability rather than always-on regularisation
3. blackboard-style multi-agent governance implementation with auditability and HITL support
4. crisis-specific empirical validation across a much larger sample than Paper 1
5. explicit distinction between aggregate performance and active-window performance

## Architecture and Governance Importance
This paper is especially valuable because it does not present the optimisation method in isolation. It connects the method to governance requirements:
- fault isolation
- audit trails
- human-in-the-loop control
- persistent blackboard architecture
- human-readable decision traces

This makes it extremely useful for the implementation and governance chapters of the thesis, not only the quantitative results chapters.

## Strongest Thesis-Relevant Angles
### For Chapter 1 / Introduction
- supports the argument that systemic-risk channels exist outside standard return/covariance modeling
- adds institutional co-ownership as a second major motivation beyond estimation instability

### For Chapter 2 / Literature Review
- bridges CVaR optimisation, financial networks, blackboard systems, and XAI/governance
- offers a strong contrast against both black-box ML and static convex optimisation

### For Chapter 3 / Contributions
- supports claims around systemic-risk capture, adaptive governance, and graph-aware tail-risk control

### For Chapter 5 / Paper 2 Methodology
- directly supports the mathematical formulation, regime thresholds, graph penalty, and ablation logic

### For Chapter 7 / System Architecture
- supports the five-agent blackboard and governance workflow narrative

### For Chapter 8 / Results Summary
- provides rich aggregate, crisis-only, regime-stratified, and ablation evidence

## Limitations Visible from the Paper
Several limitations are visible even from the extracted text:
- gains relative to Standard CVaR are concentrated in the subset of windows where the penalty is active
- pooled aggregate tests can understate the benefit because the architecture is dormant in calm periods
- graph signal quality depends on the institutional holdings data and centrality modeling assumptions
- performance comparisons differ by regime, so interpretation must be context-aware rather than headline-only

These are not weaknesses to hide; they are important to frame honestly in the thesis.

## Cross-Check with Extracted Notebook Assets
The second notebook extraction aligns well with this paper:
- `figures/paper2/` contains many more image outputs, consistent with graph, regime, ablation, validation, and governance figures
- `results/paper2/` contains one HTML table and many text outputs, likely reflecting logged experiment summaries and metrics

These assets likely support:
- contagion graph illustrations
- instability / gate visualisations
- ablation figures
- validation and cross-universe plots

## Relationship to Paper 1
Paper 2 appears to build on the governance logic introduced in Paper 1 but extends it in three major ways:
- from estimation-instability governance to systemic-risk-aware governance
- from a binary shrinkage/equal-weight rule toward graph-regularized CVaR optimisation
- from a smaller universe and shorter horizon to a much larger multi-sector, multi-crisis evaluation setup

This relationship will be important in the thesis integration chapter.

## Recommended Next Notes to Create
Best next files, still without touching `main.tex`:
1. `notes/paper2_section_mapping.md`
2. `notes/chapter_5_paper2_methodology.md`
3. `notes/figure_table_mapping.md`
4. `notes/integration_paper1_paper2.md`

## Practical Use for Thesis Writing
This paper is strong enough to anchor the full Paper 2 chapter, especially for:
- systemic-risk motivation
- graph-aware optimization methodology
- adaptive penalty interpretation
- agentic governance architecture
- crisis-only and ablation-based empirical validation

The most valuable next step is to map the extracted notebook figures and outputs to the paper’s figures, tables, and claimed results.