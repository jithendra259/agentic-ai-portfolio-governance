# Integration of Paper 1 and Paper 2

## Purpose of This Note
This note explains how Paper 1 and Paper 2 fit together as a single thesis contribution rather than two unrelated studies.

## One-Sentence Integration
Paper 1 establishes a deterministic governance framework for instability-aware portfolio control, while Paper 2 extends that governance logic into a network-aware tail-risk optimisation system that captures systemic contagion through institutional co-ownership.

## Unified Thesis Logic
The two papers address different layers of the same broad problem:
- portfolio systems must remain robust under instability
- portfolio decisions must be auditable and reproducible
- portfolio risk must account for systemic contagion, not only standalone asset behavior
- governance systems must support explanation and human oversight without compromising determinism

## What Paper 1 Contributes to the Overall Thesis
Paper 1 provides the governance foundation.

### Main role of Paper 1
- defines instability as an operational governance signal
- formalizes deterministic regime-based intervention
- introduces Governance Stability as a governance-aware evaluation criterion
- shows that auditability can be embedded directly into portfolio control logic
- demonstrates that structural LLM isolation is compatible with human-readable explanations

### Conceptual position
Paper 1 is best understood as the thesis foundation layer:
- it answers how a portfolio system should detect instability
- it answers how the system should make deterministic, auditable choices once instability is detected
- it creates the design philosophy later reused in Paper 2

## What Paper 2 Adds Beyond Paper 1
Paper 2 provides the systemic-risk extension.

### Main role of Paper 2
- introduces institutional co-ownership as a contagion channel
- embeds graph-derived systemic-risk information into CVaR optimisation
- replaces a simpler binary allocation switch with graph-aware convex optimisation under adaptive activation
- scales the empirical scope from a smaller validation setting to a much broader multi-sector, multi-crisis setting
- strengthens the agentic and governance architecture with a five-agent blackboard model and active-window crisis validation

### Conceptual position
Paper 2 is best understood as the thesis expansion layer:
- it preserves the governance principles of Paper 1
- it extends the risk model from instability response to contagion-aware tail-risk control
- it demonstrates that governance and systemic-risk modelling can coexist in a single agentic framework

## Core Continuities Between the Papers
These are the themes that unify both papers and should be emphasized in the thesis.

### 1. Governance First
Both papers treat governance as a primary design objective rather than an afterthought.

Shared themes:
- deterministic or frozen decision logic
- reproducibility from documented inputs and rules
- auditability under financial regulation
- post-decision explanation rather than unconstrained AI-driven decision making

### 2. Instability as a Triggering Principle
Both papers rely on instability detection as a central mechanism.

Shared themes:
- instability is not just descriptive, but operational
- the system changes behavior when instability rises
- the change in behavior is rule-bound and inspectable

### 3. Structural Separation of Roles
Both papers use architecture to enforce trustworthiness.

Shared themes:
- separated agents or functional layers
- blackboard / orchestrated execution logic
- read-only explanation layer
- no hidden discretionary path between explanation and decision execution

### 4. Evaluation Under Stress
Both papers prioritize crisis and stress-period relevance rather than only average-case aggregate metrics.

Shared themes:
- drawdown protection
- crisis-window validation
- robustness under severe market conditions
- comparison against conventional baselines

## Main Differences Between the Papers
These differences are important because they justify why both papers are needed.

### Paper 1
- core question: how to make instability-aware portfolio governance auditable
- main mechanism: deterministic regime switch
- main novelty: covariance drift and governance stability
- main benefit: drawdown reduction with reproducible governance logic
- scale: 19 equities, 51 windows, five universes

### Paper 2
- core question: how to incorporate contagion channels into governed tail-risk optimisation
- main mechanism: graph-regularized CVaR with sigmoid-gated penalty
- main novelty: institutional co-ownership centrality inside adaptive CVaR
- main benefit: improved tail-risk control in crisis-active windows and broader systemic-risk capture
- scale: 218 equities, 552 windows, 11 GICS universes

## Natural Thesis Narrative
A strong thesis narrative can be stated as follows:

1. Conventional portfolio optimisation fails under estimation error and lacks governance transparency.
2. Paper 1 solves the first governance layer by defining instability, deterministic response, and auditable decision flow.
3. Conventional tail-risk optimisation still ignores network contagion and systemic transmission channels.
4. Paper 2 solves the second governance layer by embedding graph-aware contagion penalties into CVaR under the same governance philosophy.
5. Together, the two papers form a unified framework for governed, explainable, systemic-risk-aware portfolio management.

## Suggested Chapter-Level Integration
### Chapter 1: Introduction
Use both papers to frame the thesis problem as three linked gaps:
- estimation instability
- auditability gap
- systemic-risk blindness

### Chapter 2: Literature Review
Split the review into streams that converge into the thesis:
- mean-variance and estimation error
- shrinkage methods
- regime-switching and auditability limits
- CVaR and tail-risk optimisation
- contagion graphs and institutional networks
- XAI and expert systems in finance

### Chapter 3: Problem Definition and Contributions
Position contributions in two layers:
- Paper 1 contributions: governance foundation
- Paper 2 contributions: network-aware extension

### Chapter 4: Paper 1
Describe the deterministic supervisory governance framework.

### Chapter 5: Paper 2
Describe the graph-regularized CVaR extension.

### Chapter 6: Integration and Unified System
This is where the relationship between the two papers should be made explicit.

Recommended framing:
- Paper 1 answers when and why intervention should occur
- Paper 2 answers how a more advanced crisis-aware optimizer should act once systemic channels matter

### Chapter 7: System Architecture
Use both papers together to explain the overall orchestrated / blackboard architecture.

### Chapter 8: Results Summary
Compare the papers not as competitors but as complementary layers with different objectives and scales.

## Evidence Available for Integration
The shared evidence base now available in the workspace includes:
- `notes/paper1_analysis.md`
- `notes/paper2_analysis.md`
- `notes/project_structure.md`
- extracted notebook figures in `figures/paper1/` and `figures/paper2/`
- extracted result tables and summaries in `results/paper1/` and `results/paper2/`

## Strong Final Framing for the Thesis
A concise final framing for the overall thesis is:

> Paper 1 establishes a deterministic, auditable supervisory governance layer for portfolio decision-making under instability. Paper 2 extends that governed framework by integrating institutional co-ownership networks into adaptive CVaR optimisation, thereby addressing systemic contagion risk without sacrificing reproducibility, oversight, or explanation.

## Recommended Next Note
The best next companion file is `notes/figure_table_mapping.md`, which should connect:
- thesis figure/table numbers
- paper-level claims
- extracted notebook outputs
- target chapter sections