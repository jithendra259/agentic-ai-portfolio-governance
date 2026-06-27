# Project Structure Mapping for Thesis

## Repository
- Repository: `jithendra259/agentic-ai-portfolio-governance`
- Branch observed: `main`
- Language mix observed from GitHub: predominantly Jupyter Notebook, with a smaller Python component.

## Top-Level Structure
The public repository layout shows these top-level items:
- `.github/`
- `api/`
- `data/`
- `evaluations/`
- `src/`
- `test/`
- `ui/`
- `requirements.txt`
- `run_ui.py`
- `yfinance_cache.sqlite`

## Thesis-Oriented Interpretation
This section is an interpretation for writing the thesis, based on folder names and dependency clues.

### 1. System Architecture & Implementation
Most likely mapped folders:
- `src/` — core implementation logic
- `api/` — backend endpoints / service integration layer
- `ui/` — governance dashboard or user-facing interface
- `run_ui.py` — application entry point for the interface

Suggested thesis use:
- Chapter 7: System Architecture & Implementation
- Section 7.1: Orchestrator Design
- Section 7.4: Storage & Versioning
- Section 7.5: Implementation Performance

### 2. Experimental Workflow
Most likely mapped folders:
- `evaluations/` — validation notebooks, experiments, benchmarking, or reporting scripts
- `data/` — datasets, cached inputs, processed artifacts, or intermediate outputs
- `yfinance_cache.sqlite` — cached market data source artifact

Suggested thesis use:
- Section 4.6: Experimental Design (Paper 1)
- Section 5.6: Experimental Design (Paper 2)
- Section 8.1: Evaluation Framework

### 3. Validation and Reproducibility
Most likely mapped folder:
- `test/` — verification logic, sanity checks, or reproducibility support

Suggested thesis use:
- Appendix D: Reproducibility Checklist & Links
- Chapter 8: Evaluation & Results Summary

## Dependency-Based Architecture Clues
The dependency file provides stronger evidence about the system design.

### Optimization Layer
Dependencies observed:
- `cvxpy`
- `ecos`
- `scs`

Interpretation:
- supports convex portfolio optimization
- likely relevant to the optimization agent and CVaR / shrinkage formulations

Suggested thesis use:
- Paper 1 optimization subsection
- Paper 2 G-CVaR optimizer subsection

### Graph Layer
Dependency observed:
- `networkx`

Interpretation:
- supports contagion graph construction and centrality computation

Suggested thesis use:
- Paper 2 contagion graph architecture
- systemic risk modeling chapter sections

### API and Interface Layer
Dependencies observed:
- `fastapi`
- `uvicorn`
- `requests`
- `gradio`

Interpretation:
- indicates separation between backend services and an interactive frontend layer

Suggested thesis use:
- HITL governance interface discussion
- architecture diagram explanation

### Storage Layer
Dependency observed:
- `pymongo`

Interpretation:
- supports MongoDB-backed persistence or blackboard-style storage

Suggested thesis use:
- blackboard architecture
- audit trail and replayability discussion

### LLM / Agent Orchestration Layer
Dependencies observed:
- `langchain`
- `langchain-ollama`
- `langgraph`
- `langchain-core`

Interpretation:
- supports agent orchestration, local-model integration, and graph-based control flow

Suggested thesis use:
- explainable AI component
- orchestrator description
- controlled role of LLM modules in governance workflows

### Research / Notebook Layer
Dependencies observed:
- `matplotlib`
- `seaborn`
- `yfinance`
- plus core numerics: `numpy`, `scipy`, `pandas`

Interpretation:
- supports notebook-based experimentation, visual analysis, and market-data acquisition

Suggested thesis use:
- methodology and result-generation pipeline
- figure provenance notes

## Mapping to Uploaded Notebook Assets
### Paper 1 notebook outputs
Current extracted assets:
- `figures/paper1/`
- `results/paper1/`
- `results/paper1/manifest.md`

Likely use:
- supervisory framework validation
- rolling performance plots
- threshold analysis
- crisis-period comparisons

### Paper 2 notebook outputs
Current extracted assets:
- `figures/paper2/`
- `results/paper2/`
- `results/paper2/manifest.md`

Likely use:
- contagion graph figures
- optimization and ablation outputs
- trust / XAI plots
- crisis-window analysis

## Recommended Writing Order
Without editing `main.tex`, the safest drafting order is:
1. `notes/chapter_7_architecture.md`
2. `notes/chapter_4_paper1_methodology.md`
3. `notes/chapter_5_paper2_methodology.md`
4. `notes/chapter_8_results_summary.md`
5. `notes/figure_table_mapping.md`

## Recommended Next Task
Create a figure-and-table mapping note that links:
- extracted notebook cell outputs
- thesis figure numbers from your outline
- target chapter/section placement