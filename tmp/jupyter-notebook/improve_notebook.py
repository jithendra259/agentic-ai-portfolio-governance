from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("report/New Project/Agentic_AI_Portfolio_Governance_from_text.ipynb")
MARKER = "<!-- codex-notebook-summary-v1 -->"


def as_text(source):
    return "".join(source) if isinstance(source, list) else (source or "")


def lines(text: str):
    return text.splitlines(keepends=True)


def markdown_cell(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

# Drop prior Codex summary/instruction cells so this script is idempotent.
nb["cells"] = [
    c for c in nb["cells"]
    if MARKER not in as_text(c.get("source", ""))
]

intro = f"""# Agentic AI Portfolio Governance — Reproducible Market-Risk Notebook

{MARKER}

This notebook turns the raw thesis-style portfolio governance text into a reproducible market-risk experiment. It cleans a multi-sector equity universe, downloads adjusted market data, constructs strict historical panels, measures asset and universe risk, evaluates diversification and instability regimes, optimizes portfolios, backtests strategies, and exports an auditable candidate register.

What this run is designed to answer:

- Which assets and universes have the strongest risk-adjusted historical profile?
- Where do hidden concentration, tail-risk, and instability signals appear?
- Which portfolio construction rules survive out-of-sample testing?
- What evidence should a governance workflow cite before approving or rejecting candidates?

Important caveat: the notebook uses public market data from Yahoo Finance via yfinance. It is a research and governance artifact, not a binding investment, credit, or D&B decision output.
"""

method = f"""## Execution and validation notes

{MARKER}

The notebook is configured for deterministic local reruns:

- Local exports are written under report/New Project/results/agentic_ai_portfolio_governance/.
- CVaR optimization uses a deterministic scenario cap so the convex programs finish predictably while preserving coverage across the full historical window.
- Missing optional D&B / 13F ownership data is treated as an explicit caveat rather than silently fabricated.
- Final claim validation only cites tables produced earlier in the notebook.
"""

nb["cells"].insert(0, markdown_cell(intro))
nb["cells"].insert(2, markdown_cell(method))

# Improve imports.
cell0 = as_text(nb["cells"][1]["source"])
cell0 = cell0.replace("from datetime import datetime\n", "from datetime import datetime, timezone\n")
if "from pathlib import Path" not in cell0:
    cell0 = cell0.replace("# Date utilities\nfrom datetime", "# Date utilities\nfrom pathlib import Path\nfrom datetime")
if "from IPython.display import display" not in cell0:
    cell0 = cell0.replace("# Display settings\n", "# Notebook display\nfrom IPython.display import display\n\n# Display settings\n")
nb["cells"][1]["source"] = lines(cell0)

# Add reproducibility and project-local output configuration.
for c in nb["cells"]:
    src = as_text(c.get("source", ""))
    if "# 1.2 Experiment Configuration" in src:
        replacement = '''# ============================================================
# 1.2 Experiment Configuration
# ============================================================

CONFIG = {
    "start_date": "2005-01-01",
    "end_date": "2025-01-01",
    "rebalance_frequency": "M",
    "initial_capital": 100000,
    "risk_free_rate": 0.02,
    "cvar_alpha": 0.95,
    "min_weight": 0.00,
    "max_weight": 0.25,
    "top_k_assets": 5,
    # Keep convex CVaR optimization reliable on local machines.
    # Scenarios are sampled evenly across the selected history, not randomly.
    "max_cvar_scenarios": 750,
    "random_seed": 42,
    "data_source": "Yahoo Finance via yfinance"
}

np.random.seed(CONFIG["random_seed"])

PROJECT_ROOT = Path.cwd()
OUTPUT_DIR = PROJECT_ROOT / "report" / "New Project" / "results" / "agentic_ai_portfolio_governance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG["output_dir"] = str(OUTPUT_DIR)
CONFIG["run_timestamp_utc"] = datetime.now(timezone.utc).isoformat()

display(pd.DataFrame([CONFIG]).T.rename(columns={0: "value"}))
'''
        c["source"] = lines(replacement)
        break

# Project-local optional 13F path.
for c in nb["cells"]:
    src = as_text(c.get("source", ""))
    if 'THIRTEEN_F_PATH = "/home/oai/share/thirteen_f_holdings.csv"' in src:
        src = src.replace(
            'THIRTEEN_F_PATH = "/home/oai/share/thirteen_f_holdings.csv"  # update with actual path',
            'THIRTEEN_F_PATH = OUTPUT_DIR / "thirteen_f_holdings.csv"  # optional local 13F input'
        )
        c["source"] = lines(src)
        break

# Bound CVaR scenario count deterministically to prevent very long local runs.
for c in nb["cells"]:
    src = as_text(c.get("source", ""))
    if "def optimize_cvar_weights(" in src and "max_cvar_scenarios" not in src:
        src = src.replace(
            '    clean = returns_df.dropna(how="any")\n    cols = clean.columns\n',
            '''    clean = returns_df.dropna(how="any")
    max_scenarios = int(CONFIG.get("max_cvar_scenarios", 750))
    if len(clean) > max_scenarios:
        scenario_positions = np.linspace(0, len(clean) - 1, max_scenarios).round().astype(int)
        clean = clean.iloc[np.unique(scenario_positions)]
    cols = clean.columns
'''
        )
        c["source"] = lines(src)
        break

# Make exports Windows/project-local and path-safe.
for c in nb["cells"]:
    src = as_text(c.get("source", ""))
    if "Export tables to CSV in the shared folder" in src:
        src = src.replace(
            "# Export tables to CSV in the shared folder\noutput_dir = '/home/oai/share'\n",
            "# Export tables to CSV in the project results folder\noutput_dir = OUTPUT_DIR\noutput_dir.mkdir(parents=True, exist_ok=True)\n"
        )
        src = src.replace(
            'csv_path = f"{output_dir}/{name}.csv"\n    df.to_csv(csv_path)',
            'csv_path = output_dir / f"{name}.csv"\n    df.to_csv(csv_path)'
        )
        c["source"] = lines(src)
    if 'export_dir = globals().get("output_dir", "/mnt/data")' in src:
        src = src.replace(
            'export_dir = globals().get("output_dir", "/mnt/data")',
            'export_dir = Path(globals().get("output_dir", OUTPUT_DIR))\nexport_dir.mkdir(parents=True, exist_ok=True)'
        )
        src = src.replace(
            'path = f"{export_dir}/{name}.csv"\n        table.to_csv(path)',
            'path = export_dir / f"{name}.csv"\n        table.to_csv(path)'
        )
        src = src.replace('export_paths.append(path)', 'export_paths.append(str(path))')
        c["source"] = lines(src)

# Add a concise final reader-facing validation note.
final_note = f"""## Final validation lens

{MARKER}

Read the final candidate register as a governed research output:

- Supported claims are backed by executed notebook tables from this run.
- Needs review claims require human analyst review before being used in a formal investment, credit, or customer-risk decision.
- Optional ownership/co-ownership enrichment is only included when a reviewed 13F holdings file is supplied in the project results folder.
"""
nb["cells"].append(markdown_cell(final_note))

NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Updated {NOTEBOOK_PATH} with {len(nb['cells'])} cells")
