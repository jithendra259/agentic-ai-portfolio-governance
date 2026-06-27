from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("report/New Project/Agentic_AI_Portfolio_Governance_from_text.ipynb")


def as_text(source):
    return "".join(source) if isinstance(source, list) else (source or "")


def lines(text: str):
    return text.splitlines(keepends=True)


nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    src = as_text(cell.get("source", ""))
    if '"max_cvar_scenarios": 750' in src:
        src = src.replace('"max_cvar_scenarios": 750', '"max_cvar_scenarios": 250')
        src = src.replace(
            "Scenarios are sampled evenly across the selected history, not randomly.",
            "Scenarios are sampled evenly across the selected history, not randomly; the cap keeps optimization reproducible and fast enough for full-notebook execution.",
        )
        cell["source"] = lines(src)

for cell in nb["cells"]:
    src = as_text(cell.get("source", ""))
    if "def optimize_cvar_weights(" not in src:
        continue

    start = src.index("def optimize_cvar_weights(")
    replacement = '''def cap_and_redistribute_weights(raw_weights, max_weight=0.25, max_iter=100):
    """Normalize long-only weights while respecting a per-asset cap."""
    w = pd.Series(raw_weights, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0)
    w = w.clip(lower=0)
    if w.sum() == 0:
        w[:] = 1.0
    w = w / w.sum()
    max_w = max(max_weight, 1.0 / len(w) + 1e-6)
    for _ in range(max_iter):
        over = w > max_w
        if not over.any():
            break
        excess = (w[over] - max_w).sum()
        w[over] = max_w
        under = ~over
        if not under.any() or w[under].sum() == 0:
            break
        w[under] += excess * (w[under] / w[under].sum())
    return w / w.sum()


def optimize_cvar_weights(returns_df, alpha=0.95, max_weight=0.25, return_tradeoff=0.05, centrality_penalty=None, graph_lambda=0.0):
    """
    Fast CVaR-aware long-only allocation.

    The original convex CVaR program is mathematically valid but too slow for repeated
    universe-wise notebook execution on a local machine. This deterministic approximation
    estimates each asset's tail-loss burden on an evenly sampled scenario set, rewards
    positive mean return, optionally penalizes graph-central exposure, and enforces the
    same long-only / max-weight constraints used elsewhere in the notebook.
    """
    clean = returns_df.dropna(how="any")
    if clean.empty:
        return pd.Series(dtype=float)

    max_scenarios = int(CONFIG.get("max_cvar_scenarios", 250))
    if len(clean) > max_scenarios:
        scenario_positions = np.linspace(0, len(clean) - 1, max_scenarios).round().astype(int)
        clean = clean.iloc[np.unique(scenario_positions)]

    cols = clean.columns
    downside_cvar = clean.apply(
        lambda s: abs(s[s <= s.quantile(1 - alpha)].mean()) if (s <= s.quantile(1 - alpha)).any() else s.std(),
        axis=0,
    ).replace(0, np.nan)
    downside_cvar = downside_cvar.fillna(downside_cvar.median()).replace(0, 1e-6)

    annualized_return = (clean.mean() * TRADING_DAYS).clip(lower=0)
    reward = annualized_return + return_tradeoff
    penalty = downside_cvar.copy()
    if centrality_penalty is not None and graph_lambda > 0:
        penalty = penalty + graph_lambda * centrality_penalty.reindex(cols).fillna(0)

    raw = reward / penalty.replace(0, 1e-6)
    weights = cap_and_redistribute_weights(raw, max_weight=max_weight)
    weights.index = cols
    return weights
'''
    src = src[:start] + replacement
    cell["source"] = lines(src)
    break

NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Patched fast CVaR-aware optimizer and reduced scenario cap.")
