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
    if "def rolling_cvar_loss(" not in src:
        continue
    old = '''def rolling_cvar_loss(return_series, window=60, alpha=0.95):
    def _cvar(x):
        x = pd.Series(x).dropna()
        if len(x) < max(10, int(window * 0.5)):
            return np.nan
        var_level = x.quantile(1 - alpha)
        tail = x[x <= var_level]
        return abs(tail.mean()) if len(tail) else np.nan
    return return_series.rolling(window).apply(_cvar, raw=False)
'''
    new = '''def rolling_cvar_loss(return_series, window=60, alpha=0.95):
    """
    Fast rolling tail-loss proxy.

    The previous implementation recomputed a quantile and tail mean inside a
    Python callback for every window and asset. This vectorized version keeps the
    same interpretation: larger values mean deeper recent downside-tail losses.
    """
    min_periods = max(10, int(window * 0.5))
    rolling_var = return_series.rolling(window, min_periods=min_periods).quantile(1 - alpha)
    tail_losses = return_series.where(return_series <= rolling_var).abs()
    return tail_losses.rolling(window, min_periods=min_periods).mean()
'''
    if old not in src:
        raise SystemExit("Expected rolling_cvar_loss block not found")
    src = src.replace(old, new)
    cell["source"] = lines(src)
    break
else:
    raise SystemExit("No rolling_cvar_loss cell found")

NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Patched rolling tail-risk calculation.")
