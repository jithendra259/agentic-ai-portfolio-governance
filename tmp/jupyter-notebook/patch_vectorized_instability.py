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
    if "def compute_asset_instability_index(" not in src:
        continue
    start = src.index("def compute_asset_instability_index(")
    end = src.index('if "instability_features" not in globals()', start)
    replacement = '''def compute_asset_instability_index(price_df, returns_df, window=60, alpha=0.95):
    """
    Vectorized asset instability engine.

    Computes the same five feature families as the original looped version:
    rolling volatility, rolling downside-tail loss, drawdown pressure, return shock,
    and correlation stress versus the equal-weight universe return.
    """
    aligned_prices = price_df.reindex(returns_df.index).ffill()
    min_periods = max(10, int(window * 0.5))

    rolling_volatility = returns_df.rolling(window, min_periods=min_periods).std() * np.sqrt(TRADING_DAYS)

    rolling_var = returns_df.rolling(window, min_periods=min_periods).quantile(1 - alpha)
    tail_losses = returns_df.where(returns_df <= rolling_var).abs()
    rolling_cvar = tail_losses.rolling(window, min_periods=min_periods).mean()

    cumulative = aligned_prices / aligned_prices.iloc[0]
    drawdown_pressure_df = ((cumulative / cumulative.cummax()) - 1).abs().reindex(returns_df.index)

    return_shock = returns_df.abs().rolling(5, min_periods=3).mean()

    universe_return = returns_df.mean(axis=1)
    correlation_stress = returns_df.rolling(window, min_periods=min_periods).corr(universe_return).abs()

    scaled_vol = rolling_volatility.apply(safe_min_max_scale)
    scaled_cvar = rolling_cvar.apply(safe_min_max_scale)
    scaled_drawdown = drawdown_pressure_df.apply(safe_min_max_scale)
    scaled_shock = return_shock.apply(safe_min_max_scale)
    scaled_corr = correlation_stress.apply(safe_min_max_scale)

    panel = (
        0.25 * scaled_vol +
        0.25 * scaled_cvar +
        0.20 * scaled_drawdown +
        0.15 * scaled_shock +
        0.15 * scaled_corr
    )

    features = {}
    for ticker in returns_df.columns:
        features[ticker] = pd.DataFrame({
            "rolling_volatility": rolling_volatility[ticker],
            "rolling_cvar_loss": rolling_cvar[ticker],
            "drawdown_pressure": drawdown_pressure_df[ticker],
            "return_shock": return_shock[ticker],
            "correlation_stress": correlation_stress[ticker],
            "instability_index": panel[ticker],
        })
    return features, panel

'''
    src = src[:start] + replacement + src[end:]
    cell["source"] = lines(src)
    break
else:
    raise SystemExit("compute_asset_instability_index cell not found")

NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Patched vectorized instability engine.")
