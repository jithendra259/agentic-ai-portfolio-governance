from __future__ import annotations


def compute_concentration_metrics(
    ticker_weights: dict[str, float],
    ticker_to_sector: dict[str, str],
) -> dict[str, float]:
    total = sum(float(weight) for weight in ticker_weights.values())
    if total <= 0:
        return {
            "ticker_hhi": 0.0,
            "ticker_effective_holdings": 0.0,
            "sector_hhi": 0.0,
            "sector_effective_sectors": 0.0,
        }

    normalized = {ticker: float(weight) / total for ticker, weight in ticker_weights.items()}
    sector_weights: dict[str, float] = {}
    for ticker, weight in normalized.items():
        sector = ticker_to_sector.get(ticker, "Other")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

    ticker_hhi = sum(weight * weight for weight in normalized.values())
    sector_hhi = sum(weight * weight for weight in sector_weights.values())
    return {
        "ticker_hhi": ticker_hhi,
        "ticker_effective_holdings": 1.0 / ticker_hhi if ticker_hhi else 0.0,
        "sector_hhi": sector_hhi,
        "sector_effective_sectors": 1.0 / sector_hhi if sector_hhi else 0.0,
    }
