from __future__ import annotations


ADVISORY_REPLACEMENTS = {
    "Optimal Allocation Weights": "Advisory Allocation Weights",
    "optimal allocation weights": "advisory allocation weights",
    "Recommended allocation weights": "Suggested exposure weights",
    "recommended allocation weights": "suggested exposure weights",
    "Expected annualized return": "Estimated/backtested annualized return",
    "expected annualized return": "estimated/backtested annualized return",
}


def normalize_advisory_language(text: str) -> str:
    normalized = str(text or "")
    for unsafe, safe in ADVISORY_REPLACEMENTS.items():
        normalized = normalized.replace(unsafe, safe)
    return normalized
