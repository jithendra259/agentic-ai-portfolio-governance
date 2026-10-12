# Portfolio Follow-Up Grounding

- Risk-profile follow-ups such as "low-risk investor", "high-risk investor", "conservative", or "aggressive" must reuse the latest selected tickers when the thread already has a portfolio.
- If Latest governance run has tickers, target date, risk metrics, weights, or an effective historical window, treat those fields as the active portfolio context until the user explicitly changes them.
- For a changed risk profile, call `run_full_governance_pipeline` with the same tickers and target date, but pass the requested `risk_tolerance` value.
- Do not require, list, or ask for "universe" or universe codes. The system is generalized and dynamically operates on any stock tickers directly.
- If the requested target date is later than available historical coverage, report both the requested target date and the effective data window returned by the tool.
