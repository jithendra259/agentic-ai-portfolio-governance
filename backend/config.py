from dataclasses import dataclass

@dataclass(frozen=True)
class CFG:
    # ── Database ──────────────────────────────────────────────
    DB_NAME:           str   = "Stock_data"

    # ── Data ─────────────────────────────────────────────────
    START_DATE:        str   = "2005-01-01"
    END_DATE:          str   = "2025-12-31"

    # ── Rolling Windows ───────────────────────────────────────
    WINDOW_SIZE:       int   = 252      # trading days (~1 year)
    STEP_SIZE:         int   = 100      # shift between windows
    N_WINDOWS:         int   = 51       # total windows per universe

    # ── Optimization ─────────────────────────────────────────
    CVAR_ALPHA:        float = 0.95     # 95% CVaR confidence
    MAX_WEIGHT:        float = 0.15     # max 15% per asset
    CONSTRAIN_WEIGHT:  float = 0.08     # HITL constrain: max 8%
    RISK_FREE:         float = 0.03     # 3% annualized

    # ── Adaptive Trust (Sigmoid) ──────────────────────────────
    LAMBDA_MAX:        float = 1.0      # maximum penalty
    K_STEEPNESS:       int   = 10       # sigmoid steepness
    I_THRESH:          float = 0.85     # instability trigger

    # ── Governance Thresholds ─────────────────────────────────
    TAU_CALM:          float = 0.50     # calm/elevated trigger
    TAU_CRISIS:        float = 0.85     # crisis HITL trigger
    TAU_TURNOVER:      float = 0.40     # turnover HITL trigger

    # ── Monte Carlo ───────────────────────────────────────────
    N_MONTE_CARLO:     int   = 10000    # synthetic scenarios
    STRESS_FACTOR:     float = 2.5      # fat-tail multiplier
    N_FAT_TAIL:        int   = 500      # fat-tail paths

    # ── LLM ──────────────────────────────────────────────────
    LLM_MODEL:         str   = "mistral:latest"
    LLM_TEMPERATURE:   float = 0.3
    LLM_MAX_TOKENS:    int   = 400

# Instantiate once — import this object everywhere
CONFIG = CFG()
