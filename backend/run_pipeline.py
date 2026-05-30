"""
run_pipeline.py — Full 51-window × 11-universe Portfolio Governance Pipeline Entry Point.

Execution order:
  0. Agent 0 (DataSentinelAgent):  Pre-fetch price matrices → blackboard_mpi
  1. Agent 1 (TimeSeriesAgent):    Compute I_t per window
  2. Agent 2 (GraphRAGAgent):      Build institutional network
  3. Agent 3 (GCVaROptimizerAgent): Optimize portfolio weights
  4. Agent 4 (GenerativeExplainerAgent): Generate HITL governance report

Usage:
  python run_pipeline.py [--universes U1 U2 ...] [--skip-sentinel]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on the path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import CONFIG
from src.agents.data_sentinel_a0 import DataSentinelAgent
from src.blackboard.memory_store import BlackboardMemoryStore

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portfolio Governance Pipeline")
    parser.add_argument(
        "--universes",
        nargs="+",
        default=None,
        metavar="U",
        help="Universes to process (e.g. U1 U2 U3). Defaults to all 11.",
    )
    parser.add_argument(
        "--skip-sentinel",
        action="store_true",
        help="Skip Agent 0 pre-fetch step (assumes blackboard is already populated).",
    )
    parser.add_argument(
        "--n-windows",
        type=int,
        default=CONFIG.N_WINDOWS,
        help=f"Number of rolling windows (default: {CONFIG.N_WINDOWS}).",
    )
    return parser.parse_args()


def run_full_pipeline(
    universes: list[str] | None = None,
    skip_sentinel: bool = False,
    n_windows: int = CONFIG.N_WINDOWS,
) -> None:
    mongo_uri = (os.getenv("MONGO_URI") or "").strip()
    if not mongo_uri:
        logger.error("MONGO_URI is not set. Exiting.")
        sys.exit(1)

    target_universes = universes or [f"U{i}" for i in range(1, 12)]
    logger.info("Pipeline starting. Universes: %s | Windows: %d", target_universes, n_windows)

    # ── Phase 0: Data Sentinel ────────────────────────────────────────────────
    if not skip_sentinel:
        logger.info("=== Phase 0: Agent 0 — Data Sentinel ===")
        sentinel = DataSentinelAgent(mongo_uri=mongo_uri)
        sentinel_summary = sentinel.run(
            universes=target_universes,
            n_windows=n_windows,
        )
        for universe_id, count in sentinel_summary.items():
            logger.info("  %s: %d windows stored", universe_id, count)
    else:
        logger.info("=== Phase 0: Skipped (--skip-sentinel) ===")

    # ── Phase 1–4: Per-universe, per-window agent execution ───────────────────
    blackboard = BlackboardMemoryStore(mongo_uri=mongo_uri, db_name=CONFIG.DB_NAME)

    from pymongo import MongoClient
    client = MongoClient(
        mongo_uri,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=10000,
    )
    db_collection = client[CONFIG.DB_NAME]["ticker"]

    import pandas as pd
    from src.agents.time_series_a1 import TimeSeriesAgent
    from src.agents.graph_rag_a2 import GraphRAGAgent
    from src.agents.optimizer_a3 import GCVaROptimizerAgent
    from src.agents.explainer_a4 import GenerativeExplainerAgent

    agent1 = TimeSeriesAgent(db_collection)
    agent2 = GraphRAGAgent(db_collection)
    agent3 = GCVaROptimizerAgent()
    agent4 = GenerativeExplainerAgent()

    results_summary: list[dict] = []

    for universe_id in target_universes:
        logger.info("=== Processing Universe: %s ===", universe_id)
        windows = blackboard.get_all_windows(universe_id)

        if not windows:
            logger.warning("No blackboard data for %s — was sentinel run?", universe_id)
            continue

        prev_weights = None

        for window_doc in windows[:n_windows]:
            window_id = window_doc.get("window_id", "?")
            window_end = window_doc.get("window_end", "?")
            logger.info("  [W%s] %s — end=%s", window_id, universe_id, window_end)

            try:
                # Load returns matrix from blackboard
                returns_json = window_doc.get("returns_matrix")
                if not returns_json:
                    continue
                returns_df = pd.read_json(returns_json)

                # ── Agent 1: I_t from pre-loaded returns ──────────────────────
                cov_matrix = returns_df.cov()
                corr_matrix = returns_df.corr()
                import numpy as np
                mean_vol = float(returns_df.std().mean() * np.sqrt(252.0))
                upper_tri = corr_matrix.to_numpy()[np.triu_indices(corr_matrix.shape[0], k=1)]
                mean_corr = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0
                cum_ret = np.exp(returns_df.cumsum())
                max_cum = cum_ret.cummax()
                drawdowns = (cum_ret - max_cum) / max_cum.replace(0.0, np.nan)
                mean_dd = float(np.abs(drawdowns.min()).mean())
                vol_norm = float(np.clip((mean_vol - 0.10) / 0.50, 0.0, 1.0))
                corr_norm = float(np.clip((mean_corr - 0.10) / 0.70, 0.0, 1.0))
                dd_norm = float(np.clip((mean_dd - 0.05) / 0.35, 0.0, 1.0))
                i_t = float(np.clip(0.4 * vol_norm + 0.3 * corr_norm + 0.3 * dd_norm, 0.0, 1.0))
                lambda_t = float(CONFIG.LAMBDA_MAX / (1.0 + np.exp(-CONFIG.K_STEEPNESS * (i_t - CONFIG.I_THRESH))))

                # ── Agent 2: Graph risk scores ────────────────────────────────
                tickers = list(returns_df.columns)
                graph_result = agent2.execute(universe_id=universe_id)
                c_vector = {
                    item["ticker"]: item["score"]
                    for item in graph_result.get("top_central_nodes", [])
                } if isinstance(graph_result, dict) else {}

                # ── Agent 3: G-CVaR Optimization ─────────────────────────────
                opt_result = agent3.execute(
                    returns_df=returns_df,
                    covariance_matrix=cov_matrix,
                    instability_index=i_t,
                    lambda_t=lambda_t,
                    c_vector=c_vector,
                    prev_weights=prev_weights,
                )

                optimal_weights = opt_result.get("optimal_weights", {})
                governance_flags = opt_result.get("governance_flags", [])
                prev_weights = optimal_weights

                # ── Agent 4: HITL Report (only in crisis) ────────────────────
                hitl_report = None
                if i_t >= CONFIG.TAU_CRISIS or governance_flags:
                    a4_result = agent4.execute(
                        date=window_end,
                        universe_id=universe_id,
                        instability_index=i_t,
                        lambda_t=lambda_t,
                        governance_flags=governance_flags,
                        c_vector=c_vector,
                        optimal_weights=optimal_weights,
                    )
                    hitl_report = a4_result.get("report")

                # ── Store results to blackboard ───────────────────────────────
                blackboard.store_window(
                    universe_id=universe_id,
                    window_id=window_id,
                    window_number=window_doc.get("window_number", 0),
                    data={
                        "i_t": i_t,
                        "lambda_t": lambda_t,
                        "optimal_weights": optimal_weights,
                        "governance_flags": governance_flags,
                        "hitl_report": hitl_report,
                    },
                )

                results_summary.append({
                    "universe": universe_id,
                    "window": window_id,
                    "window_end": window_end,
                    "i_t": round(i_t, 4),
                    "lambda_t": round(lambda_t, 4),
                    "regime": "crisis" if i_t >= CONFIG.TAU_CRISIS else "elevated" if i_t >= CONFIG.TAU_CALM else "calm",
                    "hitl_triggered": bool(governance_flags or hitl_report),
                })

            except Exception as exc:
                logger.exception("  [%s/%s] Error: %s", universe_id, window_id, exc)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=== Pipeline Complete ===")
    logger.info("Total windows processed: %d", len(results_summary))
    crisis_count = sum(1 for r in results_summary if r["regime"] == "crisis")
    hitl_count = sum(1 for r in results_summary if r["hitl_triggered"])
    logger.info("Crisis windows: %d | HITL triggered: %d", crisis_count, hitl_count)


if __name__ == "__main__":
    args = _parse_args()
    run_full_pipeline(
        universes=args.universes,
        skip_sentinel=args.skip_sentinel,
        n_windows=args.n_windows,
    )
