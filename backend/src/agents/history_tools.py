import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from langchain_core.tools import tool
from src.memory.mongodb_memory_layer import MongoMemoryManager

logger = logging.getLogger(__name__)

# Re-use the existing memory manager or create a fresh one
memory_manager = MongoMemoryManager()


@tool
def get_user_analysis_history(limit: int = 10) -> str:
    """
    Search the database for lists of previous portfolio governance analyses, 
    optimized weights, and risk assessments performed by any user in this environment.
    Use this to answer questions like 'Which universes did I analyze before?' or 
    'What was my last risk assessment target date?'.
    """
    if not memory_manager.is_available:
        return "Memory storage is currently unavailable. I cannot look up history."

    try:
        db = memory_manager._db
        # We look into regime_patterns as it's the result of run_full_governance_pipeline
        patterns = list(db["regime_patterns"].find().sort("created_at", -1).limit(limit))
        
        if not patterns:
            return "No previous governance analyses found in the database records."

        lines = ["Here are the recent portfolio governance analyses found in the database:"]
        for p in patterns:
            date_record = p.get("created_at")
            if isinstance(date_record, datetime):
                date_str = date_record.strftime("%Y-%m-%d %H:%M")
            else:
                date_str = "Unknown Date"
            
            target_date = p.get("target_date", "Unknown")
            regime = p.get("regime_type", "Unknown")
            tickers = list(p.get("weights", {}).keys())
            ticker_sample = ", ".join(tickers[:8])
            if len(tickers) > 8:
                ticker_sample += "..."

            lines.append(
                f"- [{date_str}] Targeted: {target_date} | Regime: {regime} | "
                f"Tickers ({len(tickers)}): {ticker_sample}"
            )
        
        return "\n".join(lines)
    except Exception as exc:
        logger.error("Failed to retrieve analysis history: %s", exc)
        return f"Error retrieving history: {str(exc)}"


@tool
def get_detailed_past_weights(target_date: str, ticker_subset: Optional[List[str]] = None) -> str:
    """
    Retrieve the specific portfolio weights and instability metrics from a 
    past governance run for a given target date. Use this to recall exact 
    allocations from previous sessions.
    """
    if not memory_manager.is_available:
        return "Memory storage is currently unavailable."

    try:
        db = memory_manager._db
        query = {"target_date": str(target_date)}
        # If we have many runs for the same date, take the latest one
        record = db["regime_patterns"].find_one(query, sort=[("created_at", -1)])
        
        if not record:
            return f"No records found for target date {target_date}."

        weights = record.get("weights", {})
        instability = record.get("instability_index", 0.0)
        regime = record.get("regime_type", "Unknown")
        
        # Filter by subset if provided
        if ticker_subset:
            subset_upper = [t.upper() for t in ticker_subset]
            weights = {k: v for k, v in weights.items() if k.upper() in subset_upper}

        if not weights:
            return f"No weight data matches your ticker subset for {target_date}."

        lines = [
            f"Historical Governance Result for {target_date}:",
            f"- Market Regime: {regime}",
            f"- Instability Index: {instability:.4f}",
            "- Allocations:"
        ]
        
        # Sort weights descending
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for ticker, weight in sorted_weights:
            lines.append(f"  - {ticker}: {weight*100:.2f}%")
            
        return "\n".join(lines)
    except Exception as exc:
        logger.error("Failed to retrieve detailed weights: %s", exc)
        return f"Error: {str(exc)}"
