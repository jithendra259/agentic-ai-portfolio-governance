"""Conversational router export.

Use ``src.orchestrator.langgraph_dag.SupervisoryOrchestrator`` for deterministic
paper/backtest runs. This module exports the chat-facing assistant used by the UI/API.
"""

from src.orchestrator.chatbot_orchestrator import portfolio_assistant

__all__ = ["portfolio_assistant"]
