"""Canonical deterministic DAG for thesis backtests and audit-friendly runs.

The conversational assistant exported by ``src.orchestrator.llm_router`` is the
UI-facing advisory layer. This module remains the reproducible, stepwise
orchestrator for deterministic Agent 1 -> Agent 2 -> Agent 3 -> Agent 4 runs.
"""

from typing import NotRequired, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from src.agents.explainer_a4 import GenerativeExplainerAgent
from src.agents.graph_rag_a2 import GraphRAGAgent
from src.agents.optimizer_a3 import GCVaROptimizerAgent
from src.agents.time_series_a1 import TimeSeriesAgent


class PortfolioState(TypedDict):
    universe_id: str
    target_date: str

    # Agent 1 outputs
    returns_df: NotRequired[pd.DataFrame]
    covariance_matrix: NotRequired[pd.DataFrame]
    instability_index: NotRequired[float]
    raw_instability_index: NotRequired[float]
    retained_assets: NotRequired[list[str]]
    dropped_assets: NotRequired[list[str]]
    mean_volatility: NotRequired[float]
    mean_correlation: NotRequired[float]
    mean_drawdown: NotRequired[float]

    # Agent 2 outputs
    c_vector: NotRequired[pd.Series]
    graph_method_used: NotRequired[str]
    graph_fallback_applied: NotRequired[bool]
    graph_node_count: NotRequired[int]
    graph_edge_count: NotRequired[int]
    bipartite_node_count: NotRequired[int]
    bipartite_edge_count: NotRequired[int]
    graph_context: NotRequired[dict]

    # Agent 3 outputs
    optimal_weights: NotRequired[pd.Series]
    strategy_weights: NotRequired[dict]
    lambda_t: NotRequired[float]
    turnover: NotRequired[dict]
    hitl_required: NotRequired[bool]
    hitl_crisis: NotRequired[bool]
    hitl_turnover: NotRequired[bool]
    hitl_reasons: NotRequired[list[str]]
    solver_status: NotRequired[str]
    solver_name: NotRequired[str]
    solve_time_s: NotRequired[float]
    max_weight_constraint: NotRequired[float]

    # Agent 4 outputs
    hitl_report: NotRequired[str]


class SupervisoryOrchestrator:
    """
    LangGraph state machine for deterministic graph-governed portfolio runs.
    """

    def __init__(self, db_collection):
        self.db_collection = db_collection
        self.agent1 = TimeSeriesAgent(db_collection)
        self.agent2 = GraphRAGAgent(db_collection)
        self.agent3 = GCVaROptimizerAgent()
        self.agent4 = GenerativeExplainerAgent()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(PortfolioState)

        workflow.add_node("Time_Series_Agent", self.run_agent_1)
        workflow.add_node("Graph_RAG_Agent", self.run_agent_2)
        workflow.add_node("GCVaR_Optimizer_Agent", self.run_agent_3)
        workflow.add_node("Generative_Explainer_Agent", self.run_agent_4)

        workflow.set_entry_point("Time_Series_Agent")
        workflow.add_edge("Time_Series_Agent", "Graph_RAG_Agent")
        workflow.add_edge("Graph_RAG_Agent", "GCVaR_Optimizer_Agent")
        workflow.add_conditional_edges(
            "GCVaR_Optimizer_Agent",
            self.hitl_router,
            {
                "crisis_detected": "Generative_Explainer_Agent",
                "calm_market": END,
            },
        )
        workflow.add_edge("Generative_Explainer_Agent", END)

        return workflow.compile()

    def run_agent_1(self, state: PortfolioState):
        result = self.agent1.execute(
            universe_id=state["universe_id"],
            target_date_str=state["target_date"],
        )
        return {
            **state,
            "returns_df": result["returns_df"],
            "covariance_matrix": result["covariance_matrix"],
            "instability_index": result["instability_index"],
            "raw_instability_index": result["raw_instability_index"],
            "retained_assets": result.get("retained_assets", []),
            "dropped_assets": result.get("dropped_assets", []),
            "mean_volatility": result.get("mean_volatility"),
            "mean_correlation": result.get("mean_correlation"),
            "mean_drawdown": result.get("mean_drawdown"),
        }

    def run_agent_2(self, state: PortfolioState):
        result = self.agent2.execute(universe_id=state["universe_id"])
        return {
            **state,
            "c_vector": result["c_vector"],
            "graph_method_used": result["method_used"],
            "graph_fallback_applied": result["fallback_applied"],
            "graph_node_count": result["graph_node_count"],
            "graph_edge_count": result["graph_edge_count"],
            "bipartite_node_count": result.get("bipartite_node_count"),
            "bipartite_edge_count": result.get("bipartite_edge_count"),
            "graph_context": result.get("graph_context", {}),
        }

    def run_agent_3(self, state: PortfolioState):
        result = self.agent3.execute(
            returns_df=state["returns_df"],
            c_vector=state["c_vector"],
            I_t=state["instability_index"],
            previous_weights=state.get("optimal_weights"),
        )
        return {
            **state,
            "optimal_weights": result["optimal_weights"],
            "strategy_weights": result.get("strategy_weights", {}),
            "lambda_t": result["lambda_t"],
            "turnover": result.get("turnover", {}),
            "hitl_required": result.get("hitl_required", False),
            "hitl_crisis": result.get("hitl_crisis", False),
            "hitl_turnover": result.get("hitl_turnover", False),
            "hitl_reasons": result.get("hitl_reasons", []),
            "solver_status": result.get("solver_status"),
            "solver_name": result.get("solver_name"),
            "solve_time_s": result.get("solve_time_s"),
            "max_weight_constraint": result.get("max_weight_constraint"),
        }

    def hitl_router(self, state: PortfolioState):
        """Route to the explainer only when notebook-style governance triggers fire."""
        if state.get("hitl_required"):
            instability = state["instability_index"]
            reasons = ", ".join(state.get("hitl_reasons", [])) or "governance trigger fired"
            print(
                f"WARNING: HITL review triggered (I_t = {instability:.4f}; {reasons}). "
                "Routing to the explainer."
            )
            return "crisis_detected"

        instability = state["instability_index"]
        print(
            f"INFO: No HITL escalation required (I_t = {instability:.4f})."
        )
        return "calm_market"

    def run_agent_4(self, state: PortfolioState):
        result = self.agent4.execute(state)
        return {**state, "hitl_report": result["hitl_report"]}

    def run_monthly_cycle(self, universe_id: str, target_date: str):
        print(f"\nSTARTING LangGraph orchestrator for {target_date}")
        initial_state: PortfolioState = {
            "universe_id": universe_id,
            "target_date": target_date,
        }
        final_state = self.graph.invoke(initial_state)
        print(f"EXECUTION complete for {target_date}\n")
        return final_state
