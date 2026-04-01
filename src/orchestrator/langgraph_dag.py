"""Canonical deterministic DAG for thesis backtests and audit-friendly runs.

The conversational assistant exported by ``src.orchestrator.llm_router`` is the
UI-facing advisory layer. This module remains the reproducible, stepwise
orchestrator for deterministic Agent 1 -> Agent 2 -> Agent 3 -> Agent 4 runs.
"""

from typing import NotRequired, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from agents.explainer_a4 import GenerativeExplainerAgent
from agents.graph_cag_a2 import GraphCAGAgent
from agents.optimizer_a3 import GCVaROptimizerAgent
from agents.time_series_a1 import TimeSeriesAgent


class PortfolioState(TypedDict):
    universe_id: str
    target_date: str

    # Agent 1 outputs
    returns_df: NotRequired[pd.DataFrame]
    covariance_matrix: NotRequired[pd.DataFrame]
    instability_index: NotRequired[float]
    raw_instability_index: NotRequired[float]

    # Agent 2 outputs
    c_vector: NotRequired[pd.Series]
    graph_method_used: NotRequired[str]
    graph_fallback_applied: NotRequired[bool]
    graph_node_count: NotRequired[int]
    graph_edge_count: NotRequired[int]

    # Agent 3 outputs
    optimal_weights: NotRequired[pd.Series]
    lambda_t: NotRequired[float]

    # Agent 4 outputs
    hitl_report: NotRequired[str]


class SupervisoryOrchestrator:
    """
    LangGraph state machine for deterministic CAG portfolio governance.
    """

    def __init__(self, db_collection):
        self.db_collection = db_collection
        self.agent1 = TimeSeriesAgent(db_collection)
        self.agent2 = GraphCAGAgent(db_collection)
        self.agent3 = GCVaROptimizerAgent()
        self.agent4 = GenerativeExplainerAgent()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(PortfolioState)

        workflow.add_node("Time_Series_Agent", self.run_agent_1)
        workflow.add_node("Graph_CAG_Agent", self.run_agent_2)
        workflow.add_node("GCVaR_Optimizer_Agent", self.run_agent_3)
        workflow.add_node("Generative_Explainer_Agent", self.run_agent_4)

        workflow.set_entry_point("Time_Series_Agent")
        workflow.add_edge("Time_Series_Agent", "Graph_CAG_Agent")
        workflow.add_edge("Graph_CAG_Agent", "GCVaR_Optimizer_Agent")
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
        }

    def run_agent_3(self, state: PortfolioState):
        result = self.agent3.execute(
            returns_df=state["returns_df"],
            c_vector=state["c_vector"],
            I_t=state["instability_index"],
        )
        return {
            **state,
            "optimal_weights": result["optimal_weights"],
            "lambda_t": result["lambda_t"],
        }

    def hitl_router(self, state: PortfolioState):
        """Route to the LLM explainer only when instability exceeds the threshold."""
        instability = state["instability_index"]
        if instability >= 0.5:
            print(
                f"WARNING: Crisis regime detected (I_t = {instability:.4f}). "
                "Routing to the HITL explainer."
            )
            return "crisis_detected"

        print(
            f"INFO: Calm market (I_t = {instability:.4f}). "
            "No HITL escalation required."
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
