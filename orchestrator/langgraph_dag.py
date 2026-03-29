from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
import pandas as pd

# Import our deterministic CAG agents
from agents.time_series_a1 import TimeSeriesAgent
from agents.graph_cag_a2 import GraphCAGAgent
from agents.optimizer_a3 import GCVaROptimizerAgent

# 1. Define the Blackboard (State Memory)
# This guarantees zero data leakage between months. It resets every run.
class PortfolioState(TypedDict):
    universe_id: str
    target_date: str
    
    # Agent 1 Outputs
    returns_df: Any  # pandas DataFrame
    covariance_matrix: Any # pandas DataFrame
    instability_index: float
    
    # Agent 2 Outputs
    c_vector: Any # pandas Series
    
    # Agent 3 Outputs
    optimal_weights: Any # pandas Series
    lambda_t: float

class SupervisoryOrchestrator:
    """
    LangGraph State Machine: Wires the autonomous agents into a strict DAG architecture.
    """
    def __init__(self, db_collection):
        self.db_collection = db_collection
        
        # Initialize the Agents
        self.agent1 = TimeSeriesAgent(db_collection)
        self.agent2 = GraphCAGAgent(db_collection)
        self.agent3 = GCVaROptimizerAgent() # Uses default Q1 paper parameters
        
        # Compile the DAG
        self.graph = self._build_graph()

    def _build_graph(self):
        # Initialize the State Machine
        workflow = StateGraph(PortfolioState)
        
        # Add the Nodes (The Agents)
        workflow.add_node("Time_Series_Agent", self.run_agent_1)
        workflow.add_node("Graph_CAG_Agent", self.run_agent_2)
        workflow.add_node("GCVaR_Optimizer_Agent", self.run_agent_3)
        
        # Define the Edges (The Execution Order)
        workflow.set_entry_point("Time_Series_Agent")
        workflow.add_edge("Time_Series_Agent", "Graph_CAG_Agent")
        workflow.add_edge("Graph_CAG_Agent", "GCVaR_Optimizer_Agent")
        workflow.add_edge("GCVaR_Optimizer_Agent", END)
        
        return workflow.compile()

    # --- NODE WRAPPER FUNCTIONS ---
    def run_agent_1(self, state: PortfolioState):
        """Node 1: Executes Time-Series Agent and writes to Blackboard."""
        result = self.agent1.execute(
            universe_id=state["universe_id"], 
            target_date_str=state["target_date"]
        )
        return {
            "returns_df": result["returns_df"],
            "covariance_matrix": result["covariance_matrix"],
            "instability_index": result["instability_index"]
        }

    def run_agent_2(self, state: PortfolioState):
        """Node 2: Executes Graph Agent and writes to Blackboard."""
        result = self.agent2.execute(universe_id=state["universe_id"])
        return {
            "c_vector": result["c_vector"]
        }

    def run_agent_3(self, state: PortfolioState):
        """Node 3: Executes Convex Optimizer using Blackboard data."""
        result = self.agent3.execute(
            returns_df=state["returns_df"],
            c_vector=state["c_vector"],
            I_t=state["instability_index"]
        )
        return {
            "optimal_weights": result["optimal_weights"],
            "lambda_t": result["lambda_t"]
        }

    def run_monthly_cycle(self, universe_id: str, target_date: str):
        """Triggers a single autonomous month of execution."""
        print(f"\n🚀 --- STARTING LANGGRAPH ORCHESTRATOR FOR {target_date} ---")
        initial_state = {
            "universe_id": universe_id,
            "target_date": target_date
        }
        
        # Run the compiled LangGraph
        final_state = self.graph.invoke(initial_state)
        
        print(f"🏁 --- EXECUTION COMPLETE FOR {target_date} ---\n")
        return final_state