import pandas as pd
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END

# Import our deterministic CAG agents and the Generative LLM agent
from agents.time_series_a1 import TimeSeriesAgent
from agents.graph_cag_a2 import GraphCAGAgent
from agents.optimizer_a3 import GCVaROptimizerAgent
from agents.explainer_a4 import GenerativeExplainerAgent

# 1. Define the Blackboard (State Memory)
class PortfolioState(TypedDict):
    universe_id: str
    target_date: str
    
    # Agent 1 Outputs
    returns_df: Any
    covariance_matrix: Any
    instability_index: float
    
    # Agent 2 Outputs
    c_vector: Any
    
    # Agent 3 Outputs
    optimal_weights: Any
    lambda_t: float
    
    # Agent 4 Outputs (New!)
    hitl_report: str 

class SupervisoryOrchestrator:
    """
    LangGraph State Machine: Wires the autonomous agents into a strict DAG architecture
    with conditional Generative AI routing for Human-in-the-Loop governance.
    """
    def __init__(self, db_collection):
        self.db_collection = db_collection
        
        # Initialize ALL Agents
        self.agent1 = TimeSeriesAgent(db_collection)
        self.agent2 = GraphCAGAgent(db_collection)
        self.agent3 = GCVaROptimizerAgent() 
        self.agent4 = GenerativeExplainerAgent() # The Local Ollama Agent
        
        # Compile the DAG
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(PortfolioState)
        
        # Add the Nodes
        workflow.add_node("Time_Series_Agent", self.run_agent_1)
        workflow.add_node("Graph_CAG_Agent", self.run_agent_2)
        workflow.add_node("GCVaR_Optimizer_Agent", self.run_agent_3)
        workflow.add_node("Generative_Explainer_Agent", self.run_agent_4)
        
        # Define the Edges
        workflow.set_entry_point("Time_Series_Agent")
        workflow.add_edge("Time_Series_Agent", "Graph_CAG_Agent")
        workflow.add_edge("Graph_CAG_Agent", "GCVaR_Optimizer_Agent")
        
        # 🚨 THE CONDITIONAL HITL ROUTER
        workflow.add_conditional_edges(
            "GCVaR_Optimizer_Agent",
            self.hitl_router,
            {
                "crisis_detected": "Generative_Explainer_Agent",
                "calm_market": END
            }
        )
        workflow.add_edge("Generative_Explainer_Agent", END)
        
        return workflow.compile()

    # --- NODE WRAPPER FUNCTIONS ---
    def run_agent_1(self, state: PortfolioState):
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
        result = self.agent2.execute(universe_id=state["universe_id"])
        return {"c_vector": result["c_vector"]}

    def run_agent_3(self, state: PortfolioState):
        result = self.agent3.execute(
            returns_df=state["returns_df"],
            c_vector=state["c_vector"],
            I_t=state["instability_index"]
        )
        return {
            "optimal_weights": result["optimal_weights"],
            "lambda_t": result["lambda_t"]
        }

    def hitl_router(self, state: PortfolioState):
        """Routes to the LLM Explainer ONLY if a crisis is detected."""
        if state["instability_index"] >= 0.5:
            print(f"⚠️ CRISIS REGIME DETECTED (I_t = {state['instability_index']:.4f}). Routing to LLM HITL Explainer.")
            return "crisis_detected"
        else:
            print(f"✅ Calm Market (I_t = {state['instability_index']:.4f}). Auto-approving trades. No HITL required.")
            return "calm_market"

    def run_agent_4(self, state: PortfolioState):
        result = self.agent4.execute(state)
        return {"hitl_report": result["hitl_report"]}

    def run_monthly_cycle(self, universe_id: str, target_date: str):
        print(f"\n🚀 --- STARTING LANGGRAPH ORCHESTRATOR FOR {target_date} ---")
        initial_state = {
            "universe_id": universe_id,
            "target_date": target_date
        }
        
        final_state = self.graph.invoke(initial_state)
        
        print(f"🏁 --- EXECUTION COMPLETE FOR {target_date} ---\n")
        return final_state