import os
import sys
import pandas as pd

# Tell Python to look in the main folder for 'agents'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymongo import MongoClient
from dotenv import load_dotenv
from agents.time_series_a1 import TimeSeriesAgent
from agents.graph_cag_a2 import GraphCAGAgent
from agents.optimizer_a3 import GCVaROptimizerAgent

# 1. Connect to DB
load_dotenv()
client = MongoClient(
    os.getenv("MONGO_URI"), 
    serverSelectionTimeoutMS=5000,
    tls=True,
    tlsAllowInvalidCertificates=True 
)
db_collection = client["Stock_data"]["ticker"]

# 2. Initialize Agents
agent1 = TimeSeriesAgent(db_collection)
agent2 = GraphCAGAgent(db_collection)
agent3 = GCVaROptimizerAgent() # Uses default Q1 paper parameters

universe = "U1"

# --- RUN THE PIPELINE ---
print("\n" + "="*50)
print(" SCENARIO: 2008 FINANCIAL CRISIS CRASH (High Instability)")
print("="*50)

# 1. Get math from Agent 1 (High Instability Date)
state_a1 = agent1.execute(universe_id=universe, target_date_str="2008-10-15", lookback_days=90)

# 2. Get graph from Agent 2
state_a2 = agent2.execute(universe_id=universe)

# 3. Run Optimizer (Agent 3)
state_a3 = agent3.execute(
    returns_df=state_a1["returns_df"], 
    c_vector=state_a2["c_vector"], 
    I_t=state_a1["instability_index"]
)

weights = state_a3["optimal_weights"]
print("\n✅ OPTIMAL CRISIS PORTFOLIO WEIGHTS (Non-Zero):")
print(weights[weights > 0.0].sort_values(ascending=False).to_string())

# Check what happened to PANW (The riskiest stock)
if "PANW" in weights:
    print(f"\n👉 Weight allocated to PANW (Risk Score 1.0): {weights['PANW']:.4f}")