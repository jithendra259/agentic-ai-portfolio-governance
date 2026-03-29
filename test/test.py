import os
import sys

# Tell Python to look in the main folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymongo import MongoClient
from dotenv import load_dotenv
from orchestrator.langgraph_dag import SupervisoryOrchestrator

# 1. Connect to DB
load_dotenv()
client = MongoClient(
    os.getenv("MONGO_URI"), 
    serverSelectionTimeoutMS=5000,
    tls=True,
    tlsAllowInvalidCertificates=True 
)
db_collection = client["Stock_data"]["ticker"]

# 2. Initialize the LangGraph Orchestrator
orchestrator = SupervisoryOrchestrator(db_collection)

# 3. Fire the System (Testing our 2008 Crash again, but autonomously this time)
final_state = orchestrator.run_monthly_cycle(universe_id="U1", target_date="2008-10-15")

# 4. Verify the Blackboard State
print("📊 FINAL BLACKBOARD MEMORY STATE:")
print(f"-> Instability Index: {final_state['instability_index']:.4f}")
print(f"-> Penalty Applied (λ_t): {final_state['lambda_t']:.4f}")
print("\n✅ SECURED PORTFOLIO WEIGHTS:")
weights = final_state['optimal_weights']
print(weights[weights > 0.0].sort_values(ascending=False).to_string())