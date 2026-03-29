import os
import sys

# Tell Python to look in the main folder for 'agents'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymongo import MongoClient
from dotenv import load_dotenv
from agents.graph_cag_a2 import GraphCAGAgent

# 1. Connect to DB
load_dotenv()
client = MongoClient(
    os.getenv("MONGO_URI"), 
    serverSelectionTimeoutMS=5000,
    tls=True,
    tlsAllowInvalidCertificates=True 
)
db_collection = client["Stock_data"]["ticker"]

# 2. Initialize Agent 2
agent2 = GraphCAGAgent(db_collection)

print("\n--- TEST: CAG Epistemic Graph Extraction ---")
state_graph = agent2.execute(universe_id="U1")

# Ensure the c_t vector aligns with the I_t vector from Agent 1
print(f"\n✅ Total stocks scored: {len(state_graph['c_vector'])}")