import os
import sys

# 🚨 THE FIX: Tell Python to look in the folder above this one
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymongo import MongoClient
from dotenv import load_dotenv
from agents.time_series_a1 import TimeSeriesAgent

# 1. Connect to DB (with the SSL bypass fix)
load_dotenv()
client = MongoClient(
    os.getenv("MONGO_URI"), 
    serverSelectionTimeoutMS=5000,
    tls=True,
    tlsAllowInvalidCertificates=True 
)
db_collection = client["Stock_data"]["ticker"]

# 2. Initialize Agent
agent1 = TimeSeriesAgent(db_collection)

print("\n--- TEST 1: Calm Market (Early 2006) ---")
state_calm = agent1.execute(universe_id="U1", target_date_str="2006-05-01", lookback_days=90)

print("\n--- TEST 2: The 2008 Financial Crisis Crash ---")
state_crash = agent1.execute(universe_id="U1", target_date_str="2008-10-15", lookback_days=90)