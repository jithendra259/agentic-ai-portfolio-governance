import os
import pymongo
from dotenv import load_dotenv

def setup_indexes():
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("Error: MONGO_URI is missing from .env")
        return
        
    client = pymongo.MongoClient(
        mongo_uri,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=10000
    )
    
    # We use "Stock_data" as the main DB from config
    db = client["Stock_data"]
    
    print("Setting up MongoDB indexes for Agentic AI Portfolio Governance...")
    
    # 1. TTL on graph_snapshots (e.g. 30 days expiration)
    if "graph_snapshots" not in db.list_collection_names():
        db.create_collection("graph_snapshots")
    db["graph_snapshots"].create_index("created_at", expireAfterSeconds=30*24*60*60)
    print(" - Created TTL index on graph_snapshots.created_at")
    
    # 2. Compound on hitl_decisions
    if "hitl_decisions" not in db.list_collection_names():
        db.create_collection("hitl_decisions")
    db["hitl_decisions"].create_index([("universe_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
    print(" - Created compound index on hitl_decisions (universe_id, timestamp)")
    
    # 3. Text index on pdf_chunks
    if "pdf_chunks" not in db.list_collection_names():
        db.create_collection("pdf_chunks")
    db["pdf_chunks"].create_index([("raw_text", pymongo.TEXT)])
    print(" - Created TEXT index on pdf_chunks.raw_text")

    # 4. Standard lookups (optional but good practice)
    db["blackboard_mpi"].create_index("universe_id")
    db["regime_transitions"].create_index("universe_id")
    db["backtest_results"].create_index("universe_id")
    
    print("Done. All necessary indexes are configured.")

if __name__ == "__main__":
    setup_indexes()
