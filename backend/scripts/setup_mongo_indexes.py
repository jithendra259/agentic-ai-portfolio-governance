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
    
    def safe_create_index(col, keys, **kwargs):
        try:
            col.create_index(keys, **kwargs)
        except pymongo.errors.OperationFailure as e:
            if getattr(e, "code", None) == 85:
                pass
            else:
                raise

    # 1. TTL on graph_snapshots (e.g. 30 days expiration)
    if "graph_snapshots" not in db.list_collection_names():
        db.create_collection("graph_snapshots")
    safe_create_index(db["graph_snapshots"], "created_at", expireAfterSeconds=30*24*60*60)
    print(" - Configured TTL index on graph_snapshots.created_at")
    
    # 2. Compound on hitl_decisions
    if "hitl_decisions" not in db.list_collection_names():
        db.create_collection("hitl_decisions")
    safe_create_index(db["hitl_decisions"], [("universe_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
    print(" - Configured compound index on hitl_decisions (universe_id, timestamp)")
    
    # 3. Text index on pdf_chunks
    if "pdf_chunks" not in db.list_collection_names():
        db.create_collection("pdf_chunks")
    safe_create_index(db["pdf_chunks"], [("raw_text", pymongo.TEXT)])
    print(" - Configured TEXT index on pdf_chunks.raw_text")

    # 4. Standard lookups and ticker fast indexing
    if "ticker" in db.list_collection_names():
        safe_create_index(db["ticker"], "ticker")
        safe_create_index(db["ticker"], "universes")
        print(" - Configured indexes on ticker.ticker and ticker.universes")
    safe_create_index(db["blackboard_mpi"], "universe_id")
    safe_create_index(db["regime_transitions"], "universe_id")
    safe_create_index(db["backtest_results"], "universe_id")
    
    print("Done. All necessary indexes are configured.")

if __name__ == "__main__":
    setup_indexes()
