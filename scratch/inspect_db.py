import os
import json
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

class MongoEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

def inspect_db():
    if not MONGO_URI:
        print("MONGO_URI not found!")
        return

    client = MongoClient(MONGO_URI)
    
    dbs = ['Stock_data', 'checkpointing_db']
    
    for db_name in dbs:
        print(f"\n" + "="*40)
        print(f"DATABASE: {db_name}")
        print("="*40)
        db = client[db_name]
        collections = db.list_collection_names()
        
        for coll_name in collections:
            count = db[coll_name].count_documents({})
            print(f"- {coll_name}: {count} documents")
            if count > 0:
                sample = db[coll_name].find_one()
                print(f"  Sample: {json.dumps(sample, indent=2, cls=MongoEncoder)[:800]}...")

if __name__ == "__main__":
    inspect_db()
