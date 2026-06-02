import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
db = client["Stock_data"]
col = db["ticker"]

start_str = "2020-01-01"
end_str = "2020-01-15"

print("Running filtered query with $ifNull...")
try:
    doc = col.find_one({"ticker": "AAPL"}, {
        "ticker": 1,
        "historical_prices": {
            "$filter": {
                "input": "$historical_prices",
                "as": "hp",
                "cond": {
                    "$and": [
                        {"$gte": [{"$ifNull": ["$$hp.Date", "$$hp.date"]}, start_str]},
                        {"$lte": [{"$ifNull": ["$$hp.Date", "$$hp.date"]}, end_str]}
                    ]
                }
            }
        }
    })
    print("AAPL Historical Prices filtered:")
    for hp in doc.get("historical_prices", []):
        print(hp)
except Exception as e:
    print("Query failed:", e)
