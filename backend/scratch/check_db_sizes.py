import os
import time
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
db = client["Stock_data"]
col = db["ticker"]

tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JNJ", "V", "PG", "XOM", "HD", "MA", "UNH", "PFE"]

print("1. Standard query time for 16 tickers:")
start = time.time()
docs1 = list(col.find({"ticker": {"$in": tickers}}, {
    "ticker": 1,
    "historical_prices.Date": 1,
    "historical_prices.Open": 1,
    "historical_prices.High": 1,
    "historical_prices.Low": 1,
    "historical_prices.Close": 1,
    "historical_prices.Volume": 1,
}))
end = time.time()
print(f"Fetched {len(docs1)} docs in {end - start:.4f} seconds")
if docs1:
    total_elements = sum(len(d.get("historical_prices", [])) for d in docs1)
    print(f"Total elements: {total_elements}")

print("\n2. Filtered projection query time for 16 tickers (2020-01-01 to 2025-12-31):")
start = time.time()
docs2 = list(col.find({"ticker": {"$in": tickers}}, {
    "ticker": 1,
    "historical_prices": {
        "$filter": {
            "input": "$historical_prices",
            "as": "hp",
            "cond": {
                "$and": [
                    {"$gte": ["$$hp.Date", "2020-01-01"]},
                    {"$lte": ["$$hp.Date", "2025-12-31"]}
                ]
            }
        }
    }
}))
end = time.time()
print(f"Fetched {len(docs2)} docs in {end - start:.4f} seconds")
if docs2:
    total_elements = sum(len(d.get("historical_prices", [])) for d in docs2)
    print(f"Total elements: {total_elements}")
