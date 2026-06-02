import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
db = client["Stock_data"]
col = db["ticker"]

print("Fetching one ticker...")
doc = col.find_one({}, {"ticker": 1, "historical_prices": {"$slice": 5}})
print("Ticker doc:")
import pprint
pprint.pprint(doc)
