"""
Test market data retrieval across all layers:
1. MongoDB Atlas connectivity & ticker collection inventory (universes U1..U11, sectors, price histories)
2. Tool-level market functions: list_available_universes, list_available_sectors, get_stocks_by_universe
3. Live market data fetching via yfinance (symbols, quotes, historical series)
4. Fast analytics data loading for all default dashboard universes (U1-U5)
"""
import sys
import os
import json
import time
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv
load_dotenv(backend_dir.parent / ".env")
load_dotenv(backend_dir / ".env")

print("=" * 70)
print("TESTING MARKET DATA CAPABILITIES")
print("=" * 70)

# ---------------------------------------------------------
# Test 1: Direct MongoDB Connection & Ticker Inventory
# ---------------------------------------------------------
print("\n[TEST 1] Testing MongoDB Atlas Connection & Ticker Inventory...")
from pymongo import MongoClient

mongo_uri = (os.getenv("MONGO_URI") or "").strip()
print(f"MongoDB URI present: {bool(mongo_uri)}")

t0 = time.time()
try:
    client = MongoClient(
        mongo_uri,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=8000,
    )
    # Ping
    client.admin.command('ping')
    print(f" -> MongoDB Ping SUCCESS in {time.time() - t0:.2f}s")
    
    db = client["Stock_data"]
    collections = db.list_collection_names()
    print(f" -> Available Collections: {collections}")
    
    if "ticker" in collections:
        total_tickers = db["ticker"].count_documents({})
        print(f" -> Total tickers in 'ticker' collection: {total_tickers}")
        
        # Check universes
        pipeline = [
            {"$unwind": "$universes"},
            {"$group": {"_id": "$universes", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        universe_dist = list(db["ticker"].aggregate(pipeline))
        print(" -> Universes found in MongoDB:")
        for item in universe_dist:
            print(f"    * {item['_id']}: {item['count']} tickers")
            
        # Sample document check
        sample = db["ticker"].find_one({}, {"ticker": 1, "symbol": 1, "universes": 1, "sector": 1, "dates": {"$slice": 3}, "close": {"$slice": 3}})
        if sample:
            print(f" -> Sample ticker document: ticker={sample.get('ticker') or sample.get('symbol')}, universes={sample.get('universes')}, dates_sample_len={len(sample.get('dates', []))}")
    else:
        print(" -> WARNING: 'ticker' collection not found in Stock_data.")
except Exception as e:
    print(f" -> MongoDB Test FAILED: {type(e).__name__}: {e}")

# ---------------------------------------------------------
# Test 2: Live Data Tools - list_available_universes & get_stocks_by_universe
# ---------------------------------------------------------
print("\n[TEST 2] Testing live_data_tools Market Listing Functions...")
try:
    from src.agents.live_data_tools import (
        list_available_universes,
        list_available_sectors,
        get_stocks_by_universe,
        get_universe_overview,
    )
    
    print("\nExecuting list_available_universes()...")
    universes_output = list_available_universes.invoke({})
    print(universes_output[:400] + ("..." if len(universes_output) > 400 else ""))
    
    print("\nExecuting list_available_sectors()...")
    sectors_output = list_available_sectors.invoke({})
    print(sectors_output[:300] + ("..." if len(sectors_output) > 300 else ""))
    
    print("\nExecuting get_stocks_by_universe('U1')...")
    u1_stocks = get_stocks_by_universe.invoke({"universe": "U1"})
    print(u1_stocks[:350] + ("..." if len(u1_stocks) > 350 else ""))

except Exception as e:
    print(f" -> Tools Test FAILED: {type(e).__name__}: {e}")

# ---------------------------------------------------------
# Test 3: Live Market Data via yfinance
# ---------------------------------------------------------
print("\n[TEST 3] Testing Live Market Fetching via yfinance...")
try:
    from src.agents.live_data_tools import get_yfinance_market_data, get_market_data_bundle
    
    test_symbols = ["AAPL", "NVDA", "SPY"]
    print(f"Fetching market data bundle for {test_symbols}...")
    t_yf = time.time()
    bundle_str = get_market_data_bundle.invoke({"symbols": test_symbols, "metrics": ["price", "pe_ratio", "market_cap"]})
    print(f" -> Fetched in {time.time() - t_yf:.2f}s:")
    print(bundle_str[:500] + ("..." if len(bundle_str) > 500 else ""))
    
    print("\nFetching detailed quote for 'MSFT'...")
    msft_data = get_yfinance_market_data.invoke({"symbol": "MSFT", "data_type": "quote"})
    print(msft_data[:350] + ("..." if len(msft_data) > 350 else ""))

except Exception as e:
    print(f" -> yfinance Test FAILED: {type(e).__name__}: {e}")

# ---------------------------------------------------------
# Test 4: Dashboard Universes (U1 to U5) End-to-End Resolution
# ---------------------------------------------------------
print("\n[TEST 4] Testing Dashboard Universes (U1 to U5) Data Availability...")
dashboard_universes = {
    "U1 (Tech)": ["AAPL", "MSFT", "NVDA", "AMZN", "JPM"],
    "U2 (Financials)": ["BAC", "GS", "WFC", "BLK", "AXP"],
    "U3 (Healthcare)": ["JNJ", "PFE", "UNH", "MRK", "ABBV"],
    "U4 (Consumer)": ["TSLA", "MCD", "NKE", "SBUX", "DAL"],
    "U5 (Industrials)": ["GE", "HON", "CAT", "BA", "LMT"],
}

try:
    from src.agents.ticker_data_service import get_dynamic_price_documents
    from src.agents.live_data_tools import _find_price_documents_with_retry
    
    for uname, tickers in dashboard_universes.items():
        t_sub = time.time()
        docs = get_dynamic_price_documents(
            tickers=tickers,
            mongo_docs_finder=_find_price_documents_with_retry,
        )
        print(f" -> Universe {uname}: successfully retrieved {len(docs)}/{len(tickers)} ticker price series in {time.time() - t_sub:.2f}s")
        if docs:
            first_ticker = docs[0].get("ticker")
            points = len(docs[0].get("historical_prices") or docs[0].get("close") or [])
            src = docs[0].get("_source", "MongoDB")
            print(f"    Sample: {first_ticker} with {points} price points (source: {src})")
except Exception as e:
    print(f" -> Dashboard Universes Test: {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)
