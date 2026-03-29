#!/usr/bin/env python3
"""
Test MongoDB Connection
Run this FIRST to verify your connection works
"""

import os
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from dotenv import load_dotenv
import logging
from pathlib import Path
import ssl
import certifi

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

def normalize_mongo_uri(raw_uri):
    if not raw_uri:
        return None
    uri = raw_uri.strip().strip("\"'").replace(" ", "")
    if "mongodb+srv://mongodb+srv://" in uri:
        uri = uri.replace("mongodb+srv://mongodb+srv://", "mongodb+srv://", 1)
    return uri
MONGO_URI = normalize_mongo_uri(os.getenv("MONGO_URI"))

if not MONGO_URI:
    logger.error("❌ MONGO_URI not found in .env file")
    logger.info("\nCreate .env file with:")
    logger.info("MONGO_URI=mongodb+srv://<username>:<password>@cluster0.yqwhfm6.mongodb.net/?retryWrites=true&w=majority")
    exit(1)

logger.info("\n" + "="*60)
logger.info("🧪 Testing MongoDB Connection")
logger.info("="*60)

logger.info(f"\n🔗 Connecting to: {MONGO_URI[:50]}...")

try:
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
        tls=True,
        tlsCAFile=certifi.where(),
        appname="agentic-ai-portfolio-governance"
    )
    client.admin.command("ping")
    
    logger.info("✅ Connection Successful!\n")
    
    # Show databases
    db_names = client.list_database_names()
    logger.info(f"📋 Available Databases:")
    for db_name in db_names:
        logger.info(f"   - {db_name}")
    
    # Check our specific database
    db = client["Stock_data"]
    collections = db.list_collection_names()
    
    logger.info(f"\n📦 Collections in 'Stock_data':")
    for collection_name in collections:
        logger.info(f"   - {collection_name}")
    
    # Count documents in ticker collection
    if "ticker" in collections:
        ticker_collection = db["ticker"]
        count = ticker_collection.count_documents({})
        logger.info(f"\n📊 Documents in 'ticker' collection: {count}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ Ready to run: python mongodb_population_complete.py")
    logger.info("="*60 + "\n")
    
except ServerSelectionTimeoutError as e:
    logger.error(f"\n❌ Connection Failed: {e}\n")
    logger.error("⚠️  Troubleshooting:")
    logger.error("   1. Check MONGO_URI in .env file")
    logger.error("   2. Verify cluster is running on Atlas")
    logger.error("   3. Check username/password are correct")
    logger.error("   4. Verify Atlas Network Access includes your public IP")
    logger.error("   5. Verify local firewall / antivirus / proxy is not intercepting TLS on port 27017")
    logger.error("   6. Test the same URI in MongoDB Compass")
    logger.error(f"\nTLS runtime: Python {os.sys.version.split()[0]} | {ssl.OPENSSL_VERSION}")
    logger.error(f"CA bundle: {certifi.where()}\n")
    exit(1)
except Exception as e:
    logger.error(f"\n❌ Connection Failed: {e}\n")
    exit(1)
