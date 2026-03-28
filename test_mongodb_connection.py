#!/usr/bin/env python3
"""
Test MongoDB Connection
Run this FIRST to verify your connection works
"""

import os
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Load .env
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    logger.error("❌ MONGO_URI not found in .env file")
    logger.info("\nCreate .env file with:")
    logger.info("MONGO_URI=mongodb+srv://kandulajithendrasubramanyam_db_user:6TzH8AQkMVvirJhv@cluster0.yqwhfm6.mongodb.net/?retryWrites=true&w=majority")
    exit(1)

logger.info("\n" + "="*60)
logger.info("🧪 Testing MongoDB Connection")
logger.info("="*60)

logger.info(f"\n🔗 Connecting to: {MONGO_URI[:50]}...")

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
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
    
except Exception as e:
    logger.error(f"\n❌ Connection Failed: {e}\n")
    logger.error("⚠️  Troubleshooting:")
    logger.error("   1. Check MONGO_URI in .env file")
    logger.error("   2. Verify cluster is running on Atlas")
    logger.error("   3. Check username/password are correct")
    logger.error("   4. Verify network access (IP whitelist)\n")
    exit(1)