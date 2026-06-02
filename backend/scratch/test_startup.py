import time
print("1. Starting test_startup...")

start = time.time()
import os
from dotenv import load_dotenv
load_dotenv()
print("2. Environment loaded in", time.time() - start, "seconds")

# Let's time individual imports
import psycopg_pool
print("3. psycopg_pool imported in", time.time() - start, "seconds")

from pymongo import MongoClient
print("4. pymongo imported in", time.time() - start, "seconds")

# Let's check _list_installed_ollama_models
import subprocess
print("5. Running ollama list...")
ollama_start = time.time()
try:
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    print("5b. ollama list completed in", time.time() - ollama_start, "seconds")
except Exception as e:
    print("5b. ollama list failed in", time.time() - ollama_start, "seconds:", e)

# Test MongoDB connection
mongo_uri = (os.getenv("MONGO_URI") or "").strip()
print("6. Connecting to MongoDB...")
mongo_start = time.time()
try:
    mongo_client = MongoClient(
        mongo_uri,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=10000,
    )
    mongo_client.admin.command("ping")
    print("6b. MongoDB pinged in", time.time() - mongo_start, "seconds")
except Exception as e:
    print("6b. MongoDB connection failed in", time.time() - mongo_start, "seconds:", e)

# Test Postgres Connection Pool
postgres_url = (os.getenv("SUPABASE_POSTGRES_URL") or "").strip()
print("7. Testing Postgres connection pool setup...")
pg_start = time.time()
pool = None
try:
    from psycopg_pool import ConnectionPool
    pool = ConnectionPool(
        conninfo=postgres_url,
        min_size=0,
        max_size=2,
        open=True,
        timeout=5.0,
        kwargs={"autocommit": True, "prepare_threshold": None}
    )
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
    print("7b. Postgres pool checked in", time.time() - pg_start, "seconds")
except Exception as e:
    print("7b. Postgres pool setup failed in", time.time() - pg_start, "seconds:", e)

# Let's time checkpointer setup
if pool:
    print("8. Testing checkpointer setup...")
    cp_start = time.time()
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        print("8b. PostgresSaver checkpointer setup in", time.time() - cp_start, "seconds")
    except Exception as e:
        print("8b. PostgresSaver checkpointer setup failed in", time.time() - cp_start, "seconds:", e)

print("Total startup check finished in", time.time() - start, "seconds")
if pool:
    pool.close()
