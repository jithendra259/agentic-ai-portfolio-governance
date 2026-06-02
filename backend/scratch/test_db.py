import os
import time
from dotenv import load_dotenv

load_dotenv()

postgres_url = os.getenv("SUPABASE_POSTGRES_URL")
print("postgres_url:", postgres_url)

if not postgres_url:
    print("No SUPABASE_POSTGRES_URL found in .env")
    exit(0)

try:
    from psycopg_pool import ConnectionPool
except ImportError:
    print("psycopg_pool is not installed")
    exit(0)

print("Attempting to connect with 5s timeout...")
start = time.time()
try:
    pool = ConnectionPool(
        conninfo=postgres_url,
        min_size=0,
        max_size=2,
        open=True,
        timeout=5.0,
        kwargs={"autocommit": True, "prepare_threshold": None}
    )
    print("Pool created in", time.time() - start, "seconds")
    print("Getting connection...")
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            print("Query executed successfully in", time.time() - start, "seconds")
except Exception as e:
    print("Failed in", time.time() - start, "seconds. Error:", e)
