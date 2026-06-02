import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(dotenv_path=BACKEND_DIR / ".env")

def run_tests():
    logger.info("Starting Supabase Integration Tests...")
    
    postgres_url = os.getenv("SUPABASE_POSTGRES_URL")
    if not postgres_url:
        logger.error("SUPABASE_POSTGRES_URL is not configured in .env file.")
        sys.exit(1)
        
    logger.info(f"Supabase PostgreSQL URL detected: {postgres_url[:30]}...")

    # 1. Verify credentials parsing
    from src.memory.mongodb_memory_layer import get_clean_postgres_url, _test_and_get_pool
    clean_url = get_clean_postgres_url(postgres_url)
    logger.info(f"Cleaned database URL: {clean_url[:30]}...")

    # 2. Establish connection pool
    logger.info("Attempting to connect to PostgreSQL connection pool...")
    pool = _test_and_get_pool(postgres_url)
    if not pool:
        logger.error("Failed to establish connection pool to Supabase PostgreSQL.")
        sys.exit(1)
    logger.info("Successfully established connection pool to Supabase PostgreSQL.")

    # 3. Verify checkpointer tables setup
    logger.info("Initializing PostgresSaver checkpointer...")
    from langgraph.checkpoint.postgres import PostgresSaver
    try:
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        logger.info("Successfully initialized PostgresSaver and ran checkpointer.setup().")
        
        # Verify checkpointer tables exist in public schema
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name LIKE 'checkpoint%';
                """)
                tables = [r[0] for r in cur.fetchall()]
                logger.info(f"Detected checkpoint tables: {tables}")
                if not tables:
                    logger.error("Checkpoint tables were not created!")
                    sys.exit(1)
    except Exception as e:
        logger.error(f"Checkpointer validation failed: {e}")
        sys.exit(1)

    # 4. Verify hybrid memory manager caching & visualizations tables setup
    logger.info("Testing Hybrid Memory Manager caching...")
    from src.memory.mongodb_memory_layer import MongoMemoryManager
    try:
        memory = MongoMemoryManager(postgres_url=postgres_url)
        
        # Verify custom cache tables exist
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name IN ('plan_cache', 'visualizations');
                """)
                tables = [r[0] for r in cur.fetchall()]
                logger.info(f"Detected memory manager tables: {tables}")
                if len(tables) < 2:
                    logger.error("Memory manager cache tables were not created!")
                    sys.exit(1)
                    
        # Test cache write & read
        test_hash = "test_hash_12345"
        test_payload = "test_governance_result_data_payload"
        memory.cache_governance_plan(query_hash=test_hash, payload=test_payload, ttl_days=1)
        logger.info("Wrote test governance plan to Supabase PostgreSQL plan_cache.")
        
        cached = memory.retrieve_cached_plan(test_hash)
        logger.info(f"Retrieved cached plan: {cached}")
        if cached != test_payload:
            logger.error(f"Cache validation mismatch! Expected '{test_payload}', got '{cached}'")
            sys.exit(1)
        logger.info("Successfully validated plan_cache read/write.")
        
        # Test plot write & read
        test_plot_id = "test_plot_67890"
        test_plot_data = {"plot_type": "line", "data": [1, 2, 3]}
        memory.store_plot(plot_id=test_plot_id, plot_data=test_plot_data, ttl_days=1)
        logger.info("Wrote test plot spec to Supabase PostgreSQL visualizations.")
        
        cached_plot = memory.retrieve_plot(test_plot_id)
        logger.info(f"Retrieved plot data: {cached_plot}")
        if cached_plot != test_plot_data:
            logger.error(f"Plot cache validation mismatch! Expected '{test_plot_data}', got '{cached_plot}'")
            sys.exit(1)
        logger.info("Successfully validated visualizations plot read/write.")
        
    except Exception as e:
        logger.error(f"Memory manager cache validation failed: {e}")
        sys.exit(1)

    logger.info("ALL SUPABASE INTEGRATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
