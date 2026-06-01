import hashlib
import json
import logging
import os
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

try:
    from psycopg_pool import ConnectionPool
except ImportError:
    ConnectionPool = None


logger = logging.getLogger(__name__)


def get_clean_postgres_url(url: str) -> str:
    if not url:
        return ""
    try:
        if "://" in url:
            scheme, rest = url.split("://", 1)
        else:
            scheme, rest = "postgresql", url
            
        if "@" in rest:
            credentials, host_db = rest.rsplit("@", 1)
            if ":" in credentials:
                user, password = credentials.split(":", 1)
                if password.startswith("[") and password.endswith("]"):
                    clean_password = password[1:-1]
                else:
                    clean_password = password
                
                # Unquote to avoid double encoding if already encoded in env/URL
                decoded_password = urllib.parse.unquote(clean_password)
                encoded_password = urllib.parse.quote(decoded_password)
                return f"{scheme}://{user}:{encoded_password}@{host_db}"
    except Exception:
        pass
    return url


def _test_and_get_pool(postgres_url: str) -> Any:
    if ConnectionPool is None:
        logger.warning("psycopg_pool is not installed; unable to connect to Postgres.")
        return None

    # Try clean URL without brackets
    url_no_brackets = get_clean_postgres_url(postgres_url)
    try:
        pool = ConnectionPool(
            conninfo=url_no_brackets,
            min_size=1,
            max_size=5,
            open=True,
            timeout=5.0,
            kwargs={"autocommit": True}
        )
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
        logger.info("Supabase Postgres connected successfully (brackets stripped).")
        return pool
    except Exception as e:
        logger.warning("Failed to connect with stripped brackets: %s. Trying with encoded brackets...", e)
        
    # Try clean URL preserving brackets
    try:
        if "://" in postgres_url:
            scheme, rest = postgres_url.split("://", 1)
        else:
            scheme, rest = "postgresql", postgres_url
        if "@" in rest:
            credentials, host_db = rest.rsplit("@", 1)
            if ":" in credentials:
                user, password = credentials.split(":", 1)
                decoded_password = urllib.parse.unquote(password)
                encoded_password = urllib.parse.quote(decoded_password)
                url_with_brackets = f"{scheme}://{user}:{encoded_password}@{host_db}"
                
                pool = ConnectionPool(
                    conninfo=url_with_brackets,
                    min_size=1,
                    max_size=5,
                    open=True,
                    timeout=5.0,
                    kwargs={"autocommit": True}
                )
                with pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1;")
                logger.info("Supabase Postgres connected successfully (brackets preserved & encoded).")
                return pool
    except Exception as e2:
        logger.error("Failed to connect to Supabase Postgres with encoded brackets: %s", e2)
    return None


class MongoMemoryManager:
    """Three-tier hybrid memory helper backed by MongoDB and Supabase PostgreSQL."""

    def __init__(
        self,
        mongo_uri: str | None = None,
        db_name: str = "Stock_data",
        client: MongoClient | None = None,
        postgres_url: str | None = None,
    ) -> None:
        self.mongo_uri = (mongo_uri or os.getenv("MONGO_URI") or "").strip()
        self.db_name = db_name
        self._client: MongoClient | None = client
        self._db = None

        if self._client is None and self.mongo_uri:
            try:
                self._client = MongoClient(
                    self.mongo_uri,
                    tls=True,
                    tlsAllowInvalidCertificates=True,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                    socketTimeoutMS=10000,
                    appname="agentic-ai-portfolio-governance-memory",
                )
                self._client.admin.command("ping")
            except PyMongoError as exc:
                logger.warning("MongoMemoryManager initialization failed: %s", exc)
                self._client = None

        if self._client is not None:
            self._db = self._client[self.db_name]

        # Postgres/Supabase initialization
        self.postgres_url = (postgres_url or os.getenv("SUPABASE_POSTGRES_URL") or "").strip()
        self.pg_pool = None
        if self.postgres_url:
            self.pg_pool = _test_and_get_pool(self.postgres_url)
            if self.pg_pool:
                self.setup_postgres_tables()

    @property
    def is_available(self) -> bool:
        return self._db is not None

    def _collection(self, name: str) -> Collection | None:
        if self._db is None:
            return None
        return self._db[name]

    def setup_postgres_tables(self) -> None:
        if not self.pg_pool:
            return
        try:
            with self.pg_pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS plan_cache (
                            query_hash VARCHAR(64) PRIMARY KEY,
                            payload TEXT NOT NULL,
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            expires_at TIMESTAMP WITH TIME ZONE NOT NULL
                        );
                    """)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS visualizations (
                            plot_id VARCHAR(128) PRIMARY KEY,
                            data JSONB NOT NULL,
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            expires_at TIMESTAMP WITH TIME ZONE NOT NULL
                        );
                    """)
            logger.info("Postgres cache tables checked/created.")
        except Exception as exc:
            logger.error("Failed to set up postgres tables: %s", exc)

    def setup_indexes(self) -> None:
        """L2 TTL and L3 query indexes."""
        if self.pg_pool:
            self.setup_postgres_tables()

        if not self.is_available:
            return

        try:
            plan_cache = self._collection("plan_cache")
            regime_patterns = self._collection("regime_patterns")
            visualizations = self._collection("visualizations")
            if plan_cache is None or regime_patterns is None:
                return

            plan_cache.create_index([("query_hash", ASCENDING)], unique=True, background=True)
            plan_cache.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0, background=True)

            if visualizations is not None:
                visualizations.create_index([("plot_id", ASCENDING)], unique=True, background=True)
                visualizations.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0, background=True)

            regime_patterns.create_index([("regime_type", ASCENDING), ("created_at", DESCENDING)], background=True)
            regime_patterns.create_index([("target_date", DESCENDING)], background=True)
            regime_patterns.create_index([("instability_index", DESCENDING)], background=True)
        except PyMongoError as exc:
            logger.warning("Failed to setup memory indexes: %s", exc)

    def compute_query_hash(
        self,
        tickers: list[str],
        target_date: str,
        risk_tolerance: str | None = None,
    ) -> str:
        normalized_tickers = sorted(
            {str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()}
        )
        payload = {
            "tickers": normalized_tickers,
            "target_date": str(target_date).strip(),
            "risk_tolerance": str(risk_tolerance or "moderate").strip().lower(),
        }
        payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    def cache_governance_plan(
        self,
        query_hash: str,
        payload: str,
        ttl_days: int = 7,
    ) -> None:
        if self.pg_pool:
            try:
                now = datetime.now(timezone.utc)
                expires_at = now + timedelta(days=ttl_days)
                with self.pg_pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO plan_cache (query_hash, payload, updated_at, expires_at)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (query_hash)
                            DO UPDATE SET payload = EXCLUDED.payload, updated_at = EXCLUDED.updated_at, expires_at = EXCLUDED.expires_at;
                            """,
                            (query_hash, payload, now, expires_at),
                        )
                return
            except Exception as exc:
                logger.warning("Postgres cache_governance_plan failed: %s. Falling back to Mongo...", exc)

        if not self.is_available:
            return

        try:
            plan_cache = self._collection("plan_cache")
            if plan_cache is None:
                return

            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(days=ttl_days)
            plan_cache.update_one(
                {"query_hash": query_hash},
                {
                    "$set": {
                        "query_hash": query_hash,
                        "payload": payload,
                        "updated_at": now,
                        "expires_at": expires_at,
                    },
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
        except PyMongoError as exc:
            logger.warning("Failed to cache governance plan for hash %s: %s", query_hash, exc)

    def retrieve_cached_plan(self, query_hash: str) -> str | None:
        if self.pg_pool:
            try:
                now = datetime.now(timezone.utc)
                with self.pg_pool.connection() as conn:
                    with conn.cursor() as cur:
                        # Clean up expired items
                        cur.execute("DELETE FROM plan_cache WHERE expires_at < %s;", (now,))
                        # Retrieve cache
                        cur.execute(
                            "SELECT payload FROM plan_cache WHERE query_hash = %s AND expires_at >= %s;",
                            (query_hash, now),
                        )
                        row = cur.fetchone()
                        if row:
                            return row[0]
                return None
            except Exception as exc:
                logger.warning("Postgres retrieve_cached_plan failed: %s. Falling back to Mongo...", exc)

        if not self.is_available:
            return None

        try:
            plan_cache = self._collection("plan_cache")
            if plan_cache is None:
                return None
            doc = plan_cache.find_one({"query_hash": query_hash}, {"payload": 1})
            if not doc:
                return None
            payload = doc.get("payload")
            return payload if isinstance(payload, str) else None
        except PyMongoError as exc:
            logger.warning("Failed to retrieve cache for hash %s: %s", query_hash, exc)
            return None

    def store_regime_pattern(
        self,
        target_date: str,
        regime_type: str,
        instability_index: float,
        lambda_t: float,
        weights: dict[str, float],
    ) -> None:
        if not self.is_available:
            return

        try:
            regime_patterns = self._collection("regime_patterns")
            if regime_patterns is None:
                return

            now = datetime.now(timezone.utc)
            regime_patterns.insert_one(
                {
                    "target_date": str(target_date),
                    "regime_type": str(regime_type),
                    "instability_index": float(instability_index),
                    "lambda_t": float(lambda_t),
                    "weights": {str(k): float(v) for k, v in (weights or {}).items()},
                    "created_at": now,
                }
            )
        except (PyMongoError, ValueError, TypeError) as exc:
            logger.warning("Failed to store regime pattern: %s", exc)

    def retrieve_similar_regimes(
        self,
        regime_type: str,
        instability_index: float,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if not self.is_available:
            return []

        try:
            regime_patterns = self._collection("regime_patterns")
            if regime_patterns is None:
                return []

            window = 0.15
            query = {
                "regime_type": str(regime_type),
                "instability_index": {
                    "$gte": float(instability_index) - window,
                    "$lte": float(instability_index) + window,
                },
            }

            rows = list(
                regime_patterns.find(query)
                .sort("created_at", DESCENDING)
                .limit(max(1, int(limit)))
            )

            for row in rows:
                row_id = row.get("_id")
                if isinstance(row_id, ObjectId):
                    row["_id"] = str(row_id)

            return rows
        except (PyMongoError, ValueError, TypeError) as exc:
            logger.warning("Failed to retrieve similar regimes: %s", exc)
            return []

    def store_plot(self, plot_id: str, plot_data: dict, ttl_days: int = 1) -> None:
        """Store a visualization payload in Supabase Postgres or MongoDB."""
        if self.pg_pool:
            try:
                now = datetime.now(timezone.utc)
                expires_at = now + timedelta(days=ttl_days)
                serialized_data = json.dumps(plot_data)
                with self.pg_pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO visualizations (plot_id, data, updated_at, expires_at)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (plot_id)
                            DO UPDATE SET data = EXCLUDED.data, updated_at = EXCLUDED.updated_at, expires_at = EXCLUDED.expires_at;
                            """,
                            (plot_id, serialized_data, now, expires_at),
                        )
                return
            except Exception as exc:
                logger.warning("Postgres store_plot failed: %s. Falling back to Mongo...", exc)

        if not self.is_available:
            return
            
        try:
            plots_col = self._collection("visualizations")
            if plots_col is None:
                return
                
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(days=ttl_days)
            plots_col.update_one(
                {"plot_id": plot_id},
                {
                    "$set": {
                        "plot_id": plot_id,
                        "data": plot_data,
                        "updated_at": now,
                        "expires_at": expires_at,
                    },
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
        except PyMongoError as exc:
            logger.warning("Failed to store plot %s: %s", plot_id, exc)

    def retrieve_plot(self, plot_id: str) -> dict | None:
        """Retrieve a visualization payload from Supabase Postgres or MongoDB."""
        if self.pg_pool:
            try:
                now = datetime.now(timezone.utc)
                with self.pg_pool.connection() as conn:
                    with conn.cursor() as cur:
                        # Clean up expired visualizations
                        cur.execute("DELETE FROM visualizations WHERE expires_at < %s;", (now,))
                        # Retrieve plot
                        cur.execute(
                            "SELECT data FROM visualizations WHERE plot_id = %s AND expires_at >= %s;",
                            (plot_id, now),
                        )
                        row = cur.fetchone()
                        if row:
                            data_val = row[0]
                            if isinstance(data_val, str):
                                return json.loads(data_val)
                            return data_val
                return None
            except Exception as exc:
                logger.warning("Postgres retrieve_plot failed: %s. Falling back to Mongo...", exc)

        if not self.is_available:
            return None
            
        try:
            plots_col = self._collection("visualizations")
            if plots_col is None:
                return None
            doc = plots_col.find_one({"plot_id": plot_id}, {"data": 1, "_id": 0})
            if not doc:
                return None
            return doc.get("data")
        except PyMongoError as exc:
            logger.warning("Failed to retrieve plot %s: %s", plot_id, exc)
            return None
