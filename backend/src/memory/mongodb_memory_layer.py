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


_POOL_CACHE = {}


def _test_and_get_pool(postgres_url: str) -> Any:
    global _POOL_CACHE
    if ConnectionPool is None:
        logger.warning("psycopg_pool is not installed; unable to connect to Postgres.")
        return None

    url_key = postgres_url.strip()
    if url_key in _POOL_CACHE:
        logger.info("Reusing cached Supabase Postgres connection pool.")
        return _POOL_CACHE[url_key]

    # Try clean URL without brackets
    url_no_brackets = get_clean_postgres_url(postgres_url)
    try:
        pool = ConnectionPool(
            conninfo=url_no_brackets,
            min_size=0,
            max_size=2,
            open=True,
            timeout=5.0,
            kwargs={"autocommit": True, "prepare_threshold": None}
        )
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
        logger.info("Supabase Postgres connected successfully (brackets stripped).")
        _POOL_CACHE[url_key] = pool
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
                    min_size=0,
                    max_size=2,
                    open=True,
                    timeout=5.0,
                    kwargs={"autocommit": True, "prepare_threshold": None}
                )
                with pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1;")
                logger.info("Supabase Postgres connected successfully (brackets preserved & encoded).")
                _POOL_CACHE[url_key] = pool
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
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS chat_messages (
                            id BIGSERIAL PRIMARY KEY,
                            session_id VARCHAR(128) NOT NULL,
                            role VARCHAR(32) NOT NULL,
                            content TEXT NOT NULL,
                            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
                        ON chat_messages (session_id, created_at, id);
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

    def store_plot(self, plot_id: str, plot_data: dict, ttl_days: int = 90) -> bool:
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
                return True
            except Exception as exc:
                logger.warning("Postgres store_plot failed: %s. Falling back to Mongo...", exc)

        if not self.is_available:
            return False
            
        try:
            plots_col = self._collection("visualizations")
            if plots_col is None:
                return False
                
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
            return True
        except PyMongoError as exc:
            logger.warning("Failed to store plot %s: %s", plot_id, exc)
            return False

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

    def append_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist a chat message for UI history hydration."""
        clean_session_id = str(session_id or "").strip()
        clean_role = str(role or "").strip().lower()
        clean_content = str(content or "")
        if not clean_session_id or clean_role not in {"user", "assistant", "system"} or not clean_content.strip():
            return

        metadata_payload = json.dumps(metadata or {})

        if self.pg_pool:
            try:
                with self.pg_pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO chat_messages (session_id, role, content, metadata)
                            VALUES (%s, %s, %s, %s);
                            """,
                            (clean_session_id, clean_role, clean_content, metadata_payload),
                        )
                return
            except Exception as exc:
                logger.warning("Postgres append_chat_message failed: %s. Falling back to Mongo...", exc)

        if not self.is_available:
            return

        try:
            chat_col = self._collection("chat_messages")
            if chat_col is None:
                return
            chat_col.insert_one(
                {
                    "session_id": clean_session_id,
                    "role": clean_role,
                    "content": clean_content,
                    "metadata": metadata or {},
                    "created_at": datetime.now(timezone.utc),
                }
            )
        except PyMongoError as exc:
            logger.warning("Failed to append chat message for session %s: %s", clean_session_id, exc)

    def _format_chat_session_title(self, title: Any) -> str:
        normalized = " ".join(str(title or "").split())
        if not normalized:
            return "New chat"
        if len(normalized) <= 64:
            return normalized
        return f"{normalized[:61].rstrip()}..."

    def list_chat_messages(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """Return persisted chat messages in chronological order."""
        clean_session_id = str(session_id or "").strip()
        if not clean_session_id:
            return []

        safe_limit = max(1, min(int(limit or 200), 500))

        if self.pg_pool:
            try:
                with self.pg_pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT id, role, content, metadata, created_at
                            FROM chat_messages
                            WHERE session_id = %s
                            ORDER BY created_at ASC, id ASC
                            LIMIT %s;
                            """,
                            (clean_session_id, safe_limit),
                        )
                        rows = cur.fetchall()

                messages = []
                for row in rows:
                    metadata = row[3]
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {}
                    created_at = row[4]
                    messages.append(
                        {
                            "id": str(row[0]),
                            "role": row[1],
                            "content": row[2],
                            "metadata": metadata or {},
                            "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
                        }
                    )
                return messages
            except Exception as exc:
                logger.warning("Postgres list_chat_messages failed: %s. Falling back to Mongo...", exc)

        if not self.is_available:
            return []

        try:
            chat_col = self._collection("chat_messages")
            if chat_col is None:
                return []
            cursor = chat_col.find({"session_id": clean_session_id}).sort("created_at", ASCENDING).limit(safe_limit)
            messages = []
            for row in cursor:
                created_at = row.get("created_at")
                messages.append(
                    {
                        "id": str(row.get("_id")),
                        "role": row.get("role", ""),
                        "content": row.get("content", ""),
                        "metadata": row.get("metadata") or {},
                        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or ""),
                    }
                )
            return messages
        except PyMongoError as exc:
            logger.warning("Failed to list chat messages for session %s: %s", clean_session_id, exc)
            return []

    def list_chat_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return persisted chat sessions sorted by most recent activity."""
        safe_limit = max(1, min(int(limit or 50), 100))

        if self.pg_pool:
            try:
                with self.pg_pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            WITH ranked_messages AS (
                                SELECT
                                    session_id,
                                    role,
                                    content,
                                    created_at,
                                    ROW_NUMBER() OVER (
                                        PARTITION BY session_id
                                        ORDER BY CASE WHEN role = 'user' THEN 0 ELSE 1 END, created_at ASC, id ASC
                                    ) AS title_rank
                                FROM chat_messages
                            )
                            SELECT
                                session_id,
                                MAX(CASE WHEN title_rank = 1 AND role = 'user' THEN content ELSE NULL END) AS title,
                                COUNT(*) AS message_count,
                                MIN(created_at) AS created_at,
                                MAX(created_at) AS updated_at
                            FROM ranked_messages
                            GROUP BY session_id
                            ORDER BY updated_at DESC
                            LIMIT %s;
                            """,
                            (safe_limit,),
                        )
                        rows = cur.fetchall()

                return [
                    {
                        "session_id": str(row[0]),
                        "title": self._format_chat_session_title(row[1]),
                        "message_count": int(row[2] or 0),
                        "created_at": row[3].isoformat() if hasattr(row[3], "isoformat") else str(row[3] or ""),
                        "updated_at": row[4].isoformat() if hasattr(row[4], "isoformat") else str(row[4] or ""),
                    }
                    for row in rows
                ]
            except Exception as exc:
                logger.warning("Postgres list_chat_sessions failed: %s. Falling back to Mongo...", exc)

        if not self.is_available:
            return []

        try:
            chat_col = self._collection("chat_messages")
            if chat_col is None:
                return []
            pipeline = [
                {"$sort": {"session_id": 1, "created_at": 1, "_id": 1}},
                {
                    "$group": {
                        "_id": "$session_id",
                        "messages": {"$push": {"role": "$role", "content": "$content"}},
                        "message_count": {"$sum": 1},
                        "created_at": {"$min": "$created_at"},
                        "updated_at": {"$max": "$created_at"},
                    }
                },
                {
                    "$project": {
                        "first_user": {
                            "$first": {
                                "$map": {
                                    "input": {
                                        "$filter": {
                                            "input": "$messages",
                                            "as": "message",
                                            "cond": {"$eq": ["$$message.role", "user"]},
                                        }
                                    },
                                    "as": "user_message",
                                    "in": "$$user_message.content",
                                }
                            }
                        },
                        "message_count": 1,
                        "created_at": 1,
                        "updated_at": 1,
                    }
                },
                {"$sort": {"updated_at": -1}},
                {"$limit": safe_limit},
            ]
            sessions = []
            for row in chat_col.aggregate(pipeline):
                created_at = row.get("created_at")
                updated_at = row.get("updated_at")
                sessions.append(
                    {
                        "session_id": str(row.get("_id")),
                        "title": self._format_chat_session_title(row.get("first_user")),
                        "message_count": int(row.get("message_count") or 0),
                        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or ""),
                        "updated_at": updated_at.isoformat() if hasattr(updated_at, "isoformat") else str(updated_at or ""),
                    }
                )
            return sessions
        except PyMongoError as exc:
            logger.warning("Failed to list chat sessions: %s", exc)
            return []
