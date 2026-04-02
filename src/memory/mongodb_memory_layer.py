import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError


logger = logging.getLogger(__name__)


class MongoMemoryManager:
    """Three-tier memory helper backed by MongoDB collections."""

    def __init__(
        self,
        mongo_uri: str | None = None,
        db_name: str = "Stock_data",
        client: MongoClient | None = None,
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

    @property
    def is_available(self) -> bool:
        return self._db is not None

    def _collection(self, name: str) -> Collection | None:
        if self._db is None:
            return None
        return self._db[name]

    def setup_indexes(self) -> None:
        """L2 TTL and L3 query indexes."""
        if not self.is_available:
            return

        try:
            plan_cache = self._collection("plan_cache")
            regime_patterns = self._collection("regime_patterns")
            if plan_cache is None or regime_patterns is None:
                return

            plan_cache.create_index([("query_hash", ASCENDING)], unique=True, background=True)
            plan_cache.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0, background=True)

            regime_patterns.create_index([("regime_type", ASCENDING), ("created_at", DESCENDING)], background=True)
            regime_patterns.create_index([("target_date", DESCENDING)], background=True)
            regime_patterns.create_index([("instability_index", DESCENDING)], background=True)
        except PyMongoError as exc:
            logger.warning("Failed to setup memory indexes: %s", exc)

    def compute_query_hash(self, tickers: list[str], target_date: str) -> str:
        normalized_tickers = sorted(
            {str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()}
        )
        payload = {"tickers": normalized_tickers, "target_date": str(target_date).strip()}
        payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    def cache_governance_plan(
        self,
        query_hash: str,
        payload: str,
        ttl_days: int = 7,
    ) -> None:
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
