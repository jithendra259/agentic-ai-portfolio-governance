from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from src.memory.state_sanitizer import sanitize_for_mongodb


class ArtifactStore:
    """Side-car data plane for large portfolio artifacts kept out of LangGraph state."""

    def __init__(
        self,
        mongo_uri: str | None = None,
        db_name: str = "Stock_data",
        collection_name: str = "portfolio_artifacts",
        client: MongoClient | None = None,
    ) -> None:
        self.mongo_uri = (mongo_uri or os.getenv("MONGO_URI") or "").strip()
        self.db_name = db_name
        self.collection_name = collection_name
        self._client = client
        self._fallback: dict[str, dict[str, Any]] = {}

        if self._client is None and self.mongo_uri:
            try:
                self._client = MongoClient(
                    self.mongo_uri,
                    tls=True,
                    tlsAllowInvalidCertificates=True,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                    socketTimeoutMS=10000,
                    appname="agentic-ai-portfolio-governance-artifacts",
                )
                self._client.admin.command("ping")
            except PyMongoError:
                self._client = None

        self._collection = self._client[self.db_name][self.collection_name] if self._client is not None else None
        if self._collection is not None:
            try:
                self._collection.create_index("expires_at", expireAfterSeconds=0)
                self._collection.create_index("kind")
            except PyMongoError:
                pass

    def save(self, data: Any, *, kind: str, ttl_days: int = 7, metadata: dict[str, Any] | None = None) -> str:
        artifact_id = f"artifact-{uuid4().hex}"
        payload = self._serialize_payload(data)
        doc = {
            "_id": artifact_id,
            "kind": kind,
            "payload": payload,
            "metadata": sanitize_for_mongodb(metadata or {}),
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + timedelta(days=ttl_days),
        }
        if self._collection is not None:
            try:
                self._collection.insert_one(doc)
                return artifact_id
            except PyMongoError:
                pass
        self._fallback[artifact_id] = doc
        return artifact_id

    def load(self, artifact_id: str) -> Any:
        doc = None
        if self._collection is not None:
            try:
                doc = self._collection.find_one({"_id": artifact_id})
            except PyMongoError:
                doc = None
        if doc is None:
            doc = self._fallback.get(artifact_id)
        if not doc:
            raise KeyError(f"Artifact not found: {artifact_id}")
        return self._deserialize_payload(doc.get("payload", {}))

    def _serialize_payload(self, data: Any) -> dict[str, Any]:
        if isinstance(data, pd.DataFrame):
            return {
                "type": "dataframe",
                "data": sanitize_for_mongodb(data.to_dict(orient="split")),
            }
        if isinstance(data, pd.Series):
            return {
                "type": "series",
                "data": sanitize_for_mongodb(data.to_dict()),
                "name": data.name,
            }
        return {"type": "json", "data": sanitize_for_mongodb(data)}

    def _deserialize_payload(self, payload: dict[str, Any]) -> Any:
        payload_type = payload.get("type")
        data = payload.get("data")
        if payload_type == "dataframe":
            return pd.DataFrame(**data)
        if payload_type == "series":
            return pd.Series(data, name=payload.get("name"))
        return data
