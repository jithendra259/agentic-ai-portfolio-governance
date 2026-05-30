import os
import pymongo
from typing import Dict, Any, List

class BlackboardMemoryStore:
    """
    MongoDB abstraction for blackboard_mpi and related pipeline collections.
    Provides persistence and state propagation across rolling windows.
    """
    def __init__(self, mongo_uri: str, db_name: str):
        self.client = pymongo.MongoClient(
            mongo_uri,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=10000
        )
        self.db = self.client[db_name]
        self.bb = self.db["blackboard_mpi"]

    def store_window(self, universe_id: str, window_id: str, window_number: int, data: Dict[str, Any]):
        """
        Stores or updates an entire window payload into blackboard_mpi.
        """
        self.bb.update_one(
            {"universe_id": universe_id, "window_id": window_id},
            {"$set": {"window_number": window_number, **data}},
            upsert=True
        )

    def get_window(self, universe_id: str, window_id: str) -> Dict[str, Any]:
        """
        Retrieves a window document.
        """
        return self.bb.find_one({"universe_id": universe_id, "window_id": window_id}) or {}

    def get_all_windows(self, universe_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all windows for a universe sequentially.
        """
        return list(self.bb.find({"universe_id": universe_id}).sort("window_number", pymongo.ASCENDING))

    def get_active_universes(self) -> List[str]:
        """
        Queries all distinct universes currently present in the blackboard.
        """
        return list(self.bb.distinct("universe_id"))

    def clear_universe(self, universe_id: str):
        """
        Clears all window data for a universe in the blackboard.
        """
        self.bb.delete_many({"universe_id": universe_id})
