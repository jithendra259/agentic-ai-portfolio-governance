import os
import sys
import unittest
from datetime import datetime, timezone


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.memory.mongodb_memory_layer import MongoMemoryManager


class FakeCursor:
    def __init__(self, rows=None, row=None):
        self.rows = rows or []
        self.row = row
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=None):
        self.statements.append((sql, params))

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.row


class FakeConnection:
    def __init__(self, cursor):
        self.cursor_obj = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self):
        return self.cursor_obj


class FakePool:
    def __init__(self, cursor):
        self.cursor = cursor

    def connection(self):
        return FakeConnection(self.cursor)


class SupabaseChatMemoryTests(unittest.TestCase):
    def test_setup_postgres_tables_creates_chat_messages_table_and_index(self):
        cursor = FakeCursor()
        manager = MongoMemoryManager(mongo_uri="", postgres_url="")
        manager.pg_pool = FakePool(cursor)

        manager.setup_postgres_tables()

        sql = "\n".join(statement for statement, _params in cursor.statements)
        self.assertIn("CREATE TABLE IF NOT EXISTS chat_messages", sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS conversation_state", sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS market_data_cache", sql)
        self.assertIn("idx_chat_messages_session_created", sql)
        self.assertIn("idx_chat_messages_session_recent", sql)
        self.assertIn("idx_chat_messages_session_role_created", sql)
        self.assertIn("idx_conversation_state_user_updated", sql)
        self.assertIn("idx_market_data_cache_lookup", sql)

    def test_append_chat_message_writes_user_message_to_supabase(self):
        cursor = FakeCursor()
        manager = MongoMemoryManager(mongo_uri="", postgres_url="")
        manager.pg_pool = FakePool(cursor)

        manager.append_chat_message(
            "session-1",
            "user",
            "Plot AAPL",
            metadata={"model": "mistral:latest"},
        )

        sql, params = cursor.statements[-1]
        self.assertIn("INSERT INTO chat_messages", sql)
        self.assertEqual(params[0], "session-1")
        self.assertEqual(params[1], "user")
        self.assertEqual(params[2], "Plot AAPL")
        self.assertIsNone(params[3])
        self.assertIn("mistral:latest", params[4])

    def test_list_chat_messages_returns_chronological_ui_payload(self):
        created_at = datetime(2026, 6, 2, tzinfo=timezone.utc)
        cursor = FakeCursor(rows=[(42, "assistant", "Done", {"plot_ids": []}, created_at)])
        manager = MongoMemoryManager(mongo_uri="", postgres_url="")
        manager.pg_pool = FakePool(cursor)

        rows = manager.list_chat_messages("session-1")

        self.assertEqual(
            rows,
            [
                {
                    "id": "42",
                    "role": "assistant",
                    "content": "Done",
                    "metadata": {"plot_ids": []},
                    "created_at": "2026-06-02T00:00:00+00:00",
                }
            ],
        )
        sql, params = cursor.statements[-1]
        self.assertIn("ORDER BY created_at DESC, id DESC", sql)
        self.assertIn("ORDER BY created_at ASC, id ASC", sql)
        self.assertEqual(params, ("session-1", 200))

    def test_list_chat_sessions_returns_recent_session_summaries(self):
        first_created = datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)
        second_created = datetime(2026, 6, 2, 10, 30, tzinfo=timezone.utc)
        cursor = FakeCursor(
            rows=[
                ("session-2", "Compare AAPL and MSFT performance over a long horizon", 4, second_created, second_created),
                ("session-1", "Plot TSLA", 2, first_created, first_created),
            ]
        )
        manager = MongoMemoryManager(mongo_uri="", postgres_url="")
        manager.pg_pool = FakePool(cursor)

        sessions = manager.list_chat_sessions(limit=20)

        sql, params = cursor.statements[-1]
        self.assertIn("GROUP BY session_id", sql)
        self.assertEqual(params[0], 20)
        self.assertEqual(
            sessions,
            [
                {
                    "session_id": "session-2",
                    "title": "Compare AAPL and MSFT performance over a long horizon",
                    "message_count": 4,
                    "created_at": "2026-06-02T10:30:00+00:00",
                    "updated_at": "2026-06-02T10:30:00+00:00",
                },
                {
                    "session_id": "session-1",
                    "title": "Plot TSLA",
                    "message_count": 2,
                    "created_at": "2026-06-01T09:00:00+00:00",
                    "updated_at": "2026-06-01T09:00:00+00:00",
                },
            ],
        )

    def test_list_chat_sessions_uses_new_chat_for_missing_user_title(self):
        created_at = datetime(2026, 6, 2, tzinfo=timezone.utc)
        cursor = FakeCursor(rows=[("session-empty", None, 1, created_at, created_at)])
        manager = MongoMemoryManager(mongo_uri="", postgres_url="")
        manager.pg_pool = FakePool(cursor)

        sessions = manager.list_chat_sessions()

        self.assertEqual(sessions[0]["title"], "New chat")

    def test_store_market_data_cache_writes_supabase_row(self):
        cursor = FakeCursor()
        manager = MongoMemoryManager(mongo_uri="", postgres_url="")
        manager.pg_pool = FakePool(cursor)

        stored = manager.store_market_data_cache(
            cache_key="cache-1",
            symbol="aapl",
            data_type="history",
            period="1y",
            interval="1d",
            start_date="2025-01-01",
            end_date="2025-12-31",
            payload={"symbol": "AAPL", "history": [{"Date": "2025-01-01", "Close": 100.0}]},
        )

        self.assertTrue(stored)
        sql, params = cursor.statements[-1]
        self.assertIn("INSERT INTO market_data_cache", sql)
        self.assertEqual(params[0], "cache-1")
        self.assertEqual(params[1], "AAPL")
        self.assertEqual(params[2], "history")
        self.assertIn('"symbol": "AAPL"', params[7])

    def test_retrieve_market_data_cache_returns_payload(self):
        created_at = datetime(2026, 6, 2, tzinfo=timezone.utc)
        expires_at = datetime(2026, 6, 3, tzinfo=timezone.utc)
        cursor = FakeCursor(row=({"symbol": "AAPL", "history": []}, "yfinance", created_at, expires_at))
        manager = MongoMemoryManager(mongo_uri="", postgres_url="")
        manager.pg_pool = FakePool(cursor)

        cached = manager.retrieve_market_data_cache("cache-1")

        self.assertEqual(cached["payload"]["symbol"], "AAPL")
        self.assertEqual(cached["source"], "yfinance")
        sql, params = cursor.statements[-1]
        self.assertIn("FROM market_data_cache", sql)
        self.assertEqual(params[0], "cache-1")

    def test_store_conversation_state_writes_compact_memory_to_supabase(self):
        cursor = FakeCursor()
        manager = MongoMemoryManager(mongo_uri="", postgres_url="")
        manager.pg_pool = FakePool(cursor)

        stored = manager.store_conversation_state(
            "session-1",
            {"conversation_state": {"current_topic": "Adaptive G-CVaR comparison"}},
            user_id="user-1",
        )

        self.assertTrue(stored)
        sql, params = cursor.statements[-1]
        self.assertIn("INSERT INTO conversation_state", sql)
        self.assertEqual(params[0], "session-1")
        self.assertEqual(params[1], "user-1")
        self.assertIn("Adaptive G-CVaR comparison", params[2])

    def test_retrieve_conversation_state_returns_state_payload(self):
        cursor = FakeCursor(row=({"conversation_state": {"current_strategy": "Adaptive G-CVaR"}},))
        manager = MongoMemoryManager(mongo_uri="", postgres_url="")
        manager.pg_pool = FakePool(cursor)

        state = manager.retrieve_conversation_state("session-1", user_id="user-1")

        self.assertEqual(state["conversation_state"]["current_strategy"], "Adaptive G-CVaR")
        sql, params = cursor.statements[-1]
        self.assertIn("FROM conversation_state", sql)
        self.assertEqual(params, ("session-1", "user-1"))


if __name__ == "__main__":
    unittest.main()
