import os
import sys
import unittest
from datetime import datetime, timezone


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.memory.mongodb_memory_layer import MongoMemoryManager


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=None):
        self.statements.append((sql, params))

    def fetchall(self):
        return self.rows


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
        self.assertIn("idx_chat_messages_session_created", sql)

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
        self.assertIn("mistral:latest", params[3])

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


if __name__ == "__main__":
    unittest.main()
