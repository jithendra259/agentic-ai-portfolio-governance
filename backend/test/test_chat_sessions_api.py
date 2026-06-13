import os
import sys
import types
import unittest
import json

from fastapi.testclient import TestClient


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


fake_orchestrator = types.ModuleType("src.orchestrator.chatbot_orchestrator")
fake_orchestrator.CONFIGURED_DEFAULT_LLM_MODEL = "default-model"
fake_orchestrator.FALLBACK_OLLAMA_MODEL = "fallback-model"
fake_orchestrator.INSTALLED_OLLAMA_MODELS = ["primary-model"]
fake_orchestrator.PRIMARY_OLLAMA_MODEL = "primary-model"
fake_orchestrator.memory_manager = None
fake_orchestrator.portfolio_assistant = None
fake_orchestrator.streaming_portfolio_assistant = None
sys.modules["src.orchestrator.chatbot_orchestrator"] = fake_orchestrator

from api.main import app
from src.utils.crypto_utils import create_auth_token


class FakeMemoryManager:
    pg_pool = object()
    
    def __init__(self):
        self.deleted_session_id = None
        self.deleted_user_id = None
        self.last_list_user_id = None
        self.last_legacy_session_ids = None
        self.last_messages_user_id = None
        self.last_include_legacy_unowned = None
        self.last_claim_user_id = None
        self.last_claim_session_ids = None
        self.last_claim_all = None
        self.append_calls = []

    def list_chat_sessions(self, limit=50, user_id=None, legacy_session_ids=None):
        self.last_list_user_id = user_id
        self.last_legacy_session_ids = legacy_session_ids
        return [
            {
                "session_id": "session-1",
                "title": "Plot AAPL",
                "message_count": 2,
                "created_at": "2026-06-02T00:00:00+00:00",
                "updated_at": "2026-06-02T00:05:00+00:00",
            }
        ]

    def delete_chat_session(self, session_id, user_id=None):
        self.deleted_session_id = session_id
        self.deleted_user_id = user_id
        return 2

    def append_chat_message(self, session_id, role, content, metadata=None, user_id=None):
        self.append_calls.append(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "user_id": user_id,
            }
        )
        return None

    def list_chat_messages(self, session_id, limit=200, user_id=None, include_legacy_unowned=False):
        self.last_messages_user_id = user_id
        self.last_include_legacy_unowned = include_legacy_unowned
        return []

    def claim_legacy_chat_sessions(self, user_id, session_ids=None, claim_all=False, limit=100):
        self.last_claim_user_id = user_id
        self.last_claim_session_ids = session_ids
        self.last_claim_all = claim_all
        return {"claimed_rows": 4, "claimed_sessions": ["session-legacy", "session-two"]}


class ChatSessionsApiTests(unittest.TestCase):
    def test_chat_sessions_endpoint_returns_summaries(self):
        import api.main as main

        fake_memory = FakeMemoryManager()
        original_memory_manager = main.memory_manager
        main.memory_manager = fake_memory
        try:
            client = TestClient(app)
            response = client.get("/chat/sessions?limit=25")
        finally:
            main.memory_manager = original_memory_manager

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "sessions": [
                    {
                        "session_id": "session-1",
                        "title": "Plot AAPL",
                        "message_count": 2,
                        "created_at": "2026-06-02T00:00:00+00:00",
                        "updated_at": "2026-06-02T00:05:00+00:00",
                    }
                ]
            },
        )
        self.assertIsNone(fake_memory.last_list_user_id)
        self.assertEqual(fake_memory.last_legacy_session_ids, [])

    def test_chat_sessions_endpoint_passes_authenticated_user_and_legacy_ids(self):
        import api.main as main

        fake_memory = FakeMemoryManager()
        original_memory_manager = main.memory_manager
        main.memory_manager = fake_memory
        token = create_auth_token({"user": {"id": "user-1", "email": "user@example.com"}})
        try:
            client = TestClient(app)
            response = client.get(
                "/chat/sessions?limit=25&legacy_session_ids=session-legacy,current-session",
                headers={"Authorization": f"Bearer {token}"},
            )
        finally:
            main.memory_manager = original_memory_manager

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_memory.last_list_user_id, "user-1")
        self.assertEqual(fake_memory.last_legacy_session_ids, ["session-legacy", "current-session"])

    def test_chat_messages_endpoint_can_include_browser_known_legacy_rows(self):
        import api.main as main

        fake_memory = FakeMemoryManager()
        original_memory_manager = main.memory_manager
        main.memory_manager = fake_memory
        token = create_auth_token({"user": {"id": "user-1", "email": "user@example.com"}})
        try:
            client = TestClient(app)
            response = client.get(
                "/chat/session-legacy/messages?include_legacy=true",
                headers={"Authorization": f"Bearer {token}"},
            )
        finally:
            main.memory_manager = original_memory_manager

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_memory.last_messages_user_id, "user-1")
        self.assertTrue(fake_memory.last_include_legacy_unowned)

    def test_claim_legacy_chat_sessions_requires_auth_and_passes_user(self):
        import api.main as main

        fake_memory = FakeMemoryManager()
        original_memory_manager = main.memory_manager
        main.memory_manager = fake_memory
        token = create_auth_token({"user": {"id": "user-1", "email": "user@example.com"}})
        try:
            client = TestClient(app)
            unauthenticated = client.post(
                "/chat/sessions/claim-legacy",
                json={"claim_all": True, "session_ids": ["session-legacy"]},
            )
            response = client.post(
                "/chat/sessions/claim-legacy",
                headers={"Authorization": f"Bearer {token}"},
                json={"claim_all": True, "session_ids": ["session-legacy"]},
            )
        finally:
            main.memory_manager = original_memory_manager

        self.assertEqual(unauthenticated.status_code, 401)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"claimed_rows": 4, "claimed_sessions": ["session-legacy", "session-two"]})
        self.assertEqual(fake_memory.last_claim_user_id, "user-1")
        self.assertEqual(fake_memory.last_claim_session_ids, ["session-legacy"])
        self.assertFalse(fake_memory.last_claim_all)

    def test_delete_chat_session_endpoint_deletes_history(self):
        import api.main as main

        fake_memory = FakeMemoryManager()
        original_memory_manager = main.memory_manager
        main.memory_manager = fake_memory
        main.session_memory_store.save_state("session-1", {"session_id": "session-1", "universe": {"tickers": ["AAPL"]}})
        main.session_memory_store.append_message("session-1", "user", "hello")
        try:
            client = TestClient(app)
            response = client.delete("/chat/session-1")
        finally:
            main.memory_manager = original_memory_manager

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"session_id": "session-1", "deleted_count": 2})
        self.assertEqual(fake_memory.deleted_session_id, "session-1")
        self.assertIsNone(fake_memory.deleted_user_id)
        self.assertEqual(main.session_memory_store.get_last_messages("session-1"), [])

    def test_chat_stream_chunks_fast_path_responses(self):
        import api.main as main

        token = create_auth_token({"user": {"id": "user-1", "email": "user@example.com"}})
        fake_memory = FakeMemoryManager()
        original_memory_manager = main.memory_manager
        main.memory_manager = fake_memory
        try:
            client = TestClient(app)
            with client.stream(
                "POST",
                "/chat/stream",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "session_id": "stream-session-1",
                    "user_message": "Run APG-Bench Test 1: Data Quality.",
                    "model": None,
                },
            ) as response:
                self.assertEqual(response.status_code, 200)
                events = [
                    json.loads(line)
                    for line in response.iter_lines()
                    if line
                ]
        finally:
            main.memory_manager = original_memory_manager

        deltas = [event for event in events if event.get("type") == "text-delta"]
        statuses = [event for event in events if event.get("type") == "status"]
        self.assertGreaterEqual(len(statuses), 2)
        self.assertEqual(statuses[0].get("stage"), "data_access")
        self.assertGreater(len(deltas), 3)
        self.assertTrue(all(len(event.get("delta", "")) <= 18 for event in deltas[:3]))
        self.assertTrue(fake_memory.append_calls)
        self.assertTrue(all(call["user_id"] == "user-1" for call in fake_memory.append_calls))
        self.assertEqual(fake_memory.last_messages_user_id, "user-1")

    def test_plot_line_fixture_returns_mui_line_spec(self):
        client = TestClient(app)
        response = client.get("/api/plots/test-line")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["plot_type"], "line")
        self.assertEqual(payload["x_type"], "time")
        self.assertEqual(payload["series"][0]["data"][0], {"x": "2024-01-02", "y": 184.89})
        self.assertEqual(len(payload["series"]), 2)
        self.assertEqual(payload["series"][1]["name"], "MSFT")

    def test_plot_scatter_fixture_returns_mui_scatter_spec(self):
        client = TestClient(app)
        response = client.get("/api/plots/test-scatter")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["plot_type"], "scatter")
        self.assertEqual(payload["hitAreaRadius"], 20)
        self.assertEqual(payload["grid"], {"horizontal": True, "vertical": True})
        self.assertEqual(payload["xAxis"][0]["min"], 0)
        self.assertEqual(payload["yAxis"][0]["width"], 60)
        self.assertEqual(payload["zAxis"][0]["max"], 10)
        self.assertEqual(payload["series"][0]["data"][0], {"x": 12, "y": 8, "z": 7, "id": "AAPL"})
        self.assertEqual(payload["series"][1]["markerSize"], 6)


if __name__ == "__main__":
    unittest.main()
