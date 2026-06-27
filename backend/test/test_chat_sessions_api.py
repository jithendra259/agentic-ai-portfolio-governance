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
fake_orchestrator.get_postgres_status = lambda: "connected"
previous_orchestrator_module = sys.modules.get("src.orchestrator.chatbot_orchestrator")
sys.modules["src.orchestrator.chatbot_orchestrator"] = fake_orchestrator

from api.main import app
from src.utils.crypto_utils import create_auth_token

if previous_orchestrator_module is None:
    sys.modules.pop("src.orchestrator.chatbot_orchestrator", None)
else:
    sys.modules["src.orchestrator.chatbot_orchestrator"] = previous_orchestrator_module

orchestrator_package = sys.modules.get("src.orchestrator")
if orchestrator_package is not None and getattr(orchestrator_package, "chatbot_orchestrator", None) is fake_orchestrator:
    delattr(orchestrator_package, "chatbot_orchestrator")


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
    def test_auth_login_preflight_allows_local_vite_port(self):
        client = TestClient(app)
        response = client.options(
            "/api/auth/login",
            headers={
                "Origin": "http://localhost:5000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,authorization",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5000")

    def test_auth_login_preflight_allows_private_network_vite_url(self):
        client = TestClient(app)
        response = client.options(
            "/api/auth/login",
            headers={
                "Origin": "http://10.61.12.179:5000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,authorization",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://10.61.12.179:5000")

    def test_chat_sessions_endpoint_requires_authentication(self):
        import api.main as main

        fake_memory = FakeMemoryManager()
        original_memory_manager = main.memory_manager
        main.memory_manager = fake_memory
        try:
            client = TestClient(app)
            response = client.get("/chat/sessions?limit=25")
        finally:
            main.memory_manager = original_memory_manager

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"detail": "Authentication is required to load chat sessions"})
        self.assertIsNone(fake_memory.last_list_user_id)
        self.assertIsNone(fake_memory.last_legacy_session_ids)

    def test_chat_sessions_endpoint_uses_authenticated_user_without_legacy_ids(self):
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
        self.assertEqual(fake_memory.last_legacy_session_ids, [])

    def test_chat_sessions_endpoint_accepts_clerk_session_token(self):
        import api.main as main

        fake_memory = FakeMemoryManager()
        original_memory_manager = main.memory_manager
        original_verify_clerk_token = getattr(main, "verify_clerk_token", None)
        main.memory_manager = fake_memory
        main.verify_clerk_token = lambda token: {
            "user": {
                "id": "user_clerk_123",
                "email": "clerk@example.com",
                "name": "Clerk User",
            }
        } if token == "clerk-session-token" else None
        try:
            client = TestClient(app)
            response = client.get(
                "/chat/sessions?limit=25",
                headers={"Authorization": "Bearer clerk-session-token"},
            )
        finally:
            main.memory_manager = original_memory_manager
            if original_verify_clerk_token is None:
                delattr(main, "verify_clerk_token")
            else:
                main.verify_clerk_token = original_verify_clerk_token

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_memory.last_list_user_id, "user_clerk_123")

    def test_auth_session_returns_clerk_session_payload(self):
        import api.main as main
        import api.auth_router as auth_router

        def fake_verify_clerk_token(token):
            return {
                "user": {
                    "id": "user_clerk_123",
                    "email": "clerk@example.com",
                    "name": "Clerk User",
                }
            } if token == "clerk-session-token" else None

        original_verify_clerk_token = getattr(main, "verify_clerk_token", None)
        original_router_verify_clerk_token = auth_router.verify_clerk_token
        main.verify_clerk_token = fake_verify_clerk_token
        auth_router.verify_clerk_token = fake_verify_clerk_token
        try:
            client = TestClient(app)
            response = client.get(
                "/api/auth/session",
                headers={"Authorization": "Bearer clerk-session-token"},
            )
        finally:
            if original_verify_clerk_token is None:
                delattr(main, "verify_clerk_token")
            else:
                main.verify_clerk_token = original_verify_clerk_token
            auth_router.verify_clerk_token = original_router_verify_clerk_token

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "session": {
                    "user": {
                        "id": "user_clerk_123",
                        "email": "clerk@example.com",
                        "name": "Clerk User",
                    }
                }
            },
        )

    def test_chat_messages_endpoint_does_not_include_legacy_rows_for_authenticated_users(self):
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
        self.assertFalse(fake_memory.last_include_legacy_unowned)

    def test_claim_legacy_chat_sessions_is_disabled_for_user_isolation(self):
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
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json(), {"detail": "Legacy chat claiming is disabled"})
        self.assertIsNone(fake_memory.last_claim_user_id)
        self.assertIsNone(fake_memory.last_claim_session_ids)
        self.assertIsNone(fake_memory.last_claim_all)

    def test_delete_chat_session_endpoint_deletes_history(self):
        import api.main as main

        fake_memory = FakeMemoryManager()
        original_memory_manager = main.memory_manager
        main.memory_manager = fake_memory
        main.session_memory_store.save_state("session-1", {"session_id": "session-1", "universe": {"tickers": ["AAPL"]}})
        main.session_memory_store.append_message("session-1", "user", "hello")
        token = create_auth_token({"user": {"id": "user-1", "email": "user@example.com"}})
        try:
            client = TestClient(app)
            response = client.delete("/chat/session-1", headers={"Authorization": f"Bearer {token}"})
        finally:
            main.memory_manager = original_memory_manager

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"session_id": "session-1", "deleted_count": 2})
        self.assertEqual(fake_memory.deleted_session_id, "session-1")
        self.assertEqual(fake_memory.deleted_user_id, "user-1")
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

    def test_governance_scatter_followup_uses_committed_run_not_fundamentals(self):
        import api.main as main

        session_id = "healthcare-governance-scatter"
        fake_memory = FakeMemoryManager()
        original_memory_manager = main.memory_manager
        main.memory_manager = fake_memory
        main.session_memory_store.delete_session(session_id)
        main.session_memory_store.update_state(
            session_id,
            {"active_universe": "Healthcare"},
        )
        main._store_governance_tool_result(
            session_id,
            json.dumps(
                {
                    "status": "success",
                    "target_date": "2026-06-20",
                    "valid_tickers": ["JNJ", "PFE", "UNH"],
                    "systemic_risk": {"scores": {"JNJ": 0.31, "PFE": 0.47, "UNH": 0.22}},
                    "optimization": {
                        "weights": {"JNJ": 0.55, "PFE": 0.0, "UNH": 0.45},
                        "instability_index": 0.18,
                        "lambda_t": 0.7,
                    },
                }
            ),
        )

        try:
            client = TestClient(app)
            clarify = client.post(
                "/chat",
                json={"session_id": session_id, "user_message": "plot scatter plot", "model": None},
            )
            axes = client.post(
                "/chat",
                json={
                    "session_id": session_id,
                    "user_message": "systemic risk score vs portfolio weight",
                    "model": None,
                },
            )
            repeat = client.post(
                "/chat",
                json={"session_id": session_id, "user_message": "plot the scatter plot", "model": None},
            )
        finally:
            main.memory_manager = original_memory_manager

        self.assertEqual(clarify.status_code, 200)
        self.assertIn("which governance metrics", clarify.json()["response"].lower())
        for response in (axes, repeat):
            self.assertEqual(response.status_code, 200)
            text = response.json()["response"]
            self.assertIn("Systemic Risk Score vs Portfolio Weight", text)
            self.assertNotIn("MongoDB", text)
            self.assertNotIn("Beta", text)
            self.assertNotIn("Forward P/E", text)
            plot_id = text.split("__PLOTSPEC__:", 1)[1]
            plot = main.GLOBAL_PLOT_DATA[plot_id]
            self.assertEqual(plot["data_source"], "latest_governance_run")
            self.assertEqual(plot["series"][0]["data"][1]["id"], "PFE")
            self.assertEqual(plot["series"][0]["data"][1]["y"], 0.0)

    def test_chat_start_dispatches_background_run_and_records_status(self):
        import api.main as main

        token = create_auth_token({"user": {"id": "user-1", "email": "user@example.com"}})
        fake_memory = FakeMemoryManager()
        original_memory_manager = main.memory_manager
        main.memory_manager = fake_memory
        main.CHAT_RUNS.clear()
        try:
            client = TestClient(app)
            response = client.post(
                "/chat/start",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "session_id": "background-session-1",
                    "user_message": "Run APG-Bench Test 1: Data Quality.",
                    "model": None,
                },
            )
            payload = response.json()
            status_response = client.get(f"/chat/runs/{payload['thread_id']}")
        finally:
            main.memory_manager = original_memory_manager

        self.assertEqual(response.status_code, 202)
        self.assertEqual(payload["status"], "started")
        self.assertEqual(payload["session_id"], "background-session-1")
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.json()["status"], "completed")
        self.assertIn("response", status_response.json())
        self.assertTrue(fake_memory.append_calls)
        self.assertTrue(all(call["user_id"] == "user-1" for call in fake_memory.append_calls))

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

    def test_plot_endpoint_returns_chat_generated_memory_plot(self):
        from src.agents.plot_store import GLOBAL_PLOT_DATA

        plot_id = "chat_generated_plot_1"
        GLOBAL_PLOT_DATA[plot_id] = {
            "plot_type": "gauge",
            "title": "Chat Generated Gauge",
            "value": 88,
            "valueMin": 0,
            "valueMax": 100,
        }
        try:
            client = TestClient(app)
            response = client.get(f"/api/plots/{plot_id}")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["plot_type"], "gauge")
            self.assertEqual(payload["title"], "Chat Generated Gauge")
            self.assertEqual(payload["value"], 88)
        finally:
            GLOBAL_PLOT_DATA.pop(plot_id, None)


if __name__ == "__main__":
    unittest.main()
