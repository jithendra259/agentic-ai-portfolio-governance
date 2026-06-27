# Chatbot Prior-Weight Injection and Cache Version Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reliably supply trusted current session weights to chatbot governance optimization and prevent legacy cached payloads from hiding optimization-audit fields.

**Architecture:** Resolve prior weights once from structured API session state and pass them through LangGraph configurable context. The governance tool wrapper gives explicit arguments precedence, uses session configuration as fallback, bypasses caching for weight-specific runs, and versions ordinary cache keys with a contract constant.

**Tech Stack:** Python 3, FastAPI, LangGraph configurable context, LangChain tools, `unittest`.

---

## File Structure

- Modify `backend/api/main.py`: validate session weights and attach them to all assistant configurations.
- Modify `backend/src/orchestrator/chatbot_orchestrator.py`: resolve configured weights, version cache keys, and bypass cache for weight-specific requests.
- Create `backend/test/test_chatbot_prior_weights_cache.py`: pure resolver and wrapper behavior tests.
- Modify `backend/test/test_chat_sessions_api.py`: assert REST, background, and streaming configuration propagation where practical.
- Do not commit implementation automatically because target files contain unrelated unstaged edits; stage only focused hunks if the user requests a commit.

### Task 1: Resolve trusted session weights

**Files:**
- Modify: `backend/api/main.py:580-620`
- Create: `backend/test/test_chatbot_prior_weights_cache.py`

- [ ] **Step 1: Write failing resolver tests**

```python
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import _resolve_previous_weights


class PreviousWeightResolutionTests(unittest.TestCase):
    def test_normalizes_valid_current_weights(self):
        resolved = {
            "session_state": {
                "active_weights": {
                    "type": "user_supplied",
                    "weights": {"aapl": 60, "MSFT": 40},
                    "approved_by_user": True,
                }
            }
        }
        self.assertEqual(_resolve_previous_weights(resolved), {"AAPL": 0.6, "MSFT": 0.4})

    def test_rejects_unapproved_equal_weight_proxy(self):
        resolved = {
            "session_state": {
                "active_weights": {
                    "type": "equal_weight_proxy",
                    "weights": {"AAPL": 0.5, "MSFT": 0.5},
                    "approved_by_user": False,
                }
            }
        }
        self.assertEqual(_resolve_previous_weights(resolved), {})

    def test_rejects_negative_or_nonfinite_weights(self):
        for weights in ({"AAPL": -0.1, "MSFT": 1.1}, {"AAPL": float("nan")}):
            with self.subTest(weights=weights):
                resolved = {"session_state": {"active_weights": {"weights": weights}}}
                self.assertEqual(_resolve_previous_weights(resolved), {})
```

- [ ] **Step 2: Run the resolver tests and verify RED**

```powershell
python -m unittest backend.test.test_chatbot_prior_weights_cache.PreviousWeightResolutionTests -v
```

Expected: import failure because `_resolve_previous_weights` does not exist.

- [ ] **Step 3: Implement the pure resolver**

```python
def _resolve_previous_weights(resolved_memory: dict[str, Any]) -> dict[str, float]:
    state = resolved_memory.get("session_state", {}) if isinstance(resolved_memory, dict) else {}
    active = state.get("active_weights", {}) if isinstance(state.get("active_weights"), dict) else {}
    if active.get("type") == "equal_weight_proxy" and not active.get("approved_by_user"):
        return {}
    raw = active.get("weights")
    if not isinstance(raw, dict) or not raw:
        return {}
    clean = {}
    for ticker, value in raw.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            return {}
        if not np.isfinite(number) or number < 0:
            return {}
        if str(ticker).strip() and number > 0:
            clean[str(ticker).strip().upper()] = number
    total = sum(clean.values())
    return {ticker: value / total for ticker, value in clean.items()} if total > 0 else {}
```

Add `import numpy as np` only if `math.isfinite` is not already available; prefer `math.isfinite` to avoid a new API startup dependency.

- [ ] **Step 4: Run the resolver tests and verify GREEN**

Run the command from Step 2. Expected: three tests pass.

### Task 2: Propagate prior weights through every assistant transport

**Files:**
- Modify: `backend/api/main.py:718`, `backend/api/main.py:1236`, `backend/api/main.py:1570`
- Test: `backend/test/test_chat_sessions_api.py`

- [ ] **Step 1: Add failing configuration-propagation assertions**

Use a recording fake assistant in the existing chat API tests and assert:

```python
configurable = fake_assistant.last_config["configurable"]
self.assertEqual(configurable["previous_weights"], {"AAPL": 0.6, "MSFT": 0.4})
```

Cover synchronous REST, background execution, and streaming event configuration using the existing fake orchestrator boundary.

- [ ] **Step 2: Run the affected API tests and verify RED**

```powershell
python -m unittest backend.test.test_chat_sessions_api -v
```

Expected: new assertions fail because `previous_weights` is absent.

- [ ] **Step 3: Add a single assistant-config helper and use it everywhere**

```python
def _assistant_config(request: ChatRequest, resolved_memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": request.session_id,
            "override_model": request.model,
            "previous_weights": _resolve_previous_weights(resolved_memory),
        }
    }
```

Replace all three inline assistant configuration dictionaries with `_assistant_config(request, resolved_memory)`.

- [ ] **Step 4: Run the API tests and verify GREEN**

Run the command from Step 2. Expected: all chat-session API tests pass.

### Task 3: Resolve configured weights and version governance cache keys

**Files:**
- Modify: `backend/src/orchestrator/chatbot_orchestrator.py:355-397`
- Test: `backend/test/test_chatbot_prior_weights_cache.py`

- [ ] **Step 1: Add failing wrapper tests**

```python
from unittest.mock import MagicMock, patch

from src.orchestrator.chatbot_orchestrator import GOVERNANCE_CACHE_VERSION, governance_pipeline_with_cache


class GovernanceCacheContractTests(unittest.TestCase):
    @patch("src.orchestrator.chatbot_orchestrator.run_full_governance_pipeline")
    @patch("src.orchestrator.chatbot_orchestrator.memory_manager")
    def test_configured_weights_bypass_cache_and_are_forwarded(self, memory, pipeline):
        pipeline.invoke.return_value = '{"status":"success"}'
        governance_pipeline_with_cache.func(
            tickers=["AAPL", "MSFT"],
            target_date="2026-06-20",
            config={"configurable": {"previous_weights": {"AAPL": 0.6, "MSFT": 0.4}}},
        )
        memory.retrieve_cached_plan.assert_not_called()
        memory.cache_governance_plan.assert_not_called()
        self.assertEqual(
            pipeline.invoke.call_args.args[0]["previous_weights"],
            {"AAPL": 0.6, "MSFT": 0.4},
        )

    @patch("src.orchestrator.chatbot_orchestrator.run_full_governance_pipeline")
    @patch("src.orchestrator.chatbot_orchestrator.memory_manager")
    def test_explicit_weights_override_configured_weights(self, memory, pipeline):
        pipeline.invoke.return_value = '{"status":"success"}'
        governance_pipeline_with_cache.func(
            tickers=["AAPL", "MSFT"],
            target_date="2026-06-20",
            previous_weights={"AAPL": 0.5, "MSFT": 0.5},
            config={"configurable": {"previous_weights": {"AAPL": 0.6, "MSFT": 0.4}}},
        )
        self.assertEqual(pipeline.invoke.call_args.args[0]["previous_weights"], {"AAPL": 0.5, "MSFT": 0.5})

    @patch("src.orchestrator.chatbot_orchestrator.run_full_governance_pipeline")
    @patch("src.orchestrator.chatbot_orchestrator.memory_manager")
    def test_unweighted_request_uses_versioned_cache_key(self, memory, pipeline):
        memory.compute_query_hash.return_value = "hash-v2"
        memory.retrieve_cached_plan.return_value = None
        pipeline.invoke.return_value = '{"status":"success"}'
        governance_pipeline_with_cache.func(tickers=["AAPL", "MSFT"], target_date="2026-06-20")
        self.assertIn(
            GOVERNANCE_CACHE_VERSION,
            memory.compute_query_hash.call_args.kwargs["risk_tolerance"],
        )
```

- [ ] **Step 2: Run the wrapper tests and verify RED**

```powershell
python -m unittest backend.test.test_chatbot_prior_weights_cache.GovernanceCacheContractTests -v
```

Expected: missing constant and configured weights are not forwarded.

- [ ] **Step 3: Implement precedence, cache bypass, and versioning**

```python
GOVERNANCE_CACHE_VERSION = "optimizer-audit-v2"


def _configured_previous_weights(config: RunnableConfig | None) -> dict[str, float]:
    configurable = (config or {}).get("configurable", {})
    value = configurable.get("previous_weights")
    return dict(value) if isinstance(value, dict) else {}
```

Inside `governance_pipeline_with_cache`:

```python
configured_weights = _configured_previous_weights(config)
resolved_previous_weights = previous_weights or configured_weights or None
cache_risk_tolerance = f"{normalized_risk_tolerance}|{GOVERNANCE_CACHE_VERSION}"
query_hash = memory_manager.compute_query_hash(
    tickers=tickers,
    target_date=target_date,
    risk_tolerance=cache_risk_tolerance,
)
```

Use `resolved_previous_weights` for cache bypass and tool forwarding. Log only the source label: `explicit`, `session`, or `unavailable`.

- [ ] **Step 4: Run the wrapper tests and verify GREEN**

Run the command from Step 2. Expected: three tests pass.

### Task 4: Verify the complete change

**Files:**
- Verify only.

- [ ] **Step 1: Run new and affected tests**

```powershell
python -m unittest backend.test.test_chatbot_prior_weights_cache backend.test.test_chatbot_optimizer_audit backend.test.test_chat_sessions_api backend.test.test_governance_plot_continuity -v
```

Expected: all tests pass.

- [ ] **Step 2: Compile modified files and check patch hygiene**

```powershell
python -m py_compile backend/api/main.py backend/src/orchestrator/chatbot_orchestrator.py backend/test/test_chatbot_prior_weights_cache.py
git diff --check -- backend/api/main.py backend/src/orchestrator/chatbot_orchestrator.py backend/test/test_chatbot_prior_weights_cache.py
```

Expected: both commands exit successfully.

- [ ] **Step 3: Review scope**

Confirm that prior recommendations are not treated as current holdings, unapproved proxies are rejected, weight-specific requests never use shared cache entries, ordinary requests use `optimizer-audit-v2`, and notebook/report/frontend files were not modified by this implementation.
