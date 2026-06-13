from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol


class ActionEmbeddingModel(Protocol):
    def embed(self, text: str) -> Mapping[str, float] | list[float]:
        """Return a vector representation for action arguments."""


@dataclass(frozen=True)
class LoopDetectionResult:
    detected: bool
    tool: str | None = None
    max_similarity: float = 0.0
    matched_action: dict[str, Any] | None = None
    reason: str | None = None


class HashingActionEmbeddingModel:
    """
    Lightweight local embedding fallback.

    It normalizes common finance/query synonyms, then uses token-frequency
    vectors. Production callers can pass an OpenAI or sentence-transformers
    adapter with the same embed(text) interface.
    """

    _ALIASES = {
        "aapl": "apple",
        "apple": "apple",
        "inc": "company",
        "stock": "price",
        "share": "price",
        "shares": "price",
        "value": "price",
        "quote": "price",
        "current": "current",
        "today": "current",
        "now": "current",
        "right": "current",
    }
    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "is",
        "of",
        "the",
        "to",
        "what",
        "with",
    }

    def embed(self, text: str) -> Mapping[str, float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        normalized = [
            self._ALIASES.get(token, token)
            for token in tokens
            if token not in self._STOPWORDS
        ]
        return Counter(normalized)


class SemanticLoopDetector:
    def __init__(
        self,
        embedding_model: ActionEmbeddingModel | None = None,
        similarity_threshold: float = 0.92,
        history_window: int = 5,
        min_matching_actions: int = 2,
    ) -> None:
        self.embedding_model = embedding_model or HashingActionEmbeddingModel()
        self.similarity_threshold = similarity_threshold
        self.history_window = history_window
        self.min_matching_actions = min_matching_actions

    def detect_loop(
        self,
        current_action: dict[str, Any],
        action_history: Iterable[dict[str, Any]],
    ) -> LoopDetectionResult:
        tool = self._tool_name(current_action)
        if not tool:
            return LoopDetectionResult(detected=False, reason="missing_tool")

        same_tool_actions = [
            action
            for action in self._recent_actions(action_history)
            if self._tool_name(action) == tool
        ]
        if len(same_tool_actions) < self.min_matching_actions:
            return LoopDetectionResult(detected=False, tool=tool, reason="insufficient_history")

        current_vector = self._embed_action_arguments(current_action)
        best_similarity = 0.0
        best_action: dict[str, Any] | None = None

        for previous_action in same_tool_actions:
            similarity = self._cosine_similarity(
                current_vector,
                self._embed_action_arguments(previous_action),
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_action = previous_action

        detected = best_similarity >= self.similarity_threshold
        return LoopDetectionResult(
            detected=detected,
            tool=tool,
            max_similarity=best_similarity,
            matched_action=best_action if detected else None,
            reason="semantic_repeat" if detected else "below_threshold",
        )

    def _recent_actions(self, action_history: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        entries = list(action_history)[-self.history_window :]
        actions = []
        for entry in entries:
            action = entry.get("action", entry) if isinstance(entry, dict) else entry
            if isinstance(action, dict):
                actions.append(action)
        return actions

    def _embed_action_arguments(self, action: dict[str, Any]) -> Mapping[str, float] | list[float]:
        payload = action.get("parameters", action.get("params", {}))
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return self.embedding_model.embed(serialized)

    @staticmethod
    def _tool_name(action: dict[str, Any]) -> str | None:
        tool = action.get("tool") or action.get("tool_name") or action.get("name")
        return str(tool) if tool else None

    @staticmethod
    def _cosine_similarity(
        vec_a: Mapping[str, float] | list[float],
        vec_b: Mapping[str, float] | list[float],
    ) -> float:
        if isinstance(vec_a, Mapping) and isinstance(vec_b, Mapping):
            keys = set(vec_a) | set(vec_b)
            dot = sum(float(vec_a.get(key, 0.0)) * float(vec_b.get(key, 0.0)) for key in keys)
            norm_a = math.sqrt(sum(float(value) ** 2 for value in vec_a.values()))
            norm_b = math.sqrt(sum(float(value) ** 2 for value in vec_b.values()))
        else:
            if not isinstance(vec_a, list) or not isinstance(vec_b, list):
                return 0.0
            size = min(len(vec_a), len(vec_b))
            dot = sum(float(vec_a[i]) * float(vec_b[i]) for i in range(size))
            norm_a = math.sqrt(sum(float(value) ** 2 for value in vec_a))
            norm_b = math.sqrt(sum(float(value) ** 2 for value in vec_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
