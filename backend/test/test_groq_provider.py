import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.providers.groq_provider import (
    GROQ_OPENAI_BASE_URL,
    DEFAULT_GROQ_MODEL,
    normalize_groq_base_url,
    normalize_groq_model_name,
    is_groq_model,
)


class GroqProviderTests(unittest.TestCase):
    def test_normalizes_base_url(self):
        self.assertEqual(
            normalize_groq_base_url("https://api.groq.com/openai/v1"),
            GROQ_OPENAI_BASE_URL,
        )
        self.assertEqual(
            normalize_groq_base_url("https://api.groq.com/openai/v1/chat/completions"),
            GROQ_OPENAI_BASE_URL,
        )
        self.assertEqual(
            normalize_groq_base_url("https://api.groq.com/openai"),
            GROQ_OPENAI_BASE_URL,
        )

    def test_normalizes_model_names(self):
        self.assertEqual(
            normalize_groq_model_name("groq/openai/gpt-oss-120b"),
            "openai/gpt-oss-120b",
        )
        self.assertEqual(
            normalize_groq_model_name("groq/llama-3.3-70b-versatile"),
            "llama-3.3-70b-versatile",
        )
        self.assertEqual(
            normalize_groq_model_name("openai/gpt-oss-120b"),
            "openai/gpt-oss-120b",
        )
        self.assertEqual(
            normalize_groq_model_name(""),
            DEFAULT_GROQ_MODEL,
        )

    def test_identifies_groq_models(self):
        self.assertTrue(is_groq_model("groq/openai/gpt-oss-120b"))
        self.assertTrue(is_groq_model("openai/gpt-oss-120b"))
        self.assertTrue(is_groq_model("qwen/qwen3.8-27b"))
        self.assertTrue(is_groq_model("llama-3.3-70b-versatile"))
        self.assertTrue(is_groq_model("groq"))
        self.assertFalse(is_groq_model("mistral:latest"))
        self.assertFalse(is_groq_model("ashnaai"))


if __name__ == "__main__":
    unittest.main()
