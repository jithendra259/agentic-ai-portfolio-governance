import unittest
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.providers.ashna_provider import ASHNA_OPENAI_BASE_URL, normalize_ashna_base_url


class AshnaProviderTests(unittest.TestCase):
    def test_normalizes_host_to_openai_compatible_root(self):
        self.assertEqual(
            normalize_ashna_base_url("https://api.ashna.ai"),
            ASHNA_OPENAI_BASE_URL,
        )

    def test_normalizes_v1_root_to_ashna_api_root(self):
        self.assertEqual(
            normalize_ashna_base_url("https://api.ashna.ai/v1"),
            ASHNA_OPENAI_BASE_URL,
        )

    def test_preserves_current_ashna_api_root(self):
        self.assertEqual(
            normalize_ashna_base_url("https://api.ashna.ai/v1/api"),
            ASHNA_OPENAI_BASE_URL,
        )

    def test_strips_endpoint_path(self):
        self.assertEqual(
            normalize_ashna_base_url("https://api.ashna.ai/v1/api/chat/completions"),
            ASHNA_OPENAI_BASE_URL,
        )


if __name__ == "__main__":
    unittest.main()
