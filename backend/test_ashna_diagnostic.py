#!/usr/bin/env python
"""Diagnostic script to test Ashna API connectivity and configuration."""

import os
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv

from src.providers.ashna_provider import normalize_ashna_base_url


load_dotenv(BACKEND_ROOT / ".env")
load_dotenv()


def test_ashna_config():
    """Test Ashna API configuration and connectivity."""
    print("=" * 80)
    print("ASHNA API DIAGNOSTIC TEST")
    print("=" * 80)

    api_key = os.getenv("ASHNA_API_KEY")
    base_url = os.getenv("ASHNA_BASE_URL")

    print(f"\nASHNA_API_KEY set: {bool(api_key)}")
    if api_key:
        print(f"  Key (masked): {api_key[:4]}...{api_key[-4:]}")

    print(f"ASHNA_BASE_URL set: {bool(base_url)}")
    if base_url:
        normalized_url = normalize_ashna_base_url(base_url)
        print(f"  URL: {base_url}")
        print(f"  Normalized OpenAI-compatible URL: {normalized_url}")

    if not api_key or not base_url:
        print("\nMissing configuration.")
        return False

    print("\n" + "=" * 80)
    print("TESTING ASHNA API CONNECTION")
    print("=" * 80)

    try:
        from langchain_openai import ChatOpenAI

        normalized_url = normalize_ashna_base_url(base_url)
        print("\nAttempting connection with:")
        print(f"  Base URL: {normalized_url}")
        print(f"  API Key: {'*' * len(api_key)}")

        llm = ChatOpenAI(
            model="ashnaai",
            api_key=api_key,
            base_url=normalized_url,
            timeout=10,
            max_retries=1,
        )

        print("\nSending test message to Ashna API...")
        response = llm.invoke("Say 'Hello from Ashna API test' in one sentence.")

        print("\nSUCCESS: Ashna API is responding")
        print(f"Response: {response.content}")
        return True

    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}")
        print(f"   {exc}")

        error_msg = str(exc).lower()
        if "invalid_request_error" in error_msg or "404" in error_msg:
            print("\nSuggestion: the endpoint might be incorrect.")
            print("   Try: ASHNA_BASE_URL=https://api.ashna.ai/v1/api")
        elif "authentication" in error_msg or "401" in error_msg or "unauthorized" in error_msg:
            print("\nSuggestion: the API key might be invalid or expired.")
            print("   Check your ASHNA_API_KEY in backend/.env.")
        elif "timeout" in error_msg or "timed out" in error_msg:
            print("\nSuggestion: network timeout or Ashna API is slow/down.")
            print("   Check your internet connection or try again later.")
        elif "connection" in error_msg:
            print("\nSuggestion: cannot reach the API endpoint.")
            print("   Check the ASHNA_BASE_URL is correct and accessible.")

        return False


def test_ollama_fallback():
    """Test if Ollama fallback is available."""
    print("\n" + "=" * 80)
    print("TESTING OLLAMA FALLBACK")
    print("=" * 80)

    try:
        from langchain_ollama import ChatOllama

        print("\nAttempting to connect to local Ollama...")
        llm = ChatOllama(
            model=os.getenv("PORTFOLIO_OLLAMA_MODEL", "qwen3-coder-next:cloud"),
            temperature=0.2,
        )
        response = llm.invoke("Say hello in one sentence.")

        print("\nSUCCESS: Ollama fallback is working")
        print(f"Response: {response.content[:200]}")
        return True

    except Exception as exc:
        print(f"\nOllama fallback not available: {type(exc).__name__}: {exc}")
        return False


if __name__ == "__main__":
    ashna_ok = test_ashna_config()
    ollama_ok = test_ollama_fallback()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Ashna API: {'Ready' if ashna_ok else 'Not responding'}")
    print(f"Ollama Fallback: {'Ready' if ollama_ok else 'Not available'}")
    print("\nIf Ashna API is not working, the system will:")
    print("  1. Log a warning")
    print("  2. Automatically fall back to qwen3-coder-next:cloud (Ollama)")
    print("=" * 80)
