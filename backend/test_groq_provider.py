"""
Diagnostic script for Groq API Integration.
Tests API Key, models list, and live chat completions.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_ROOT))

# Load .env
env_file = BACKEND_ROOT / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    print("=" * 60)
    print("[*] GROQ API INTEGRATION DIAGNOSTIC")
    print("=" * 60)

    api_key = os.getenv("GROQ_API_KEY")
    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    print(f"\n1. Environment Check:")
    print(f"   GROQ_API_KEY: {'[SET] ' + api_key[:8] + '...' + api_key[-4:] if api_key else '[NOT SET]'}")
    print(f"   GROQ_BASE_URL: {base_url}")
    print(f"   GROQ_MODEL: {model}")

    if not api_key:
        print("\n[!] Error: GROQ_API_KEY is missing in backend/.env")
        return False

    print("\n2. Testing Groq Models Endpoint...")
    models_url = f"{base_url.rstrip('/')}/models"
    try:
        req = urllib.request.Request(
            models_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Mozilla/5.0",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("id") for m in data.get("data", [])]
            print(f"   [+] Successfully retrieved {len(models)} models:")
            for m in models:
                print(f"      - {m}")
    except Exception as exc:
        print(f"   [!] Models endpoint failed: {exc}")
        return False

    print(f"\n3. Testing Chat Completion with model '{model}'...")
    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a quantitative financial risk advisor."},
            {"role": "user", "content": "What is CVaR (Conditional Value at Risk) in one concise sentence?"}
        ],
        "temperature": 0.2,
        "max_tokens": 100,
    }

    try:
        req = urllib.request.Request(
            chat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            answer = data["choices"][0]["message"]["content"].strip()
            safe_answer = answer.encode("ascii", "replace").decode("ascii")
            print(f"   [+] Response received:\n      \"{safe_answer}\"")
    except Exception as exc:
        print(f"   [!] Chat completion failed: {exc}")
        return False

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL GROQ API INTEGRATION CHECKS PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
