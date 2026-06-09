#!/usr/bin/env python
"""
Diagnostic script to test Ashna API connectivity and configuration
"""
import os
import sys
from pathlib import Path

# Add backend to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

def test_ashna_config():
    """Test Ashna API configuration and connectivity"""
    print("=" * 80)
    print("ASHNA API DIAGNOSTIC TEST")
    print("=" * 80)
    
    api_key = os.getenv("ASHNA_API_KEY")
    base_url = os.getenv("ASHNA_BASE_URL")
    
    print(f"\n✓ ASHNA_API_KEY set: {bool(api_key)}")
    if api_key:
        print(f"  Key (masked): {api_key[:4]}...{api_key[-4:]}")
    
    print(f"✓ ASHNA_BASE_URL set: {bool(base_url)}")
    if base_url:
        print(f"  URL: {base_url}")
        
        # Check URL format
        if "/v1/api" in base_url:
            print("\n⚠️  WARNING: Base URL contains '/v1/api'")
            print("   Standard OpenAI-compatible APIs use '/v1' only")
            print("   The '/chat/completions' endpoint is auto-appended")
            print(f"   Current: {base_url}/chat/completions")
            print(f"   Suggested: {base_url.replace('/v1/api', '/v1')}/chat/completions")
        elif not base_url.endswith("/v1"):
            print(f"\n⚠️  WARNING: Base URL should end with '/v1'")
            print(f"   Suggested: {base_url.rstrip('/')}/v1")
    
    if not api_key or not base_url:
        print("\n❌ Missing configuration!")
        return False
    
    # Test connection
    print("\n" + "=" * 80)
    print("TESTING ASHNA API CONNECTION...")
    print("=" * 80)
    
    try:
        from langchain_openai import ChatOpenAI
        
        # Test with the current configuration
        print(f"\nAttempting connection with:")
        print(f"  Base URL: {base_url}")
        print(f"  API Key: {'*' * len(api_key)}")
        
        llm = ChatOpenAI(
            model="ashnaai",
            api_key=api_key,
            base_url=base_url,
            timeout=10,
            max_retries=1
        )
        
        print("\n⏳ Sending test message to Ashna API...")
        response = llm.invoke("Say 'Hello from Ashna API test' in one sentence.")
        
        print("\n✅ SUCCESS! Ashna API is responding")
        print(f"Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}")
        print(f"   {str(e)}")
        
        # Provide diagnostic suggestions
        error_msg = str(e).lower()
        if "invalid_request_error" in error_msg or "404" in error_msg:
            print("\n💡 Suggestion: The endpoint might be incorrect")
            print("   Try changing ASHNA_BASE_URL to remove '/api' suffix")
        elif "authentication" in error_msg or "401" in error_msg or "unauthorized" in error_msg:
            print("\n💡 Suggestion: The API key might be invalid or expired")
            print("   Check your ASHNA_API_KEY in the .env file")
        elif "timeout" in error_msg or "timed out" in error_msg:
            print("\n💡 Suggestion: Network timeout or Ashna API is slow/down")
            print("   Check your internet connection or try again later")
        elif "connection" in error_msg:
            print("\n💡 Suggestion: Cannot reach the API endpoint")
            print("   Check the ASHNA_BASE_URL is correct and accessible")
        
        return False


def test_ollama_fallback():
    """Test if Ollama fallback is available"""
    print("\n" + "=" * 80)
    print("TESTING OLLAMA FALLBACK...")
    print("=" * 80)
    
    try:
        from langchain_ollama import ChatOllama
        
        print("\nAttempting to connect to local Ollama...")
        llm = ChatOllama(
            model="qwen3-coder-next:cloud",
            timeout=5
        )
        
        print("⏳ Sending test message to Ollama...")
        response = llm.invoke("Say 'Hello from Ollama' in one sentence.")
        
        print("\n✅ SUCCESS! Ollama fallback is available")
        print(f"Response: {response.content[:100]}...")
        return True
        
    except Exception as e:
        print(f"\n❌ Ollama fallback unavailable: {type(e).__name__}")
        print(f"   {str(e)[:200]}")
        return False


if __name__ == "__main__":
    ashna_ok = test_ashna_config()
    ollama_ok = test_ollama_fallback()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Ashna API: {'✅ Ready' if ashna_ok else '❌ Not responding'}")
    print(f"Ollama Fallback: {'✅ Ready' if ollama_ok else '❌ Not available'}")
    print("\nIf Ashna API is not working, the system will:")
    print("  1. Log a warning")
    print("  2. Automatically fall back to qwen3-coder-next:cloud (Ollama)")
    print("=" * 80)
