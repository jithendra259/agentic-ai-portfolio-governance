#!/usr/bin/env python
"""
Test and verify Ashna API integration after fixes
Run this to confirm the Ashna models are responding correctly
"""
import asyncio
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

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def test_ashna_integration():
    """Test the complete Ashna integration after fixes"""
    print("\n" + "=" * 80)
    print("ASHNA API INTEGRATION TEST")
    print("=" * 80)
    
    # Test 1: Configuration check
    print("\n[1/4] Checking configuration...")
    api_key = os.getenv("ASHNA_API_KEY")
    base_url = os.getenv("ASHNA_BASE_URL")
    
    if not api_key:
        print("❌ ASHNA_API_KEY not set in .env")
        return False
    if not base_url:
        print("❌ ASHNA_BASE_URL not set in .env")
        return False
    
    print(f"✅ ASHNA_API_KEY: {api_key[:8]}...{api_key[-4:]}")
    print(f"✅ ASHNA_BASE_URL: {base_url}")
    
    # Test 2: URL format validation
    print("\n[2/4] Validating URL format...")
    if "/api" in base_url:
        print("⚠️  Warning: URL contains '/api' - code will remove it automatically")
    if not base_url.endswith(("/v1", "/")):
        print("ℹ️  URL will be normalized to include /v1")
    print("✅ URL format acceptable")
    
    # Test 3: Import and initialize
    print("\n[3/4] Testing ChatOpenAI initialization...")
    try:
        from langchain_openai import ChatOpenAI
        
        # Normalize URL like the code does
        normalized_url = base_url.rstrip("/")
        if normalized_url.endswith("/api"):
            normalized_url = normalized_url[:-4]
        if not normalized_url.endswith("/v1"):
            normalized_url = normalized_url + "/v1"
        
        print(f"   Base URL (normalized): {normalized_url}")
        
        llm = ChatOpenAI(
            model="ashnaai",
            api_key=api_key,
            base_url=normalized_url,
            timeout=15,
            max_retries=1,
        )
        print("✅ ChatOpenAI initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize ChatOpenAI: {e}")
        return False
    
    # Test 4: API call test
    print("\n[4/4] Testing API call to Ashna...")
    try:
        print("   Sending test prompt: 'Say hello in one sentence.'")
        response = llm.invoke("Say hello in one sentence.")
        print(f"✅ Response received!")
        print(f"   Content: {response.content[:100]}...")
        
    except Exception as e:
        print(f"❌ API call failed: {type(e).__name__}: {e}")
        print("\n💡 Troubleshooting suggestions:")
        
        error_msg = str(e).lower()
        if "404" in error_msg or "not found" in error_msg:
            print("   • The API endpoint is not found")
            print("   • Check if ASHNA_BASE_URL is correct")
            print("   • Try: ASHNA_BASE_URL=https://api.ashna.ai")
        elif "401" in error_msg or "unauthorized" in error_msg or "authentication" in error_msg:
            print("   • API authentication failed")
            print("   • Check if ASHNA_API_KEY is valid and not expired")
            print("   • Verify the key is correct in your .env file")
        elif "timeout" in error_msg or "timed out" in error_msg:
            print("   • Request timed out - Ashna API might be slow or down")
            print("   • Check your internet connection")
            print("   • Try again in a few moments")
        elif "connection" in error_msg:
            print("   • Cannot connect to the Ashna API")
            print("   • Verify the base URL is accessible")
            print("   • Check your firewall/proxy settings")
        else:
            print(f"   • Unexpected error: {e}")
            print("   • Check the Ashna API documentation for this error")
        
        return False
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - Ashna API is working!")
    print("=" * 80)
    return True


async def test_with_orchestrator():
    """Test using the actual orchestrator code"""
    print("\n" + "=" * 80)
    print("TESTING WITH CHATBOT ORCHESTRATOR")
    print("=" * 80)
    
    try:
        from src.orchestrator.chatbot_orchestrator import _get_chat_llm
        
        print("\n[1/2] Creating LLM with 'ashnaai' model...")
        llm = _get_chat_llm("ashnaai", temperature=0.2)
        print(f"✅ LLM created: {type(llm).__name__}")
        
        print("\n[2/2] Testing invocation...")
        message = "What is 2+2?"
        response = llm.invoke(message)
        print(f"✅ Response: {response.content[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run all tests"""
    success = await test_ashna_integration()
    
    if success:
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("""
The Ashna API is now correctly configured. The system will:
1. Use Ashna models when available (ashnaai, ashna-x1, ashna/gpt-4o, etc.)
2. Automatically fall back to Ollama (qwen3-coder-next:cloud) if Ashna fails
3. Log detailed information about model selection and failures

To verify it's working with the orchestrator:
  python backend/test_orchestrator_ashna.py
        """)
    else:
        print("\n" + "=" * 80)
        print("TROUBLESHOOTING")
        print("=" * 80)
        print("""
Please check:
1. ASHNA_API_KEY is set correctly in backend/.env
2. ASHNA_BASE_URL is set to: https://api.ashna.ai
3. Your internet connection is working
4. The Ashna API service is online (https://api.ashna.ai)

If the problem persists, the system will automatically fall back to Ollama models.
        """)


if __name__ == "__main__":
    asyncio.run(main())
