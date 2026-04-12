import os
import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.orchestrator.chatbot_orchestrator import _get_global_activity_summary

def verify_global_memory():
    print("Testing Global Memory Recovery...")
    summary = _get_global_activity_summary()
    if summary:
        print("\nSUCCESS! Found recent activity in the database:")
        print("-" * 40)
        print(summary)
        print("-" * 40)
    else:
        print("\nNo recent activity found. (Was there a governance run in the last 24h?)")

if __name__ == "__main__":
    verify_global_memory()
