import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from orchestrator.caveman_agent import detect_caveman_request, get_caveman_system_prompt

def test_caveman():
    # Test detection
    print("Testing detection...")
    print(f"Enter lite: {detect_caveman_request('please enter caveman mode lite')}")
    print(f"Enter ultra: {detect_caveman_request('/caveman ultra')}")
    print(f"Exit: {detect_caveman_request('back to normal mode please')}")
    
    # Test prompt application
    print("\nTesting prompt application...")
    lite_prompt = get_caveman_system_prompt("lite")
    print(f"Lite Prompt:\n{lite_prompt}")
    
    ultra_prompt = get_caveman_system_prompt("ultra")
    print(f"Ultra Prompt:\n{ultra_prompt}")
    
    if "maximum compression" in ultra_prompt.lower():
        print("\nUltra prompt seems correct.")

if __name__ == "__main__":
    test_caveman()
