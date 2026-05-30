import requests
import json
import sys

def test_chat():
    url = "http://127.0.0.1:8000/chat/stream"
    payload = {
        "session_id": "test-session-123",
        "user_message": "Plot AAPL and MSFT prices from 2020 to 2024"
    }
    
    response = requests.post(url, json=payload, stream=True)
    
    print("Response status:", response.status_code)
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            print(f"Event type: {data.get('type')}")
            if data.get("type") == "data-plot":
                print("FOUND DATA-PLOT EVENT!")
                print(json.dumps(data, indent=2))
            elif data.get("type") == "tool-input-start":
                print(f"TOOL STARTED: {data.get('toolName')}")
            elif data.get("type") == "text-delta":
                print(f"TEXT: {data.get('delta', '').strip()}")

if __name__ == "__main__":
    test_chat()
