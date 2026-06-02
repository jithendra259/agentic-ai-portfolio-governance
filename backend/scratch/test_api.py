import requests
import json
import time

print("Testing complex query on chat/stream endpoint on port 8010...")
payload = {
    "session_id": "test-session-1234",
    "user_message": "Scatter plot of annualised volatility vs total return for U3 from 2020 to 2025"
}
start = time.time()
try:
    response = requests.post("http://127.0.0.1:8010/chat/stream", json=payload, stream=True, timeout=120)
    print("Chat stream status:", response.status_code)
    first_chunk = True
    for line in response.iter_lines():
        if line:
            if first_chunk:
                print("First chunk received in:", time.time() - start, "seconds")
                first_chunk = False
            data = json.loads(line)
            print(f"Event: {data.get('type')} | Content: {str(data)[:120]}")
    print("Total stream took:", time.time() - start, "seconds")
except Exception as e:
    print("Chat stream failed in", time.time() - start, "seconds:", e)
