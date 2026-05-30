import requests
import json
import uuid

def test_chat():
    url = 'http://127.0.0.1:8000/chat/stream'
    rand_session = str(uuid.uuid4())
    payload = {
        'session_id': rand_session,
        'user_message': 'Plot AAPL and MSFT prices from 2020 to 2024'
    }
    
    response = requests.post(url, json=payload, stream=True)
    
    print('Response status:', response.status_code)
    for line in response.iter_lines():
        if line:
            raw = line.decode('utf-8')
            print(raw)

if __name__ == '__main__':
    test_chat()
