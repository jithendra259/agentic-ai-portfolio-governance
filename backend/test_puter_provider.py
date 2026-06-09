"""Quick test for Puter backend adapter.

Usage:
    python test_puter_provider.py

Set env vars in backend/.env or export in your shell:
- PUTER_BASE_URL
- PUTER_API_KEY (optional)
"""
from src.providers.puter_provider import call_puter_chat, PuterError


def run_test():
    prompt = "Say hello in one sentence."
    model = "qwen/qwen3.7-plus"
    try:
        res = call_puter_chat(prompt, model=model, max_tokens=60)
        print("CALL SUCCESS")
        text = res.get('text')
        if text:
            print('Assistant:', text)
        else:
            print('No assistant text extracted — raw response:')
            import json
            print(json.dumps(res.get('raw'), indent=2)[:2000])
    except PuterError as e:
        print('CALL FAILED:', e)


if __name__ == '__main__':
    run_test()
