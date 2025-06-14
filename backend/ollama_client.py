import os
import requests

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-reasoner"
DEEPSEEK_API_KEY = os.getenv("sk-ebe886b5b79b469f8023fad761cbb3fc")  # Set your API key in your environment variables

def query_deepseek(prompt: str, system_prompt: str = None, max_tokens: int = 2048, **kwargs) -> str:
    if not DEEPSEEK_API_KEY:
        raise ValueError("DeepSeek API key not set in environment variable DEEPSEEK_API_KEY.")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    payload.update(kwargs)

    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    if not response.ok:
        raise RuntimeError(f"DeepSeek API response code: {response.status_code} {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]