import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))  # Ensure .env is loaded
import requests

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-reasoner"  # or "deepseek-chat" according to your purchase
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # <--- FIXED HERE

def ask_deepseek(prompt: str, system_prompt: str = None, stream: bool = False, max_tokens: int = 2048, **kwargs) -> str:
    """
    Sends a prompt to the DeepSeek cloud API and returns the response.
    """
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
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    payload.update(kwargs)

    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, stream=stream)
    if not response.ok:
        raise RuntimeError(f"DeepSeek API response code: {response.status_code} {response.text}")

    if stream:
        output = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = line.decode('utf-8')
                    import json
                    chunk = json.loads(data)
                    output += chunk["choices"][0]["delta"].get("content", "")
                except Exception:
                    continue
        return output
    else:
        data = response.json()
        return data["choices"][0]["message"]["content"]