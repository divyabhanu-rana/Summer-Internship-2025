# Ollama Local Setup

1. Download and install Ollama from [https://ollama.com/download](https://ollama.com/download)
2. Pull the desired model:
   ```sh
   ollama pull deepseek-r1:7b
   ```
3. Start the Ollama service (in its own terminal):
   ```sh
   ollama serve
   ```
4. Ollama API runs at `http://localhost:11434`
5. Update backend `.env` to point `OLLAMA_URL` to this address.