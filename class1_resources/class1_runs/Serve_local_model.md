# Demo: Serve a Local Model with RelayServe

Use this in class to **demo** how to run a small LLM on the student’s machine with RelayServe, then use the **Components walkthrough** to explain how it works.

**Goal:** From a clean environment to a working RelayServe that answers `POST /v1/chat/completions` in **under 30 minutes**.

> **Note:** In the commands below, replace `/path/to/class1_resources` with the actual path to your class1_resources folder (e.g. `~/Desktop/.../class1_resources` or the full path).

---

## Prerequisites (one-time)

- **Python 3.9+**
- **Prebuilt llama.cpp server** (we use the official binaries; no build required)
- **One small GGUF model** (e.g. [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) or [TinyLlama](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF))

---

## Step 1: Get the model (GGUF)

Pick one and download a **Q4_K_M** (or similar) GGUF file to a known path.

**Option A – Script (TinyLlama, small and fast):**  
The script always saves the model in the repo at `class1_resources/models/` (no other location).
```bash
pip install huggingface_hub
cd /path/to/class1_resources
python scripts/download_model.py
# Set the path it prints (always repo/models/), e.g.:
export LLAMA_MODEL_PATH="/path/to/class1_resources/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
```

**Option B – Manual (TinyLlama):**  
Download from [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) (e.g. `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`), then:
```bash
export LLAMA_MODEL_PATH="$HOME/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
mkdir -p "$(dirname "$LLAMA_MODEL_PATH")"
```

**Option C – Phi-3-mini:**
```bash
# e.g. from Hugging Face: microsoft/Phi-3-mini-4k-instruct-gguf
export LLAMA_MODEL_PATH="$HOME/models/Phi-3-mini-4k-instruct-q4.gguf"
```

Use whatever path you actually downloaded the file to.

---

## Step 2: Get the llama.cpp server (prebuilt)

**Option A – Script (recommended):** downloads the right prebuilt for your OS and extracts it into the repo.
```bash
cd /path/to/class1_resources
python scripts/download_llama_server.py
# Then set the path it prints, e.g.:
export LLAMA_SERVER_PATH="/path/to/class1_resources/llama_server_bin/bin/llama-server"
```

**Option B – Manual:** open [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases), download the archive for your system (e.g. `llama-*-bin-macos-arm64.tar.gz` for Apple Silicon), extract it, and set `LLAMA_SERVER_PATH` to the path of `llama-server` (or `llama-server.exe` on Windows) inside the extracted folder.

**Verify before Step 3:** After setting both `LLAMA_SERVER_PATH` and `LLAMA_MODEL_PATH`, run:
```bash
cd /path/to/class1_resources
python scripts/check_llama_setup.py
```
It checks that the server binary exists and is executable, and that the model file exists.

---

## Step 3: Start the backend (llama.cpp server)

In a **dedicated terminal**:

```bash
export LLAMA_SERVER_PATH="/path/to/class1_resources/llama_server_bin/bin/llama-server"
export LLAMA_MODEL_PATH="/path/to/class1_resources/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
export LLAMA_PORTS=8081
cd "/path/to/class1_resources"
python scripts/spawn_backends.py
```

Leave this running. You should see the server loading the model and listening on port 8081.

---

## Step 4: Start RelayServe

In a **second terminal**:

```bash
cd "/path/to/class1_resources/RelayServe"
pip install -e .
export RELAYSERVE_BACKENDS=http://localhost:8081
relayserve
```

You should see something like:
```
Relay starting
- Listening: :8080
- Model: relay-gguf
- Backends: http://localhost:8081
```

**If you get "Echo from mock backend"** when you curl: something else is already using port 8080 (usually the **RelayServe Demo** notebook). Do this:
1. **Free port 8080:** Restart the Jupyter kernel or close the notebook that ran the demo, or run: `lsof -i :8080 -t | xargs kill -9`
2. **Start RelayServe again** with the real backend. Either run the same `export` and `relayserve` as above in a **new** terminal, or use the helper script (env var is set for you):
   ```bash
   cd "/path/to/class1_resources"
   bash scripts/start_relay_tinyllama.sh
   ```
   When it starts, the log must show `Backends: http://localhost:8081`. Then curl will hit TinyLlama and you’ll get real replies, not echo.

---

## Step 5: Test the API

Responses use **pretty** format by default (readable “Relay Response” with Reply, Device, Backend, Queue, TTFT). In a **third terminal** (or same as Step 4 after backgrounding RelayServe):

**Recommended – streaming script (pretty output):**
```bash
cd "/path/to/class1_resources"
python scripts/streaming_chat.py "What is 2+2?" "class-demo"
```
Prints status, headers, then the streamed reply and `[DONE]`.

**Non-streaming (curl, pretty):**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"relay-gguf","messages":[{"role":"user","content":"Tell me about life"}]}'
```
You’ll see “Relay Response”, Reply, Device, Backend, Queue, TTFT.

**With request-id (curl -i to see headers):**
```bash
curl -i -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-123" \
  -d '{"model":"relay-gguf","messages":[{"role":"user","content":"Tell me about life"}]}'
```

Check the response header `X-Request-ID: demo-123`.

**Streaming (curl):**
```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"relay-gguf","messages":[{"role":"user","content":"Tell me about life"}],"stream":true}'
```
You should see `data: {...}` lines then `data: [DONE]`.

---

## Quick reference (all in one place)

```bash
# Terminal 1 – backend (from class1_resources root)
cd /path/to/class1_resources
export LLAMA_SERVER_PATH=/path/to/llama-server
export LLAMA_MODEL_PATH=/path/to/model.gguf
export LLAMA_PORTS=8081
python scripts/spawn_backends.py

# Terminal 2 – RelayServe (from class1_resources/RelayServe)
cd /path/to/class1_resources/RelayServe
pip install -e .
export RELAYSERVE_BACKENDS=http://localhost:8081
relayserve

# Terminal 3 – test (pretty output)
cd /path/to/class1_resources
python scripts/streaming_chat.py "Hello" "demo"
# Or curl (also pretty by default):
# curl -X POST http://localhost:8080/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{"model":"relay-gguf","messages":[{"role":"user","content":"Hello"}]}'
```


