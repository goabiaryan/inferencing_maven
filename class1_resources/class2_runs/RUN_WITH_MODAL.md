# Run RelayServe with Local + Modal (Class 2)

**Follow-up to Class 1.** You ran one mock backend and RelayServe via `RELAYSERVE_BACKENDS`. Class 2 adds **config-driven routing**: one gateway, **local** and **Modal** backends; the request `model` field picks which one. No client code changes when you switch.

**What this gives you:** Local llama.cpp (port 8081) + Modal GPU (deployed app). RelayServe (port 8080) routes by `model`. Test script hits both and prints latency + replies.

---

## One-time setup

### 1. Deploy Modal

```bash
cd "/path/to/class1_resources"
pip install modal
modal token set   # or modal token new (browser); if already done, skip
bash scripts/deploy_modal.sh
```

Copy the printed Modal web URL (e.g. `https://<your-modal-app>.modal.run`).

### 2. Set Modal URL in config

Edit `RelayServe/config.yaml` and set `backends.modal.url` to that URL (no trailing slash).

---

## Run (three terminals)

Use **three terminals**. Base path: `/path/to/class1_resources`. Adjust paths if your repo is elsewhere.

### Terminal 1 – Local llama.cpp

Copy-paste the block below. **Leave this terminal running** until the server is listening on 8081.

```bash
cd "/path/to/class1_resources"
export LLAMA_SERVER_PATH="/path/to/class1_resources/llama_server_bin/bin/llama-server"
export LLAMA_MODEL_PATH="/path/to/class1_resources/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
export LLAMA_PORTS=8081
python scripts/spawn_backends.py
```

### Terminal 2 – RelayServe

Copy-paste the block below. **Leave this terminal running.**

```bash
cd "/path/to/class1_resources"
bash scripts/start_relay_class2.sh
```

You should see `Listening: :8080` and **`Backends: none`**. That’s correct: backends come from `RelayServe/config.yaml` (local + modal). The terminal will sit with no new output—RelayServe is running and waiting for requests. Use another terminal for the next step.

### Terminal 3 – Test both backends

Copy-paste the block below. The script will **ask you to enter a prompt** (e.g. “Explain KV cache in one sentence”); type it and press Enter. It then runs the latency tests and prints replies for each backend.

```bash
cd "/path/to/class1_resources"
bash scripts/test_modal_backend.sh
```

You should see **local** and **modal** replies and latency. First Modal request can take 5–30 s (cold start); later ones are faster.

---

## Quick curl checks

**Local:**  
`curl -s -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -H "Accept: application/json" -d '{"model":"local","messages":[{"role":"user","content":"Hi"}],"stream":false}'`

**Modal:**  
Same, with `"model":"modal"`.

---

## Lab checklist (what this runbook does)

| Task | How you do it |
|------|----------------|
| Deploy Modal | Step 1 above; paste URL into `config.yaml` |
| Config-driven routing | `config.yaml` defines local + modal; request `model` selects backend |
| Test all three | `scripts/test_modal_backend.sh` — local llama, modal llama, modal vLLM (optional) |
| Stretch: extra backend | Add another entry in `config.yaml` (e.g. `modal_phi`) and use `"model":"modal_phi"` |

---

## Optional: Modal vLLM

The test script runs **three** backends: **local (llama.cpp)**, **modal (llama.cpp)**, and **modal_vllm**. The first two are required; the third is optional.

To add **Modal vLLM**: deploy a vLLM app on Modal (separate from the llama app), then set `backends.modal_vllm.url` in `config.yaml` to that app’s URL. The test script sends a **cold** request first (container spin-up + model load), then a **warm** request; the summary shows both. Modal vLLM scales to zero, so the first request is often 3–4s; the warm request is the one that reflects real inference latency. If you don’t set the URL, the script prints a WARNING for `modal_vllm` and continues.

---

## Demo without real backends (notebook)

To see routing **without** deploying Modal or running llama: open **RelayServe_Class2_Demo.ipynb** and run all cells. It uses mock backends and a temporary config.

---

## Paths and other docs

- **If llama paths don’t exist:** Get the binary and model from [Serve_local_model.md](../class1_runs/Serve_local_model.md), then set `LLAMA_SERVER_PATH` and `LLAMA_MODEL_PATH` in Terminal 1 to your actual paths.
- **More material:** [CLASS2_EXTENDING_RELAYSERVE.md](../CLASS2_EXTENDING_RELAYSERVE.md), [visualizations/](../visualizations/README.md).
