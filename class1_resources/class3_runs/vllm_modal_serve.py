from __future__ import annotations

import os
import subprocess

import modal

# --- Image: vLLM + Hugging Face ---
# First deploy can take 10–15 min (image build + model download). Later deploys use cache.
# Using runtime (not devel) base for faster image pull; pip vLLM wheels work with it.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-runtime-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .pip_install("vllm", "huggingface-hub")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# --- Model and GPU ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # small, non-gated; change as needed
GPU_TYPE = "A10G"  # or "T4", "H100", etc.
N_GPU = 1
VLLM_PORT = 8000
MINUTES = 60

# --- Volumes: cache model weights and vLLM JIT artifacts (per Modal docs) ---
hf_cache_vol = modal.Volume.from_name("vllm-demo-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-demo-vllm-cache", create_if_missing=True)

# Fast cold start (enforce-eager); set False for better throughput when replicas stay warm
FAST_BOOT = True

# Throughput tuning (see notebook "vLLM tuning: 6 knobs"): set True to use throughput-heavy presets
THROUGHPUT_TUNED = False  # set True, redeploy, then compare in notebook benchmark

app = modal.App("vllm-chat-demo")

# Secret must contain HF_TOKEN for gated Hugging Face models. Create with:
#   modal secret create huggingface HF_TOKEN=hf_...
# Non-gated models (e.g. Qwen2.5-1.5B) work without it.
HF_SECRET = modal.Secret.from_name("huggingface")


@app.function(secrets=[HF_SECRET])
def check_hf_secret():
    """Verify the huggingface secret is available (e.g. for gated models)."""
    if os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is set (huggingface secret OK).")
    else:
        print("HF_TOKEN is not set. For gated models, create: modal secret create huggingface HF_TOKEN=hf_...")


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    timeout=10 * MINUTES,
    scaledown_window=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[HF_SECRET],
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--tensor-parallel-size", str(N_GPU),
    ]
    if FAST_BOOT:
        cmd += ["--enforce-eager"]
    if THROUGHPUT_TUNED:
        cmd += [
            "--max-num-batched-tokens", "16384",
            "--gpu-memory-utilization", "0.95",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
        ]
    print(" ".join(cmd), flush=True)
    subprocess.Popen(cmd)


@app.local_entrypoint()
def main():
    """Run this app to get the server URL and optionally hit the health endpoint."""
    # Optional: verify huggingface secret is available (for gated models)
    check_hf_secret.remote()
    url = serve.get_web_url()
    print(f"vLLM server URL: {url}")
    print(f"Set in your notebook or env: OPENAI_BASE_URL={url}/v1  MODEL={MODEL_NAME}")
    print("Then run the vllm_chat_and_crewai_demo notebook.")
