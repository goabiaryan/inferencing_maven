import modal

# Model baked into the image (must match MODEL in vllm_chat_and_crewai_demo.ipynb)
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def _download_model() -> None:
    """Pre-download model into the image so the notebook loads instantly."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID)


# Pre-built image: vLLM + model. First build can take 15–20 min; then cached.
# Runtime base = smaller/faster pull than devel.
vllm_notebook_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-runtime-ubuntu22.04", add_python="3.11")
    .pip_install("typing_extensions>=4.4.0", "vllm", "huggingface-hub")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(_download_model)  # Qwen2.5-1.5B is non-gated; no HF_TOKEN needed
)

app = modal.App("vllm-notebook")


@app.function(image=vllm_notebook_image)
def _notebook_image():
    """Dummy function so the image can be deployed and attached to notebooks."""
    pass
