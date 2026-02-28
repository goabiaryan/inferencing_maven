#!/usr/bin/env python3
"""
Step 1 for Serve_local_model: download a small GGUF model into the repo.

Always saves to class1_resources/models/ (repo root). Run from class1_resources:
  pip install huggingface_hub
  cd /path/to/class1_resources && python scripts/download_model.py
  export LLAMA_MODEL_PATH="<path script prints>"

Uses TinyLlama 1.1B Chat Q4_K_M by default (small, fast for demos).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
DEFAULT_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Always use repo root / models (class1_resources/models)
_REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = _REPO_ROOT / "models"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a small GGUF model into the repo (saves to repo/models/).")
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO, help="Hugging Face repo")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="GGUF filename in repo")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("This script needs: pip install huggingface_hub\nAlternatively download a GGUF manually and set LLAMA_MODEL_PATH.", file=sys.stderr)
        return 1

    out_dir = Path(MODELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to repo models/: {out_dir}")
    print(f"Downloading {args.filename} from {args.repo} ...")
    try:
        path = hf_hub_download(
            repo_id=args.repo,
            filename=args.filename,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1

    print(f"Downloaded to: {path}")
    print("\nNext (Step 3 in Serve_local_model.md):")
    print(f'  export LLAMA_MODEL_PATH="{path}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
