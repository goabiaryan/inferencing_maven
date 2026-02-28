#!/usr/bin/env python3
"""
Step 2 for Serve_local_model: verify llama.cpp server and model paths are set and valid.

Usage:
  export LLAMA_SERVER_PATH=/path/to/llama.cpp/build/bin/llama-server
  export LLAMA_MODEL_PATH=/path/to/model.gguf
  cd /path/to/class1_resources && python scripts/check_llama_setup.py

Exits 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    ok = True

    server_path = os.getenv("LLAMA_SERVER_PATH", "").strip()
    if not server_path:
        print("LLAMA_SERVER_PATH is not set.", file=sys.stderr)
        print("  Set it to the path of the llama.cpp server binary (e.g. ./server).", file=sys.stderr)
        ok = False
    else:
        p = Path(server_path)
        if not p.exists():
            print(f"LLAMA_SERVER_PATH does not exist: {server_path}", file=sys.stderr)
            ok = False
        elif not p.is_file():
            print(f"LLAMA_SERVER_PATH is not a file: {server_path}", file=sys.stderr)
            ok = False
        elif not os.access(server_path, os.X_OK):
            print(f"LLAMA_SERVER_PATH is not executable: {server_path}", file=sys.stderr)
            ok = False
        else:
            print(f"LLAMA_SERVER_PATH OK: {server_path}")

    model_path = os.getenv("LLAMA_MODEL_PATH", "").strip()
    if not model_path:
        print("LLAMA_MODEL_PATH is not set.", file=sys.stderr)
        print("  Set it to the path of your GGUF model file (see Step 1 in Serve_local_model.md).", file=sys.stderr)
        ok = False
    else:
        p = Path(model_path)
        if not p.exists():
            print(f"LLAMA_MODEL_PATH does not exist: {model_path}", file=sys.stderr)
            ok = False
        elif not p.is_file():
            print(f"LLAMA_MODEL_PATH is not a file: {model_path}", file=sys.stderr)
            ok = False
        elif p.suffix.lower() != ".gguf":
            print(f"Warning: LLAMA_MODEL_PATH does not end with .gguf: {model_path}", file=sys.stderr)
            print(f"  (Path is still considered OK.)", file=sys.stderr)
            print(f"LLAMA_MODEL_PATH OK: {model_path}")
        else:
            print(f"LLAMA_MODEL_PATH OK: {model_path}")

    if ok:
        print("\nSetup looks good. Next: run scripts/spawn_backends.py (Step 3).")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
