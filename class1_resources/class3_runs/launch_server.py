#!/usr/bin/env python3
"""
Launch Tokasaurus, vLLM, or SGLang using options from a file (or defaults).
Usage:
  python launch_server.py --engine vllm --model Qwen/Qwen2.5-1.5B-Instruct
  python launch_server.py --engine vllm --model Qwen/Qwen2.5-1.5B-Instruct --options-file vllm_serve_options.txt
  python launch_server.py --engine toka --model Qwen/Qwen2.5-1.5B-Instruct --options-file toka_serve_options.txt
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def read_options_file(path: Path, engine: str) -> list[str]:
    """Read options file: one flag per line for vllm/sglang, key=value for toka."""
    if not path.is_file():
        raise FileNotFoundError(f"Options file not found: {path}")
    args: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if engine == "toka":
                # toka uses key=value; allow multiple on one line or one per line
                for part in line.split():
                    if "=" in part:
                        args.append(part)
            else:
                # vllm/sglang: split on whitespace so "--flag value" becomes two args
                args.extend(line.split())
    return args


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch inference server (Tokasaurus, vLLM, SGLang) with optional options file."
    )
    parser.add_argument("--engine", choices=["toka", "vllm", "sglang"], required=True)
    parser.add_argument("--model", default="", help="Model name (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument(
        "--options-file",
        default="",
        help="Path to options file (e.g. vllm_serve_options.txt). Optional.",
    )
    parser.add_argument(
        "--cwd",
        default="",
        help="Working directory (default: script dir; for toka, use path to tokasaurus repo to ensure 'toka' is available).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cwd = Path(args.cwd) if args.cwd else root

    if args.options_file:
        opts_path = Path(args.options_file)
        if not opts_path.is_absolute():
            opts_path = (root / opts_path).resolve()
        file_args = read_options_file(opts_path, args.engine)
    else:
        file_args = []

    if args.engine == "toka":
        # toka model=... key=value ...
        cmd_args = ["toka"]
        if args.model:
            cmd_args.append(f"model={args.model}")
        cmd_args.extend(file_args)
        # Run from tokasaurus repo if we're in class3_runs and tokasaurus exists
        toka_dir = root / "tokasaurus"
        if toka_dir.is_dir():
            cwd = toka_dir
    elif args.engine == "vllm":
        # python -m vllm.entrypoints.openai.api_server --model MODEL [file args]
        cmd_args = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            args.model or os.environ.get("INFERENCE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        ]
        cmd_args.extend(file_args)
    else:
        # sglang
        cmd_args = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            args.model or os.environ.get("INFERENCE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        ]
        cmd_args.extend(file_args)

    print("Running:", " ".join(cmd_args))
    print("cwd:", cwd)
    return subprocess.run(cmd_args, cwd=cwd).returncode


if __name__ == "__main__":
    sys.exit(main())
