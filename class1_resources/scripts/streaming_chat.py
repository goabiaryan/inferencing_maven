#!/usr/bin/env python3
"""Minimal example: streaming chat completion with RelayServe and optional X-Request-ID.

Run from class1_resources root (RelayServe must be running on 8080):
  python scripts/streaming_chat.py "Your prompt" "optional-request-id"
"""
import json
import os
import sys

from urllib.request import Request, urlopen

BASE = os.getenv("RELAYSERVE_EXAMPLE_BASE", "http://127.0.0.1:8080")


def stream_chat(prompt: str, request_id: str | None = None) -> None:
    url = f"{BASE.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": "relay-gguf",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    if request_id:
        headers["X-Request-ID"] = request_id
    req = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urlopen(req, timeout=60) as resp:
        print(f"Status: {resp.status}")
        print(f"X-Request-ID: {resp.headers.get('X-Request-ID', '(none)')}")
        print("Content-Type:", resp.headers.get("Content-Type"))
        print("--- stream ---")
        for line in resp:
            line = line.decode("utf-8").strip()
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    print("[DONE]")
                    break
                try:
                    chunk = json.loads(data)
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        if delta.get("content"):
                            print(delta["content"], end="", flush=True)
                except json.JSONDecodeError:
                    pass
        print("\n--- end ---")


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Say hello in one sentence."
    rid = sys.argv[2] if len(sys.argv) > 2 else None
    stream_chat(prompt, rid)
