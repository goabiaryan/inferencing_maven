from __future__ import annotations

import json
from typing import Any, Iterator
from urllib import request

from .backend_interface import Backend


class LocalBackend(Backend):
    def __init__(self, url: str, timeout: int = 120) -> None:
        self._base_url = url.rstrip("/")
        self._timeout = timeout

    def generate(self, prompt: str, stream: bool = False) -> str | Iterator[dict[str, Any]]:
        if stream:
            return self._stream(prompt)
        return self._sync(prompt)

    def _sync(self, prompt: str) -> str:
        url = f"{self._base_url}/v1/chat/completions"
        payload = {"model": "default", "messages": [{"role": "user", "content": prompt}], "stream": False}
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=self._timeout) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        return _text_from_response(out)

    def _stream(self, prompt: str) -> Iterator[dict[str, Any]]:
        url = f"{self._base_url}/v1/chat/completions"
        payload = {"model": "default", "messages": [{"role": "user", "content": prompt}], "stream": True}
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=self._timeout) as resp:
            buffer = ""
            while True:
                chunk = resp.read(4096).decode("utf-8", errors="replace")
                if not chunk:
                    break
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line.startswith("data: "):
                        part = line[6:].strip()
                        if part == "[DONE]":
                            return
                        try:
                            obj = json.loads(part)
                            for c in obj.get("choices", []):
                                delta = c.get("delta", {}) or {}
                                content = (delta.get("content") or "").strip()
                                if content:
                                    yield {"content": content}
                        except json.JSONDecodeError:
                            pass


def _text_from_response(obj: dict) -> str:
    if "content" in obj:
        return str(obj["content"] or "").strip()
    for choice in obj.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message") or choice.get("delta") or {}
        text = (msg.get("content") or msg.get("text") or "").strip()
        if text:
            return text
    return ""
