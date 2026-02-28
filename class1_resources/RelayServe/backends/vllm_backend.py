from __future__ import annotations

import json
from typing import Any, Iterator
from urllib import request

from .backend_interface import Backend


class VllmBackend(Backend):
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
        choices = out.get("choices") or []
        if not choices:
            return ""
        msg = (choices[0] or {}).get("message") or {}
        return str(msg.get("content") or "").strip()

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
                            for c in obj.get("choices") or []:
                                content = (c.get("delta") or {}).get("content") or ""
                                if content:
                                    yield {"content": content}
                        except json.JSONDecodeError:
                            pass
