from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


def _relay_serve_root() -> Path:
    root = os.environ.get("RELAYSERVE_ROOT")
    if root:
        return Path(root).resolve()
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path(os.getcwd())


def _ensure_path() -> None:
    root = _relay_serve_root()
    import sys
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_config(config_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    path = config_path or _relay_serve_root() / "config.yaml"
    if not path.exists():
        return None
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def build_backends(config: dict[str, Any]) -> dict[str, Any]:
    _ensure_path()
    from backends.local_backend import LocalBackend
    from backends.modal_backend import ModalBackend
    from backends.vllm_backend import VllmBackend

    backends: dict[str, Any] = {}
    raw = config.get("backends") or {}
    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        t = (cfg.get("type") or "").strip().lower()
        url = (cfg.get("url") or "").strip()
        if not url:
            continue
        if t == "local":
            backends[name] = LocalBackend(url=url)
        elif t == "modal":
            backends[name] = ModalBackend(url=url)
        elif t == "vllm":
            backends[name] = VllmBackend(url=url)
    return backends


class Router:
    def __init__(self, config: Optional[dict[str, Any]] = None, config_path: Optional[Path] = None) -> None:
        self._config = config or load_config(config_path)
        self._backends: dict[str, Any] = {}
        self._default_key: Optional[str] = None
        if self._config:
            self._backends = build_backends(self._config)
            self._default_key = (self._config.get("default_backend") or "").strip() or None
            if self._default_key and self._default_key not in self._backends:
                self._default_key = next(iter(self._backends), None)

    def get_backend(self, model: Optional[str] = None):
        if not self._backends:
            return None
        key = (model or "").strip() if model else None
        if key and key in self._backends:
            return self._backends[key]
        if self._default_key:
            return self._backends.get(self._default_key)
        return next(iter(self._backends.values()), None)

    @property
    def has_backends(self) -> bool:
        return bool(self._backends)


def get_router(config_path: Optional[Path] = None) -> Router:
    return Router(config_path=config_path)
