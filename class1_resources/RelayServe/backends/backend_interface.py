from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator


class Backend(ABC):
    @abstractmethod
    def generate(self, prompt: str, stream: bool = False) -> str | Iterator[dict[str, Any]]:
        raise NotImplementedError
