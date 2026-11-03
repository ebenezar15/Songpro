"""Common engine interface: models can be swapped without changing the app."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EngineUnavailable(RuntimeError):
    """Raised with a user-facing explanation (what is missing, size, licence, how to install)."""


class Cancelled(RuntimeError):
    pass


class VocalEngine(ABC):
    id: str = "base"
    name: str = "Base"
    license_id: str = ""
    languages: tuple = ()
    input_types: tuple = ("hum", "sung", "voice_sample")
    native_sr: int = 48000
    deterministic: bool = True

    def available(self) -> tuple[bool, str]:
        return True, "ready"

    def info(self) -> dict:
        ok, msg = self.available()
        return {"id": self.id, "name": self.name, "license_id": self.license_id, "languages": list(self.languages),
                "input_types": list(self.input_types), "available": ok, "status": msg,
                "native_sr": self.native_sr, "deterministic": self.deterministic, "version": getattr(self, "version", "0")}

    @abstractmethod
    def render(self, plan: dict, seed: int, progress=lambda f, m: None, cancel=lambda: False) -> tuple[np.ndarray, int]:
        """Return (mono float32 audio, sample_rate). The audio starts at t=0 of the song."""
