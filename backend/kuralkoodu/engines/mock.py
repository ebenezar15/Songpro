"""MockEngine: instant, dependency-free test output (band-limited saw on the plan pitch)."""
from __future__ import annotations

import numpy as np

from .base import Cancelled, VocalEngine


class MockEngine(VocalEngine):
    id = "mock"
    name = "Mock (test tones)"
    license_id = "kuralkoodu"
    languages = ("any",)
    native_sr = 48000
    version = "mock-1"

    def render(self, plan, seed, progress=lambda f, m: None, cancel=lambda: False):
        sr = self.native_sr
        fps = plan["fps"]
        n = int(plan["duration"] * sr)
        tt = np.arange(n) / sr * fps
        f0 = np.interp(tt, np.arange(plan["n_frames"]), plan["f0_hz"])
        amp = np.interp(tt, np.arange(plan["n_frames"]), plan["amp"])
        ph = 2 * np.pi * np.cumsum(f0) / sr
        y = np.zeros(n)
        for k in range(1, 9):
            if cancel():
                raise Cancelled()
            y += np.where(k * f0 < sr / 2, np.sin(k * ph) / k, 0)
        progress(1.0, "mock render done")
        return (0.25 * y * amp).astype(np.float32), sr
