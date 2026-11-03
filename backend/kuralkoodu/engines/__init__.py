from .base import Cancelled, EngineUnavailable, VocalEngine
from .dsp_convert import DspConvertEngine
from .formant import FormantSingerEngine
from .mock import MockEngine
from .neural import SeedVCEngine, SoulXSingerEngine

ENGINES = {e.id: e for e in (FormantSingerEngine(), MockEngine(), DspConvertEngine(), SoulXSingerEngine(), SeedVCEngine())}


def get_engine(engine_id: str) -> VocalEngine:
    if engine_id not in ENGINES:
        raise EngineUnavailable(f"Unknown engine '{engine_id}'")
    return ENGINES[engine_id]
