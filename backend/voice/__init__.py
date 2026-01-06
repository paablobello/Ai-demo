"""
Voice Pipeline Module
Reutilizado de VoicePipeline con optimizaciones para demos.
"""

from .vad import SileroVAD
from .stt import DeepgramSTT
from .tts import CartesiaTTS
from .tts_elevenlabs import ElevenLabsTTS
from .turn_detector import TurnDetector
from .eou_detector import get_eou_detector
from .backchannel import get_backchannel_detector

__all__ = [
    "SileroVAD",
    "DeepgramSTT",
    "CartesiaTTS",
    "ElevenLabsTTS",
    "TurnDetector",
    "get_eou_detector",
    "get_backchannel_detector",
]
