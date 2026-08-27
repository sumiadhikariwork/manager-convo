"""Speech-to-text providers."""

from __future__ import annotations

from app.config import Settings, get_settings
from app.transcription.base import (
    SpeechProvider,
    Transcript,
    TranscriptionError,
    TranscriptSegment,
)
from app.transcription.fixture import FixtureSpeechProvider
from app.transcription.whisper import WhisperSpeechProvider

__all__ = [
    "SpeechProvider",
    "Transcript",
    "TranscriptSegment",
    "TranscriptionError",
    "FixtureSpeechProvider",
    "WhisperSpeechProvider",
    "get_speech_provider",
]


def get_speech_provider(settings: Settings | None = None) -> SpeechProvider:
    settings = settings or get_settings()
    provider = settings.speech_provider.lower()
    if provider in ("fixture", "mock", "script"):
        return FixtureSpeechProvider()
    if provider in ("faster_whisper", "whisper"):
        return WhisperSpeechProvider(
            model_size=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    raise TranscriptionError(
        f"Unknown SPEECH_PROVIDER {settings.speech_provider!r}. Use 'faster_whisper' or 'fixture'."
    )
