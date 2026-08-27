"""Speech-to-text providers."""

from __future__ import annotations

from app.config import Settings, get_settings
from app.transcription.assemblyai import AssemblyAISpeechProvider
from app.transcription.base import (
    AsyncSpeechProvider,
    SpeechProvider,
    Transcript,
    TranscriptionError,
    TranscriptSegment,
)
from app.transcription.fixture import FixtureSpeechProvider
from app.transcription.whisper import WhisperSpeechProvider

__all__ = [
    "AsyncSpeechProvider",
    "SpeechProvider",
    "Transcript",
    "TranscriptSegment",
    "TranscriptionError",
    "AssemblyAISpeechProvider",
    "FixtureSpeechProvider",
    "WhisperSpeechProvider",
    "get_speech_provider",
    "is_async_provider",
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
    if provider in ("assemblyai", "assembly"):
        return AssemblyAISpeechProvider(
            api_key=settings.assemblyai_api_key,
            language=settings.assemblyai_language,
            speakers_expected=settings.assemblyai_speakers_expected,
        )
    raise TranscriptionError(
        f"Unknown SPEECH_PROVIDER {settings.speech_provider!r}. "
        "Use 'faster_whisper', 'assemblyai' or 'fixture'."
    )


def is_async_provider(provider: object) -> bool:
    """True when the provider calls us back rather than making us wait."""
    return hasattr(provider, "submit") and hasattr(provider, "fetch")
