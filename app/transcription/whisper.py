"""Local speech-to-text backed by faster-whisper.

Keeps audio on the machine that runs the app, which is usually the deciding
factor for recordings of one-to-one conversations. The model is loaded lazily
and cached per process - the first transcription pays the load cost.
"""

from __future__ import annotations

import threading
from pathlib import Path

from app.transcription.base import Transcript, TranscriptionError, TranscriptSegment

_model_lock = threading.Lock()
_model_cache: dict[tuple[str, str, str], object] = {}


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctranslate2  # type: ignore

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _load_model(model_size: str, device: str, compute_type: str):
    key = (model_size, device, compute_type)
    with _model_lock:
        if key not in _model_cache:
            try:
                from faster_whisper import WhisperModel  # type: ignore
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise TranscriptionError(
                    "faster-whisper is not installed. Run `pip install -r requirements-asr.txt`, "
                    "or set SPEECH_PROVIDER=fixture to work from a sidecar transcript."
                ) from exc
            _model_cache[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
        return _model_cache[key]


class WhisperSpeechProvider:
    """faster-whisper transcription with per-segment timestamps."""

    name = "faster_whisper"

    def __init__(self, model_size: str = "small", device: str = "auto", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = _resolve_device(device)
        self.compute_type = compute_type

    def transcribe(self, audio_path: Path) -> Transcript:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")

        model = _load_model(self.model_size, self.device, self.compute_type)
        try:
            raw_segments, info = model.transcribe(  # type: ignore[attr-defined]
                str(audio_path),
                vad_filter=True,
                beam_size=5,
                # Whisper will happily merge a long exchange into one block;
                # punctuation-based splitting keeps utterances addressable.
                condition_on_previous_text=False,
            )
        except Exception as exc:  # pragma: no cover - depends on optional extra
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

        segments = [
            TranscriptSegment(
                index=i,
                start=round(float(s.start), 3),
                end=round(float(s.end), 3),
                text=(s.text or "").strip(),
            )
            for i, s in enumerate(raw_segments)
            if (s.text or "").strip()
        ]
        if not segments:
            raise TranscriptionError("No speech was found in the audio.")

        return Transcript(
            segments=segments,
            language=getattr(info, "language", "") or "",
            provider=self.name,
            model=self.model_size,
            duration=float(getattr(info, "duration", 0.0) or segments[-1].end),
        )
