"""Transcript data structures and the speech-provider contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class TranscriptSegment:
    """One utterance with its place on the timeline."""

    index: int
    start: float
    end: float
    text: str
    #: Raw diarisation label from the provider ("SPEAKER_00", "A", ...), if any.
    speaker: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Transcript:
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = ""
    provider: str = ""
    model: str = ""
    duration: float = 0.0

    def __post_init__(self) -> None:
        if not self.duration and self.segments:
            self.duration = max(s.end for s in self.segments)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def speakers(self) -> list[str]:
        seen: list[str] = []
        for s in self.segments:
            if s.speaker and s.speaker not in seen:
                seen.append(s.speaker)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "language": self.language,
            "duration": self.duration,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transcript":
        segments = [
            TranscriptSegment(
                index=int(s.get("index", i)),
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                text=str(s.get("text", "")).strip(),
                speaker=str(s.get("speaker", "") or ""),
            )
            for i, s in enumerate(data.get("segments", []))
        ]
        return cls(
            segments=segments,
            language=str(data.get("language", "")),
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            duration=float(data.get("duration") or 0.0),
        )


class TranscriptionError(RuntimeError):
    """Raised when audio could not be turned into text."""


@runtime_checkable
class SpeechProvider(Protocol):
    """Anything that can turn an audio file into a timestamped transcript."""

    name: str

    def transcribe(self, audio_path: Path) -> Transcript: ...
