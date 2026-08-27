"""Fixture speech provider.

Reads a transcript that sits next to the audio file rather than running a
model. Two forms are supported:

* ``<audio>.transcript.json`` - the full :class:`Transcript` dictionary,
  including speaker labels. Used by the test suite.
* ``<audio>.txt`` - a plain "Speaker: line" script. Timings are laid out
  evenly across the audio duration, weighted by line length.

This makes the whole pipeline runnable, demonstrable and testable with no
model download and no network.
"""

from __future__ import annotations

import json
import re
import wave
from pathlib import Path

from app.transcription.base import Transcript, TranscriptionError, TranscriptSegment

_SPEAKER_LINE = re.compile(r"^\s*(?:\[(?P<t>[\d:.]+)\]\s*)?(?P<speaker>[A-Za-z][\w .'-]{0,40}):\s*(?P<text>.+)$")


def _wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            return wav.getnframes() / rate if rate else None
    except Exception:
        return None


def _parse_timestamp(raw: str) -> float | None:
    parts = raw.split(":")
    try:
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + float(part)
        return seconds
    except ValueError:
        return None


def _script_to_transcript(script: str, duration: float | None) -> Transcript:
    rows: list[tuple[str, str, float | None]] = []
    for line in script.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SPEAKER_LINE.match(line)
        if match:
            stamp = match.group("t")
            rows.append(
                (
                    match.group("speaker").strip(),
                    match.group("text").strip(),
                    _parse_timestamp(stamp) if stamp else None,
                )
            )
        elif rows:
            speaker, text, at = rows[-1]
            rows[-1] = (speaker, f"{text} {line}", at)
        else:
            rows.append(("", line, None))

    if not rows:
        raise TranscriptionError("Fixture transcript is empty.")

    total_chars = sum(max(len(text), 1) for _, text, _ in rows)
    span = duration if duration and duration > 0 else float(total_chars) / 15.0

    segments: list[TranscriptSegment] = []
    cursor = 0.0
    for i, (speaker, text, at) in enumerate(rows):
        start = at if at is not None else cursor
        share = (max(len(text), 1) / total_chars) * span
        end = start + max(share, 0.5)
        # Keep the timeline monotonic even if explicit stamps disagree.
        if segments and start < segments[-1].end:
            start = segments[-1].end
            end = start + max(share, 0.5)
        segments.append(
            TranscriptSegment(index=i, start=round(start, 3), end=round(end, 3), text=text, speaker=speaker)
        )
        cursor = end

    return Transcript(segments=segments, language="en", provider="fixture", model="script")


class FixtureSpeechProvider:
    """Speech provider that reads a sidecar transcript instead of decoding audio."""

    name = "fixture"

    def transcribe(self, audio_path: Path) -> Transcript:
        audio_path = Path(audio_path)
        json_sidecar = audio_path.with_suffix(audio_path.suffix + ".transcript.json")
        if not json_sidecar.exists():
            json_sidecar = audio_path.with_suffix(".transcript.json")
        if json_sidecar.exists():
            transcript = Transcript.from_dict(json.loads(json_sidecar.read_text(encoding="utf-8")))
            transcript.provider = self.name
            return transcript

        for candidate in (
            audio_path.with_suffix(audio_path.suffix + ".txt"),
            audio_path.with_suffix(".txt"),
        ):
            if candidate.exists():
                duration = _wav_duration(audio_path) if audio_path.exists() else None
                return _script_to_transcript(candidate.read_text(encoding="utf-8"), duration)

        raise TranscriptionError(
            "SPEECH_PROVIDER=fixture needs a sidecar transcript next to the audio "
            f"({audio_path.name}.transcript.json or {audio_path.stem}.txt). "
            "Set SPEECH_PROVIDER=faster_whisper to transcribe real audio."
        )
