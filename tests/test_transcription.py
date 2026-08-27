"""The speech layer: transcript structures and the fixture provider."""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.transcription import get_speech_provider
from app.transcription.base import Transcript, TranscriptionError, TranscriptSegment
from app.transcription.fixture import FixtureSpeechProvider, _script_to_transcript


def test_script_lines_become_timed_segments(script):
    transcript = _script_to_transcript(script, 445.0)
    assert len(transcript.segments) > 20
    assert transcript.segments[0].speaker == "Priya"
    assert transcript.segments[1].speaker == "Sam"
    assert "how has the month been" in transcript.segments[0].text.lower()


def test_the_timeline_never_runs_backwards(script):
    transcript = _script_to_transcript(script, 445.0)
    for previous, current in zip(transcript.segments, transcript.segments[1:]):
        assert current.start >= previous.end - 1e-6
        assert current.end > current.start


def test_comments_and_blank_lines_are_skipped(script):
    transcript = _script_to_transcript(script, 445.0)
    assert not any(s.text.startswith("#") for s in transcript.segments)


def test_a_continuation_line_joins_the_utterance_above_it():
    transcript = _script_to_transcript(
        "Priya: The first part\nand the rest of the same sentence.\nSam: Understood.", 20.0
    )
    assert len(transcript.segments) == 2
    assert transcript.segments[0].text == "The first part and the rest of the same sentence."


def test_an_empty_script_is_an_error():
    with pytest.raises(TranscriptionError):
        _script_to_transcript("# only a comment\n\n", 10.0)


def test_a_json_sidecar_is_read_verbatim(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    payload = {
        "language": "en",
        "duration": 12.5,
        "segments": [
            {"index": 0, "start": 0.0, "end": 6.0, "text": "First.", "speaker": "A"},
            {"index": 1, "start": 6.0, "end": 12.5, "text": "Second.", "speaker": "B"},
        ],
    }
    audio.with_suffix(".wav.transcript.json").write_text(json.dumps(payload))

    transcript = FixtureSpeechProvider().transcribe(audio)
    assert transcript.provider == "fixture"
    assert transcript.duration == 12.5
    assert [s.text for s in transcript.segments] == ["First.", "Second."]
    assert transcript.speakers == ["A", "B"]


def test_a_missing_sidecar_says_how_to_fix_it(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    with pytest.raises(TranscriptionError, match="faster_whisper"):
        FixtureSpeechProvider().transcribe(audio)


def test_transcript_round_trips_through_a_dict(script):
    original = _script_to_transcript(script, 445.0)
    restored = Transcript.from_dict(original.to_dict())
    assert len(restored.segments) == len(original.segments)
    assert restored.segments[3].text == original.segments[3].text
    assert restored.duration == original.duration


def test_duration_defaults_to_the_last_segment_end():
    transcript = Transcript(segments=[TranscriptSegment(index=0, start=0.0, end=9.5, text="Hi.")])
    assert transcript.duration == 9.5


def test_provider_selection_follows_the_setting():
    assert get_speech_provider(Settings(speech_provider="fixture")).name == "fixture"
    assert get_speech_provider(Settings(speech_provider="faster_whisper")).name == "faster_whisper"
    with pytest.raises(TranscriptionError, match="Unknown SPEECH_PROVIDER"):
        get_speech_provider(Settings(speech_provider="telepathy"))
