"""The AssemblyAI provider, against a stubbed HTTP client."""

from __future__ import annotations

import pytest

from app.transcription.assemblyai import WEBHOOK_HEADER, AssemblyAISpeechProvider
from app.transcription.base import TranscriptionError


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text or str(self._body)

    def json(self):
        return self._body


class FakeClient:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[str] = []

    def post(self, url, headers=None, json=None):
        self.posts.append((url, json or {}))
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def get(self, url, headers=None):
        self.gets.append(url)
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def provider(*responses, **kwargs):
    client = FakeClient(*responses)
    return AssemblyAISpeechProvider(api_key="key", client=client, **kwargs), client


COMPLETED = {
    "status": "completed",
    "language_code": "en_us",
    "audio_duration": 430.5,
    "utterances": [
        {"speaker": "A", "start": 2000, "end": 11250, "text": "How has the month been?"},
        {"speaker": "B", "start": 11250, "end": 26000, "text": "Honestly? Mixed."},
        {"speaker": "A", "start": 26000, "end": 34000, "text": "  Say more about that.  "},
    ],
}


# -- submit ----------------------------------------------------------------

def test_submit_returns_the_job_id():
    speech, client = provider(FakeResponse(body={"id": "abc123", "status": "queued"}))
    assert speech.submit("https://bucket/a.wav", "https://app/hook", "s3cret") == "abc123"


def test_submit_asks_for_diarisation_and_a_callback():
    speech, client = provider(FakeResponse(body={"id": "abc123"}))
    speech.submit("https://bucket/a.wav", "https://app/hook", "s3cret")

    url, body = client.posts[0]
    assert url.endswith("/v2/transcript")
    assert body["audio_url"] == "https://bucket/a.wav"
    assert body["speaker_labels"] is True
    assert body["webhook_url"] == "https://app/hook"
    assert body["webhook_auth_header_name"] == WEBHOOK_HEADER
    assert body["webhook_auth_header_value"] == "s3cret"


def test_two_speakers_are_declared_so_one_voice_is_not_split():
    speech, client = provider(FakeResponse(body={"id": "x"}))
    speech.submit("https://bucket/a.wav", "https://app/hook")
    _, body = client.posts[0]
    assert body["speaker_options"] == {"min_speakers_expected": 2, "max_speakers_expected": 2}


def test_language_is_detected_unless_one_is_configured():
    speech, client = provider(FakeResponse(body={"id": "x"}))
    speech.submit("https://bucket/a.wav", "https://app/hook")
    assert client.posts[0][1]["language_detection"] is True

    speech, client = provider(FakeResponse(body={"id": "y"}), language="en_uk")
    speech.submit("https://bucket/a.wav", "https://app/hook")
    assert client.posts[0][1]["language_code"] == "en_uk"
    assert "language_detection" not in client.posts[0][1]


def test_no_secret_means_no_webhook_auth_headers():
    speech, client = provider(FakeResponse(body={"id": "x"}))
    speech.submit("https://bucket/a.wav", "https://app/hook", "")
    assert "webhook_auth_header_name" not in client.posts[0][1]


def test_a_rejected_submission_explains_itself():
    speech, _ = provider(FakeResponse(status_code=401, text="invalid api key"))
    with pytest.raises(TranscriptionError, match="401"):
        speech.submit("https://bucket/a.wav", "https://app/hook")


def test_a_response_without_an_id_is_an_error():
    speech, _ = provider(FakeResponse(body={"status": "queued"}))
    with pytest.raises(TranscriptionError, match="no job id"):
        speech.submit("https://bucket/a.wav", "https://app/hook")


def test_a_missing_api_key_is_caught_at_construction():
    with pytest.raises(TranscriptionError, match="ASSEMBLYAI_API_KEY"):
        AssemblyAISpeechProvider(api_key="")


# -- fetch -----------------------------------------------------------------

def test_utterances_become_segments_with_seconds_not_milliseconds():
    """AssemblyAI reports milliseconds; everything downstream assumes seconds."""
    speech, _ = provider(FakeResponse(body=COMPLETED))
    transcript = speech.fetch("abc123")

    assert [s.start for s in transcript.segments] == [2.0, 11.25, 26.0]
    assert [s.end for s in transcript.segments] == [11.25, 26.0, 34.0]
    assert transcript.duration == 430.5


def test_diarisation_labels_survive_as_speakers():
    speech, _ = provider(FakeResponse(body=COMPLETED))
    transcript = speech.fetch("abc123")
    assert [s.speaker for s in transcript.segments] == ["A", "B", "A"]
    assert transcript.speakers == ["A", "B"]


def test_segments_are_indexed_and_trimmed():
    speech, _ = provider(FakeResponse(body=COMPLETED))
    transcript = speech.fetch("abc123")
    assert [s.index for s in transcript.segments] == [0, 1, 2]
    assert transcript.segments[2].text == "Say more about that."


def test_a_transcript_with_no_utterances_falls_back_to_the_whole_text():
    speech, _ = provider(FakeResponse(body={
        "status": "completed", "utterances": [], "text": "One unbroken block.", "audio_duration": 12.0
    }))
    transcript = speech.fetch("abc123")
    assert len(transcript.segments) == 1
    assert transcript.segments[0].text == "One unbroken block."
    assert transcript.segments[0].end == 12.0


def test_silence_is_reported_rather_than_returned_as_an_empty_transcript():
    speech, _ = provider(FakeResponse(body={"status": "completed", "utterances": [], "text": ""}))
    with pytest.raises(TranscriptionError, match="no speech"):
        speech.fetch("abc123")


def test_a_failure_at_the_service_carries_its_reason():
    speech, _ = provider(FakeResponse(body={"status": "error", "error": "Download failed"}))
    with pytest.raises(TranscriptionError, match="Download failed"):
        speech.fetch("abc123")


def test_fetching_too_early_says_so():
    speech, _ = provider(FakeResponse(body={"status": "processing"}))
    with pytest.raises(TranscriptionError, match="not ready yet"):
        speech.fetch("abc123")


def test_a_network_failure_becomes_a_transcription_error():
    speech, _ = provider(RuntimeError("connection reset"))
    with pytest.raises(TranscriptionError, match="Could not reach AssemblyAI"):
        speech.fetch("abc123")
