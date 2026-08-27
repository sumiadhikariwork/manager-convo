"""The deferred pipeline: the shape that survives a host with no background.

Nothing here touches a real service. A stub transcription provider and a stub
bucket stand in, so what is under test is the sequencing: what runs when, what
is safe to run twice, and what happens when an invocation dies partway.
"""

from __future__ import annotations

import pytest

from app import pipeline
from app.config import Settings, get_settings
from app.db import SessionLocal
from app.models import AuditEvent, Conversation, ProcessingStatus
from app.templates import GROW_MONTHLY
from app.transcription.base import TranscriptionError
from app.transcription.fixture import _script_to_transcript


class StubHostedTranscriber:
    """A transcription service that fetches by URL and calls back."""

    name = "assemblyai"

    def __init__(self, script: str, fail_fetch: bool = False):
        self.script = script
        self.fail_fetch = fail_fetch
        self.submissions: list[tuple[str, str, str]] = []
        self.fetches: list[str] = []

    def submit(self, audio_url: str, webhook_url: str, webhook_secret: str = "") -> str:
        self.submissions.append((audio_url, webhook_url, webhook_secret))
        return f"job-{len(self.submissions)}"

    def fetch(self, job_id: str):
        self.fetches.append(job_id)
        if self.fail_fetch:
            raise TranscriptionError("the service could not read the audio")
        transcript = _script_to_transcript(self.script, 445.0)
        transcript.provider = "assemblyai"
        transcript.model = "universal"
        return transcript


class StubBucket:
    """Remote storage: reachable by URL, never present on this machine."""

    name = "s3"
    supports_direct_upload = True

    def playback_url(self, key, expires_in=3600):
        return f"https://bucket.example/{key}?sig=abc"

    def local_path(self, key):
        return None

    def exists(self, key):
        return True

    def size(self, key):
        return 1024

    def delete(self, key):
        pass


@pytest.fixture
def serverless(monkeypatch, script):
    """A deployment configured the way Vercel would be."""
    settings = Settings(
        data_dir=get_settings().data_dir,
        database_url=get_settings().database_url,
        speech_provider="assemblyai",
        assemblyai_api_key="fake",
        analysis_provider="heuristic",
        storage_backend="s3",
        storage_bucket="recordings",
        job_runner="deferred",
        public_base_url="https://records.example",
        webhook_secret="s3cret",
    )
    transcriber = StubHostedTranscriber(script)
    monkeypatch.setattr(pipeline, "get_speech_provider", lambda *_: transcriber)
    monkeypatch.setattr(pipeline, "get_storage", lambda *_: StubBucket())
    return settings, transcriber


def _conversation() -> str:
    with SessionLocal() as session:
        conversation = Conversation(
            title="Remote check-in",
            template_id=GROW_MONTHLY.id,
            manager_name="Priya",
            report_name="Sam",
            consent_confirmed=True,
            audio_filename="a.wav",
            audio_key="audio/a.wav",
            audio_mime="audio/wav",
        )
        session.add(conversation)
        session.commit()
        return conversation.id


def _load(conversation_id) -> Conversation:
    with SessionLocal() as session:
        return session.get(Conversation, conversation_id)


def _actions(conversation_id) -> list[str]:
    from sqlalchemy import select

    with SessionLocal() as session:
        return [
            e.action
            for e in session.scalars(
                select(AuditEvent)
                .where(AuditEvent.conversation_id == conversation_id)
                .order_by(AuditEvent.at)
            )
        ]


# -- handoff ---------------------------------------------------------------

def test_upload_hands_off_and_returns_immediately(serverless):
    """The request must not wait on transcription - it has seconds, not minutes."""
    settings, transcriber = serverless
    conversation_id = _conversation()

    pipeline.submit(conversation_id, settings)

    conversation = _load(conversation_id)
    assert conversation.status is ProcessingStatus.TRANSCRIBING
    assert conversation.transcription_job_id == "job-1"
    assert len(transcriber.submissions) == 1
    assert transcriber.fetches == [], "nothing should have been waited for"


def test_the_service_is_given_a_url_and_somewhere_to_call_back(serverless):
    settings, transcriber = serverless
    pipeline.submit(_conversation(), settings)

    audio_url, webhook_url, secret = transcriber.submissions[0]
    assert audio_url.startswith("https://bucket.example/audio/a.wav")
    assert webhook_url == "https://records.example/api/webhooks/transcription"
    assert secret == "s3cret"


def test_without_a_public_address_the_handoff_fails_loudly(serverless):
    """A callback that cannot reach us would strand the conversation silently."""
    settings, _ = serverless
    settings.public_base_url = ""
    conversation_id = _conversation()

    pipeline.submit(conversation_id, settings)

    conversation = _load(conversation_id)
    assert conversation.status is ProcessingStatus.FAILED
    assert "PUBLIC_BASE_URL" in conversation.error


def test_a_local_model_cannot_be_pointed_at_remote_audio(monkeypatch, serverless):
    settings, _ = serverless

    class LocalOnly:
        name = "faster_whisper"

        def transcribe(self, path):  # pragma: no cover - never reached
            raise AssertionError("should not be called")

    monkeypatch.setattr(pipeline, "get_speech_provider", lambda *_: LocalOnly())
    conversation_id = _conversation()
    pipeline.submit(conversation_id, settings)

    conversation = _load(conversation_id)
    assert conversation.status is ProcessingStatus.FAILED
    assert "not readable on this machine" in conversation.error


# -- the callback ----------------------------------------------------------

def test_the_webhook_carries_the_conversation_all_the_way_to_ready(serverless):
    """It is the only invocation guaranteed to happen, so it does the rest."""
    settings, transcriber = serverless
    conversation_id = _conversation()
    pipeline.submit(conversation_id, settings)

    pipeline.receive_transcription("job-1", settings)

    conversation = _load(conversation_id)
    assert conversation.status is ProcessingStatus.READY
    assert conversation.transcript_json is not None
    assert conversation.analysis_json["sections"]
    assert _actions(conversation_id) == [
        "transcription_submitted", "transcribed", "aligned", "drafted"
    ]


def test_a_duplicate_callback_changes_nothing(serverless):
    """Delivery is at-least-once, so this has to be safe."""
    settings, transcriber = serverless
    conversation_id = _conversation()
    pipeline.submit(conversation_id, settings)
    pipeline.receive_transcription("job-1", settings)

    before = _actions(conversation_id)
    pipeline.receive_transcription("job-1", settings)

    assert _actions(conversation_id) == before
    assert len(transcriber.fetches) == 1, "the transcript must not be re-fetched"
    assert _load(conversation_id).status is ProcessingStatus.READY


def test_a_callback_for_an_unknown_job_is_ignored(serverless):
    settings, _ = serverless
    assert pipeline.receive_transcription("job-that-never-was", settings) is None


def test_a_failure_at_the_service_lands_on_the_record(monkeypatch, serverless, script):
    settings, _ = serverless
    failing = StubHostedTranscriber(script, fail_fetch=True)
    monkeypatch.setattr(pipeline, "get_speech_provider", lambda *_: failing)

    conversation_id = _conversation()
    pipeline.submit(conversation_id, settings)
    pipeline.receive_transcription("job-1", settings)

    conversation = _load(conversation_id)
    assert conversation.status is ProcessingStatus.FAILED
    assert "could not read the audio" in conversation.error
    assert "failed" in _actions(conversation_id)


# -- resumption ------------------------------------------------------------

def test_waiting_on_the_service_is_not_treated_as_stalled(serverless):
    """Otherwise every status poll would resubmit the job."""
    settings, transcriber = serverless
    conversation_id = _conversation()
    pipeline.submit(conversation_id, settings)

    for _ in range(3):
        assert pipeline.resume(conversation_id, settings) is ProcessingStatus.TRANSCRIBING
    assert len(transcriber.submissions) == 1


def test_an_invocation_that_died_after_transcribing_is_picked_back_up(serverless):
    """The instance is frozen mid-run; the next request has to finish the job."""
    settings, transcriber = serverless
    conversation_id = _conversation()
    pipeline.submit(conversation_id, settings)

    # Store the transcript, then simulate the process vanishing before analysis.
    pipeline._store_transcript(conversation_id, transcriber.fetch("job-1"), "assemblyai")
    assert _load(conversation_id).status is ProcessingStatus.ALIGNING

    assert pipeline.resume(conversation_id, settings) is ProcessingStatus.READY
    assert _load(conversation_id).analysis_json["sections"]


def test_a_run_that_died_midway_through_drafting_restarts_that_stage(serverless):
    settings, transcriber = serverless
    conversation_id = _conversation()
    pipeline.submit(conversation_id, settings)
    pipeline._store_transcript(conversation_id, transcriber.fetch("job-1"), "assemblyai")

    with SessionLocal() as session:
        conversation = session.get(Conversation, conversation_id)
        conversation.status = ProcessingStatus.DRAFTING
        conversation.stage_started_at = None
        session.commit()

    assert pipeline.resume(conversation_id, settings) is ProcessingStatus.READY


def test_resume_leaves_a_finished_conversation_alone(serverless):
    settings, _ = serverless
    conversation_id = _conversation()
    pipeline.submit(conversation_id, settings)
    pipeline.receive_transcription("job-1", settings)

    before = _load(conversation_id).updated_at
    assert pipeline.resume(conversation_id, settings) is ProcessingStatus.READY
    assert _load(conversation_id).updated_at == before


def test_a_stage_in_progress_is_left_running(serverless):
    """Two overlapping requests must not both start the same work."""
    settings, transcriber = serverless
    conversation_id = _conversation()
    pipeline.submit(conversation_id, settings)
    pipeline._store_transcript(conversation_id, transcriber.fetch("job-1"), "assemblyai")

    with SessionLocal() as session:
        conversation = session.get(Conversation, conversation_id)
        conversation.status = ProcessingStatus.DRAFTING  # someone else is on it
        session.commit()

    # Fresh stage_started_at means "started moments ago", so resume backs off.
    assert pipeline.resume(conversation_id, settings) is ProcessingStatus.DRAFTING


def test_only_one_caller_can_claim_a_stage(serverless):
    settings, _ = serverless
    conversation_id = _conversation()

    with SessionLocal() as session:
        first = pipeline._claim(
            session, conversation_id, ProcessingStatus.UPLOADED,
            ProcessingStatus.TRANSCRIBING, "mine",
        )
        second = pipeline._claim(
            session, conversation_id, ProcessingStatus.UPLOADED,
            ProcessingStatus.TRANSCRIBING, "mine too",
        )
    assert first is True
    assert second is False


def test_restart_sends_a_finished_conversation_back_through(serverless):
    settings, transcriber = serverless
    conversation_id = _conversation()
    pipeline.submit(conversation_id, settings)
    pipeline.receive_transcription("job-1", settings)

    pipeline.restart(conversation_id, settings)
    conversation = _load(conversation_id)
    assert conversation.status is ProcessingStatus.UPLOADED
    assert conversation.transcription_job_id == ""

    pipeline.submit(conversation_id, settings)
    assert _load(conversation_id).status is ProcessingStatus.TRANSCRIBING
    assert len(transcriber.submissions) == 2
