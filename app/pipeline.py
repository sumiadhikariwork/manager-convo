"""The ingest pipeline: audio in, filled-in feedback record out.

    upload → transcribe → align to the agenda → draft every field → ready

The pipeline is a **stage machine keyed on the conversation's status**, not a
straight-line function. Each stage reads what it needs from the database, does
one thing, and commits the next status. That has two consequences worth the
indirection:

* Nothing is held in memory between stages, so a host that freezes or discards
  the process after a response loses no work. A stalled conversation is resumed
  by calling ``advance`` again - from a webhook, a cron, or the browser that is
  already polling for status.
* Every stage is idempotent. Running one twice recomputes and rewrites; it
  never double-appends, and it never overwrites something the manager edited.

Two runners sit on top of the same stages:

* ``thread``   - a background thread walks every stage to completion. Right for
                 a long-lived server, where waiting is free.
* ``deferred`` - one stage per request. The upload hands off to a transcription
                 service and returns; the service's webhook drives the rest.
                 The only shape that works where a request is short-lived.
"""

from __future__ import annotations

import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.analysis import AnalysisError, HeuristicAnalyst, compute_metrics, get_analyst
from app.analysis.types import DraftResult, StructureResult
from app.config import Settings, get_settings
from app.db import session_scope
from app.models import (
    Conversation,
    FieldSource,
    FormField,
    ProcessingStatus,
    Segment,
    SpeakerRole,
    record_event,
    utcnow,
)
from app.storage import Storage, get_storage
from app.templates import ConversationTemplate, get_template
from app.transcription import TranscriptionError, get_speech_provider, is_async_provider
from app.transcription.base import Transcript, TranscriptSegment
from app.util import truncate

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline")
_inflight: set[str] = set()
_inflight_lock = threading.Lock()

#: A recording is handed to the transcription service as a signed URL. It has
#: to outlive the queue there, which can be minutes on a busy day.
AUDIO_URL_TTL = 6 * 3600

#: Past this, a stage someone has claimed is treated as abandoned and may be
#: restarted. Only in-progress states are subject to it.
STALL_SECONDS = 900

#: Nothing holds these: no worker has claimed them, so a caller may start work
#: immediately. On a serverless host that is the normal state between requests.
READY_TO_RUN = (ProcessingStatus.UPLOADED, ProcessingStatus.ALIGNING)

#: Somebody claimed this and may still be working. Left alone until it stalls.
IN_PROGRESS = (ProcessingStatus.DRAFTING,)

#: Waiting on something outside this system. Never resumed - the callback comes
#: when it comes, and resubmitting would duplicate the job.
WAITING_ELSEWHERE = (ProcessingStatus.TRANSCRIBING,)


class PipelineError(RuntimeError):
    """Raised when a stage cannot proceed."""


# --------------------------------------------------------------------------
# Status bookkeeping
# --------------------------------------------------------------------------

def _set_status(
    session: Session, conversation: Conversation, status: ProcessingStatus, detail: str = ""
) -> None:
    conversation.status = status
    conversation.status_detail = detail
    conversation.stage_started_at = utcnow()
    conversation.updated_at = utcnow()
    session.add(conversation)
    session.commit()


def _claim(session: Session, conversation_id: str, expected: ProcessingStatus,
           claimed: ProcessingStatus, detail: str) -> bool:
    """Take ownership of a stage, atomically.

    A conditional UPDATE is the whole locking story: whichever caller flips the
    status first does the work, and any other caller sees zero rows changed and
    steps aside. Works the same on SQLite and Postgres, and needs no lock table.
    """
    result = session.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id, Conversation.status == expected)
        .values(status=claimed, status_detail=detail, stage_started_at=utcnow(), updated_at=utcnow())
    )
    session.commit()
    return bool(result.rowcount)


def _fail(conversation_id: str, exc: BaseException) -> None:
    logger.exception("Pipeline failed for %s", conversation_id)
    with session_scope() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            return
        conversation.error = f"{type(exc).__name__}: {exc}"
        record_event(
            session, conversation_id, "failed",
            error=str(exc), trace=truncate(traceback.format_exc(), 2000),
        )
        _set_status(session, conversation, ProcessingStatus.FAILED, truncate(str(exc), 300))


# --------------------------------------------------------------------------
# Persistence helpers
# --------------------------------------------------------------------------

def _store_segments(
    session: Session, conversation: Conversation, transcript: Transcript, structure: StructureResult
) -> None:
    session.execute(delete(Segment).where(Segment.conversation_id == conversation.id))
    for i, seg in enumerate(transcript.segments):
        role = structure.speaker_roles[i] if i < len(structure.speaker_roles) else "unknown"
        session.add(
            Segment(
                conversation_id=conversation.id,
                index=i,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker_label=seg.speaker or "",
                speaker_role=(
                    SpeakerRole(role) if role in SpeakerRole._value2member_map_ else SpeakerRole.UNKNOWN
                ),
                speaker_confidence=(
                    structure.speaker_confidence[i] if i < len(structure.speaker_confidence) else 0.0
                ),
                section_id=structure.section_ids[i] if i < len(structure.section_ids) else None,
                section_confidence=(
                    structure.section_confidence[i] if i < len(structure.section_confidence) else 0.0
                ),
            )
        )
    session.commit()


def _build_evidence(
    segments: Sequence[TranscriptSegment], indices: Sequence[int]
) -> list[dict[str, object]]:
    """Resolve cited segment indices to real quotes from the stored transcript.

    Quote text always comes from here, never from a model, so a citation on the
    record cannot be a paraphrase of something that was not said.
    """
    evidence = []
    for index in indices:
        if 0 <= index < len(segments):
            seg = segments[index]
            evidence.append(
                {
                    "segment_index": index,
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "quote": truncate(seg.text, 320),
                }
            )
    return evidence


def _store_fields(
    session: Session,
    conversation: Conversation,
    template: ConversationTemplate,
    transcript: Transcript,
    draft: DraftResult,
) -> None:
    """Write drafted values, preserving anything the manager has already edited."""
    existing = {
        (f.section_id, f.field_id): f
        for f in session.scalars(
            select(FormField).where(FormField.conversation_id == conversation.id)
        )
    }
    drafted = {(d.section_id, d.field_id): d for d in draft.fields}
    source = FieldSource.CLAUDE if draft.method == "claude" else FieldSource.HEURISTIC

    order = 0
    for section in template.sections:
        for spec in section.fields:
            key = (section.id, spec.id)
            item = drafted.get(key)
            value = item.value if item else ([] if spec.kind in ("list", "actions") else "")
            evidence = _build_evidence(transcript.segments, item.evidence if item else [])
            confidence = item.confidence if item else 0.0

            field = existing.get(key)
            if field is None:
                field = FormField(conversation_id=conversation.id, section_id=section.id, field_id=spec.id)
                session.add(field)

            field.order = order
            field.kind = spec.kind
            field.draft_value = value
            field.confidence = confidence
            field.evidence_json = evidence
            field.source = FieldSource.MANAGER if field.edited else source
            if not field.edited:
                field.value = value
            order += 1

    wanted = {(s.id, f.id) for s in template.sections for f in s.fields}
    for key, field in existing.items():
        if key not in wanted and not field.edited:
            session.delete(field)
    session.commit()


def _store_transcript(conversation_id: str, transcript: Transcript, provider_name: str) -> None:
    """Write the transcript and move the conversation on to alignment."""
    with session_scope() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise PipelineError(f"Conversation {conversation_id} disappeared mid-run.")
        conversation.transcript_json = transcript.to_dict()
        conversation.speech_provider = transcript.provider or provider_name
        conversation.speech_model = transcript.model
        conversation.language = transcript.language
        conversation.duration_seconds = transcript.duration or conversation.duration_seconds
        record_event(
            session, conversation_id, "transcribed",
            provider=conversation.speech_provider,
            model=conversation.speech_model,
            segments=len(transcript.segments),
            duration_seconds=conversation.duration_seconds,
        )
        _set_status(session, conversation, ProcessingStatus.ALIGNING,
                    "Aligning the audio to the agenda")


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def _audio_url(storage: Storage, conversation: Conversation) -> str:
    url = storage.playback_url(conversation.audio_key, expires_in=AUDIO_URL_TTL)
    if not url:
        raise PipelineError(
            "The transcription service needs a URL it can fetch the recording from, but "
            "STORAGE_BACKEND=local serves audio only through this application. Use "
            "STORAGE_BACKEND=s3, or a speech provider that reads a local file."
        )
    return url


def _stage_transcribe(conversation_id: str, settings: Settings) -> None:
    """Turn the recording into text - or hand it off to something that will."""
    with session_scope() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise PipelineError(f"Conversation {conversation_id} not found.")
        audio_key = conversation.audio_key
        audio_path = conversation.audio_path

    speech = get_speech_provider(settings)
    storage = get_storage(settings)

    if is_async_provider(speech):
        with session_scope() as session:
            conversation = session.get(Conversation, conversation_id)
            url = _audio_url(storage, conversation)
        webhook = f"{settings.base_url}/api/webhooks/transcription" if settings.base_url else ""
        if not webhook:
            raise PipelineError(
                "A hosted transcription service has to be able to call back, but this "
                "deployment does not know its own address. Set PUBLIC_BASE_URL."
            )
        job_id = speech.submit(url, webhook, settings.webhook_secret)
        with session_scope() as session:
            conversation = session.get(Conversation, conversation_id)
            conversation.transcription_job_id = job_id
            conversation.speech_provider = speech.name
            record_event(session, conversation_id, "transcription_submitted",
                         provider=speech.name, job_id=job_id)
            _set_status(session, conversation, ProcessingStatus.TRANSCRIBING,
                        "Waiting for the transcription service")
        return

    # Synchronous provider: it needs a file on this machine.
    path = storage.local_path(audio_key) or (Path(audio_path) if audio_path else None)
    if path is None or not path.exists():
        raise TranscriptionError(
            f"The recording is not readable on this machine ({audio_key!r}). "
            "A local speech provider cannot transcribe from remote storage."
        )
    transcript = speech.transcribe(path)
    if not transcript.segments:
        raise TranscriptionError("No speech was found in the audio.")
    _store_transcript(conversation_id, transcript, getattr(speech, "name", ""))


def _stage_analyse(conversation_id: str, settings: Settings) -> None:
    """Align the transcript to the agenda and draft every field.

    Works purely from the stored transcript, so it can be retried at any point
    without the recording being reachable.
    """
    with session_scope() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise PipelineError(f"Conversation {conversation_id} not found.")
        if not conversation.transcript_json:
            raise PipelineError("Cannot analyse a conversation that has no transcript yet.")
        transcript = Transcript.from_dict(conversation.transcript_json)
        template = get_template(conversation.template_id)
        manager_name = conversation.manager_name
        report_name = conversation.report_name

    analyst = get_analyst(settings)
    fallback = HeuristicAnalyst()
    degraded = ""

    try:
        structure = analyst.structure(
            transcript.segments, template, manager_name, report_name, transcript.duration
        )
    except AnalysisError as exc:
        logger.warning("Alignment fell back to the offline analyst: %s", exc)
        degraded = str(exc)
        analyst = fallback
        structure = fallback.structure(
            transcript.segments, template, manager_name, report_name, transcript.duration
        )

    # Speaker labels that came from the recording beat anything inferred.
    if any(s.speaker for s in transcript.segments):
        labelled = fallback.structure(
            transcript.segments, template, manager_name, report_name, transcript.duration
        )
        structure.speaker_roles = labelled.speaker_roles
        structure.speaker_confidence = labelled.speaker_confidence

    with session_scope() as session:
        conversation = session.get(Conversation, conversation_id)
        _store_segments(session, conversation, transcript, structure)
        record_event(session, conversation_id, "aligned",
                     method=structure.method,
                     sections=len(set(structure.section_ids)),
                     notes=structure.notes or None)
        _set_status(session, conversation, ProcessingStatus.DRAFTING,
                    "Filling in the feedback form")

    try:
        draft = analyst.draft(transcript.segments, structure, template, manager_name, report_name)
    except AnalysisError as exc:
        logger.warning("Drafting fell back to the offline analyst: %s", exc)
        degraded = degraded or str(exc)
        draft = fallback.draft(transcript.segments, structure, template, manager_name, report_name)

    metrics = compute_metrics(transcript.segments, structure, template, transcript.duration)
    metrics["draft_method"] = draft.method
    metrics["draft_model"] = draft.model
    if degraded:
        metrics["degraded_reason"] = degraded

    with session_scope() as session:
        conversation = session.get(Conversation, conversation_id)
        _store_fields(session, conversation, template, transcript, draft)
        conversation.analysis_json = metrics
        conversation.analysis_provider = draft.method
        conversation.analysis_model = draft.model
        conversation.processed_at = utcnow()
        conversation.error = None
        record_event(session, conversation_id, "drafted",
                     method=draft.method, model=draft.model,
                     fields=len(draft.fields), degraded_reason=degraded or None)
        detail = (
            f"Drafted offline: {truncate(degraded, 160)}" if degraded else f"Drafted with {draft.method}"
        )
        _set_status(session, conversation, ProcessingStatus.READY, detail)


# --------------------------------------------------------------------------
# The stage machine
# --------------------------------------------------------------------------

def advance(conversation_id: str, settings: Settings | None = None) -> ProcessingStatus:
    """Run whichever stage the conversation is due, and return its new status.

    Safe to call at any time from anywhere. If the conversation is finished, or
    another caller already holds the stage, this does nothing.
    """
    settings = settings or get_settings()

    with session_scope() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise PipelineError(f"Conversation {conversation_id} not found.")
        status = conversation.status

    try:
        if status is ProcessingStatus.UPLOADED:
            with session_scope() as session:
                if not _claim(session, conversation_id, ProcessingStatus.UPLOADED,
                              ProcessingStatus.TRANSCRIBING, "Transcribing the recording"):
                    return _status_of(conversation_id)
            _stage_transcribe(conversation_id, settings)

        elif status is ProcessingStatus.ALIGNING:
            # Claiming moves it out of READY_TO_RUN, so a second caller
            # arriving mid-analysis steps aside instead of duplicating it.
            with session_scope() as session:
                if not _claim(session, conversation_id, ProcessingStatus.ALIGNING,
                              ProcessingStatus.DRAFTING, "Aligning the audio to the agenda"):
                    return _status_of(conversation_id)
            _stage_analyse(conversation_id, settings)

        elif status is ProcessingStatus.DRAFTING:
            # A previous attempt claimed this and never finished. Alignment is
            # cheap to redo and the transcript is already stored, so start the
            # analysis stage over rather than trying to resume inside it.
            _stage_analyse(conversation_id, settings)

    except Exception as exc:  # noqa: BLE001 - the failure belongs on the record
        _fail(conversation_id, exc)

    return _status_of(conversation_id)


def _status_of(conversation_id: str) -> ProcessingStatus:
    with session_scope() as session:
        conversation = session.get(Conversation, conversation_id)
        return conversation.status if conversation else ProcessingStatus.FAILED


def run_until_blocked(
    conversation_id: str, settings: Settings | None = None, max_stages: int = 6
) -> ProcessingStatus:
    """Advance repeatedly until the conversation finishes or has to wait.

    It has to wait when a hosted transcription service is working: there is
    nothing to do until the webhook arrives.
    """
    settings = settings or get_settings()
    status = _status_of(conversation_id)
    for _ in range(max_stages):
        if conversation_is_finished(status) or status in WAITING_ELSEWHERE:
            break
        next_status = advance(conversation_id, settings)
        if next_status is status:
            break
        status = next_status
    return status


def receive_transcription(job_id: str, settings: Settings | None = None) -> str | None:
    """Handle a transcription service telling us a job is done.

    Collects the transcript, then carries straight on into the analysis stage -
    this callback is the only invocation guaranteed to happen, so it does the
    remaining work rather than handing off again.
    """
    settings = settings or get_settings()

    with session_scope() as session:
        conversation = session.scalars(
            select(Conversation).where(Conversation.transcription_job_id == job_id)
        ).first()
        if conversation is None:
            logger.warning("Transcription callback for unknown job %s", job_id)
            return None
        conversation_id = conversation.id
        already_done = conversation.transcript_json is not None

    if already_done:
        # A duplicate callback. Push the conversation along in case it stalled,
        # but do not re-fetch a transcript we already hold.
        run_until_blocked(conversation_id, settings)
        return conversation_id

    try:
        speech = get_speech_provider(settings)
        if not is_async_provider(speech):
            raise PipelineError(
                f"Got a transcription callback but SPEECH_PROVIDER is {settings.speech_provider!r}, "
                "which does not use callbacks."
            )
        transcript = speech.fetch(job_id)
        _store_transcript(conversation_id, transcript, speech.name)
    except Exception as exc:  # noqa: BLE001
        _fail(conversation_id, exc)
        return conversation_id

    run_until_blocked(conversation_id, settings)
    return conversation_id


def restart(conversation_id: str, settings: Settings | None = None) -> None:
    """Send a conversation back to the start of the pipeline.

    Because stages are status-driven, re-running is a matter of resetting the
    status - not of calling anything twice. Fields the manager edited survive,
    since _store_fields never overwrites them.
    """
    with session_scope() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise PipelineError(f"Conversation {conversation_id} not found.")
        conversation.transcription_job_id = ""
        conversation.error = None
        _set_status(session, conversation, ProcessingStatus.UPLOADED, "Queued for reprocessing")


def resume(conversation_id: str, settings: Settings | None = None) -> ProcessingStatus:
    """Nudge a conversation that looks stuck.

    Called by the browser while it polls for status, and safe to wire to a cron.
    A conversation waiting on a transcription webhook is left alone - it is not
    stuck, it is queued somewhere else.
    """
    settings = settings or get_settings()
    with session_scope() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise PipelineError(f"Conversation {conversation_id} not found.")
        status = conversation.status
        age = conversation.stage_age_seconds

    if conversation_is_finished(status) or status in WAITING_ELSEWHERE:
        return status
    if status in IN_PROGRESS and age is not None and age < STALL_SECONDS:
        return status  # somebody is on it; leave them to it
    return run_until_blocked(conversation_id, settings)


def conversation_is_finished(status: ProcessingStatus) -> bool:
    return status in (ProcessingStatus.READY, ProcessingStatus.FAILED)


# --------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------

def process_conversation(conversation_id: str, settings: Settings | None = None) -> None:
    """Run every stage to completion. Used by the thread runner and by tests."""
    settings = settings or get_settings()
    try:
        run_until_blocked(conversation_id, settings)
    finally:
        with _inflight_lock:
            _inflight.discard(conversation_id)


def submit(conversation_id: str, settings: Settings | None = None) -> bool:
    """Start processing, in whichever way this deployment can.

    On a long-lived server the work goes to a background thread. On a
    serverless host there is no such thing as a background thread that outlives
    the response, so the first stage runs inline - it is a fast handoff to the
    transcription service - and the webhook drives the rest.
    """
    settings = settings or get_settings()

    if settings.is_serverless:
        run_until_blocked(conversation_id, settings)
        return True

    with _inflight_lock:
        if conversation_id in _inflight:
            return False
        _inflight.add(conversation_id)
    _executor.submit(process_conversation, conversation_id, settings)
    return True


def is_processing(conversation_id: str) -> bool:
    with _inflight_lock:
        return conversation_id in _inflight
