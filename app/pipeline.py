"""The ingest pipeline: audio in, filled-in feedback record out.

    upload → transcribe → align to the agenda → draft every field → ready

Each stage writes its status so the browser can follow along, and each stage is
recoverable: if the Claude call is unavailable or declines, the run continues on
the offline analyst rather than failing, and says so on the record.
"""

from __future__ import annotations

import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from sqlalchemy import delete, select
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
from app.templates import ConversationTemplate, get_template
from app.transcription import TranscriptionError, get_speech_provider
from app.transcription.base import Transcript, TranscriptSegment
from app.util import truncate

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline")
_inflight: set[str] = set()
_inflight_lock = threading.Lock()


# --------------------------------------------------------------------------
# Persistence helpers
# --------------------------------------------------------------------------

def _set_status(
    session: Session, conversation: Conversation, status: ProcessingStatus, detail: str = ""
) -> None:
    conversation.status = status
    conversation.status_detail = detail
    conversation.updated_at = utcnow()
    session.add(conversation)
    session.commit()


def _store_segments(
    session: Session,
    conversation: Conversation,
    transcript: Transcript,
    structure: StructureResult,
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
                speaker_role=SpeakerRole(role) if role in SpeakerRole._value2member_map_ else SpeakerRole.UNKNOWN,
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

    # Drop fields left over from a template that no longer has them.
    wanted = {(s.id, f.id) for s in template.sections for f in s.fields}
    for key, field in existing.items():
        if key not in wanted and not field.edited:
            session.delete(field)
    session.commit()


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------

def process_conversation(conversation_id: str, settings: Settings | None = None) -> None:
    """Run the full pipeline for one conversation. Safe to call again."""
    settings = settings or get_settings()

    with session_scope() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            logger.warning("process_conversation: %s not found", conversation_id)
            return
        template = get_template(conversation.template_id)
        audio_path = Path(conversation.audio_path)
        manager_name = conversation.manager_name
        report_name = conversation.report_name

    try:
        # -- 1. transcribe -------------------------------------------------
        with session_scope() as session:
            conversation = session.get(Conversation, conversation_id)
            _set_status(session, conversation, ProcessingStatus.TRANSCRIBING, "Transcribing the recording")

        speech = get_speech_provider(settings)
        transcript = speech.transcribe(audio_path)
        if not transcript.segments:
            raise TranscriptionError("No speech was found in the audio.")

        with session_scope() as session:
            conversation = session.get(Conversation, conversation_id)
            conversation.transcript_json = transcript.to_dict()
            conversation.speech_provider = transcript.provider or getattr(speech, "name", "")
            conversation.speech_model = transcript.model
            conversation.language = transcript.language
            conversation.duration_seconds = transcript.duration or conversation.duration_seconds
            record_event(
                session,
                conversation_id,
                "transcribed",
                provider=conversation.speech_provider,
                model=conversation.speech_model,
                segments=len(transcript.segments),
                duration_seconds=conversation.duration_seconds,
            )
            _set_status(session, conversation, ProcessingStatus.ALIGNING, "Aligning the audio to the agenda")

        # -- 2. align ------------------------------------------------------
        analyst = get_analyst(settings)
        fallback = HeuristicAnalyst()
        degraded: str = ""

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

        # Speaker labels straight from the recording beat anything inferred.
        if any(s.speaker for s in transcript.segments):
            labelled = fallback.structure(
                transcript.segments, template, manager_name, report_name, transcript.duration
            )
            structure.speaker_roles = labelled.speaker_roles
            structure.speaker_confidence = labelled.speaker_confidence

        with session_scope() as session:
            conversation = session.get(Conversation, conversation_id)
            _store_segments(session, conversation, transcript, structure)
            record_event(
                session,
                conversation_id,
                "aligned",
                method=structure.method,
                sections=len({s for s in structure.section_ids}),
                notes=structure.notes or None,
            )
            _set_status(session, conversation, ProcessingStatus.DRAFTING, "Filling in the feedback form")

        # -- 3. draft ------------------------------------------------------
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

        # -- 4. store ------------------------------------------------------
        with session_scope() as session:
            conversation = session.get(Conversation, conversation_id)
            _store_fields(session, conversation, template, transcript, draft)
            conversation.analysis_json = metrics
            conversation.analysis_provider = draft.method
            conversation.analysis_model = draft.model
            conversation.processed_at = utcnow()
            conversation.error = None
            record_event(
                session,
                conversation_id,
                "drafted",
                method=draft.method,
                model=draft.model,
                fields=len(draft.fields),
                degraded_reason=degraded or None,
            )
            detail = (
                f"Drafted offline: {truncate(degraded, 160)}"
                if degraded
                else f"Drafted with {draft.method}"
            )
            _set_status(session, conversation, ProcessingStatus.READY, detail)

    except Exception as exc:  # noqa: BLE001 - the failure belongs on the record
        logger.exception("Pipeline failed for %s", conversation_id)
        with session_scope() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is not None:
                conversation.error = f"{type(exc).__name__}: {exc}"
                record_event(
                    session,
                    conversation_id,
                    "failed",
                    error=str(exc),
                    trace=truncate(traceback.format_exc(), 2000),
                )
                _set_status(session, conversation, ProcessingStatus.FAILED, truncate(str(exc), 300))
    finally:
        with _inflight_lock:
            _inflight.discard(conversation_id)


def submit(conversation_id: str, settings: Settings | None = None) -> bool:
    """Queue a conversation for processing. Returns False if already queued."""
    with _inflight_lock:
        if conversation_id in _inflight:
            return False
        _inflight.add(conversation_id)
    _executor.submit(process_conversation, conversation_id, settings)
    return True


def is_processing(conversation_id: str) -> bool:
    with _inflight_lock:
        return conversation_id in _inflight
