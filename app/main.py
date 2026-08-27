"""HTTP API and web application."""

from __future__ import annotations

import json
import logging
import mimetypes
import re
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, Iterator

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import export
from app.config import get_settings
from app.db import get_session, init_db
from app.models import (
    Conversation,
    FieldSource,
    FormField,
    ProcessingStatus,
    Segment,
    record_event,
    utcnow,
)
from app.pipeline import is_processing, submit
from app.schemas import (
    AuditEventOut,
    ConversationDetail,
    ConversationSummary,
    EvidenceOut,
    FieldOut,
    FieldSpecOut,
    FieldUpdate,
    SectionSpecOut,
    SegmentOut,
    TemplateOut,
)
from app.templates import ConversationTemplate, get_template, list_templates
from app.util import normalise

logger = logging.getLogger(__name__)
settings = get_settings()

STATIC_DIR = Path(__file__).parent / "static"

ALLOWED_AUDIO_SUFFIXES = {
    ".wav", ".mp3", ".m4a", ".mp4", ".aac", ".flac", ".ogg", ".oga", ".opus", ".webm", ".wma", ".aiff",
}

@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    init_db()
    yield


app = FastAPI(
    title="Manager conversation records",
    description=(
        "Upload the recording of a coaching conversation; get back an aligned "
        "transcript and a feedback form filled in from what was actually said."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------

def _template_out(template: ConversationTemplate) -> TemplateOut:
    return TemplateOut(
        id=template.id,
        name=template.name,
        description=template.description,
        planned_minutes=template.planned_minutes,
        sections=[
            SectionSpecOut(
                id=s.id,
                title=s.title,
                minutes=s.minutes,
                prompt=s.prompt,
                kind=s.kind,
                fields=[
                    FieldSpecOut(
                        id=f.id,
                        label=f.label,
                        kind=f.kind,
                        placeholder=f.placeholder,
                        guidance=f.guidance,
                        choices=list(f.choices),
                    )
                    for f in s.fields
                ],
            )
            for s in template.sections
        ],
    )


def _summary(conversation: Conversation, template: ConversationTemplate, session: Session) -> ConversationSummary:
    total = session.scalar(
        select(func.count()).select_from(FormField).where(FormField.conversation_id == conversation.id)
    ) or 0
    edited = session.scalar(
        select(func.count())
        .select_from(FormField)
        .where(FormField.conversation_id == conversation.id, FormField.edited.is_(True))
    ) or 0
    return ConversationSummary(
        id=conversation.id,
        title=conversation.title,
        template_id=conversation.template_id,
        template_name=template.name,
        manager_name=conversation.manager_name,
        report_name=conversation.report_name,
        occurred_on=conversation.occurred_on,
        status=conversation.status.value,
        status_detail=conversation.status_detail,
        error=conversation.error,
        duration_seconds=conversation.duration_seconds,
        audio_filename=conversation.audio_filename,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
        processed_at=conversation.processed_at.isoformat() if conversation.processed_at else None,
        analysis_provider=conversation.analysis_provider,
        edited_field_count=edited,
        field_count=total,
    )


def _field_out(field: FormField, template: ConversationTemplate) -> FieldOut:
    spec = template.field(field.section_id, field.field_id)
    return FieldOut(
        id=field.id,
        section_id=field.section_id,
        field_id=field.field_id,
        label=spec.label if spec else field.field_id,
        kind=field.kind,
        placeholder=spec.placeholder if spec else "",
        guidance=spec.guidance if spec else "",
        choices=list(spec.choices) if spec else [],
        value=field.value,
        draft_value=field.draft_value,
        source=field.source.value,
        confidence=field.confidence,
        edited=field.edited,
        edited_at=field.edited_at.isoformat() if field.edited_at else None,
        edited_by=field.edited_by,
        evidence=[EvidenceOut(**e) for e in (field.evidence_json or [])],
    )


def _detail(conversation: Conversation, session: Session) -> ConversationDetail:
    template = get_template(conversation.template_id)
    summary = _summary(conversation, template, session)
    fields = list(
        session.scalars(
            select(FormField)
            .where(FormField.conversation_id == conversation.id)
            .order_by(FormField.order)
        )
    )
    segments = list(
        session.scalars(
            select(Segment).where(Segment.conversation_id == conversation.id).order_by(Segment.index)
        )
    )
    return ConversationDetail(
        **summary.model_dump(),
        consent_confirmed=conversation.consent_confirmed,
        speech_provider=conversation.speech_provider,
        speech_model=conversation.speech_model,
        analysis_model=conversation.analysis_model,
        language=conversation.language,
        template=_template_out(template),
        metrics=conversation.analysis_json or {},
        fields=[_field_out(f, template) for f in fields],
        segments=[
            SegmentOut(
                index=s.index,
                start=s.start,
                end=s.end,
                text=s.text,
                speaker_label=s.speaker_label,
                speaker_role=s.speaker_role.value,
                speaker_confidence=s.speaker_confidence,
                section_id=s.section_id,
                section_confidence=s.section_confidence,
            )
            for s in segments
        ],
    )


def _get_conversation(conversation_id: str, session: Session) -> Conversation:
    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------

@app.get("/api/templates", response_model=list[TemplateOut])
def api_templates() -> list[TemplateOut]:
    return [_template_out(t) for t in list_templates()]


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    """What this deployment is wired up to. Shown in the UI footer."""
    return {
        "speech_provider": settings.speech_provider,
        "speech_model": settings.whisper_model if settings.speech_provider != "fixture" else "sidecar",
        "analysis_provider": "claude" if settings.claude_available() else "heuristic",
        "analysis_model": settings.claude_model if settings.claude_available() else "extractive",
        "max_upload_mb": settings.max_upload_mb,
    }


# --------------------------------------------------------------------------
# Conversations
# --------------------------------------------------------------------------

@app.post("/api/conversations", response_model=ConversationDetail, status_code=201)
async def api_create_conversation(
    audio: UploadFile = File(...),
    title: str = Form(""),
    template_id: str = Form(""),
    manager_name: str = Form(""),
    report_name: str = Form(""),
    occurred_on: str = Form(""),
    consent_confirmed: bool = Form(False),
    session: Session = Depends(get_session),
) -> ConversationDetail:
    if not consent_confirmed:
        raise HTTPException(
            status_code=400,
            detail=(
                "Confirm that both people knew the conversation was being recorded "
                "before uploading it."
            ),
        )

    try:
        template = get_template(template_id or None)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    filename = Path(audio.filename or "recording").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type {suffix or '(none)'}. Accepted: "
            + ", ".join(sorted(ALLOWED_AUDIO_SUFFIXES)),
        )

    conversation = Conversation(
        title=normalise(title) or f"Conversation with {normalise(report_name) or 'team member'}",
        template_id=template.id,
        manager_name=normalise(manager_name),
        report_name=normalise(report_name),
        occurred_on=normalise(occurred_on) or date.today().isoformat(),
        consent_confirmed=True,
        audio_filename=filename,
        audio_mime=audio.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream",
    )
    session.add(conversation)
    session.flush()

    destination = settings.audio_dir / f"{conversation.id}{suffix}"
    written = 0
    try:
        with destination.open("wb") as handle:
            while chunk := await audio.read(1024 * 1024):
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    handle.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Recording is larger than the {settings.max_upload_mb} MB limit.",
                    )
                handle.write(chunk)
    except HTTPException:
        session.rollback()
        raise
    finally:
        await audio.close()

    if written == 0:
        destination.unlink(missing_ok=True)
        session.rollback()
        raise HTTPException(status_code=400, detail="The uploaded file was empty.")

    conversation.audio_path = str(destination)
    conversation.audio_bytes = written
    record_event(
        session,
        conversation.id,
        "uploaded",
        actor=conversation.manager_name or "manager",
        filename=filename,
        bytes=written,
        template_id=template.id,
    )
    session.commit()

    submit(conversation.id, settings)
    return _detail(conversation, session)


@app.get("/api/conversations", response_model=list[ConversationSummary])
def api_list_conversations(session: Session = Depends(get_session)) -> list[ConversationSummary]:
    conversations = session.scalars(
        select(Conversation).order_by(Conversation.created_at.desc())
    ).all()
    return [_summary(c, get_template(c.template_id), session) for c in conversations]


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
def api_get_conversation(conversation_id: str, session: Session = Depends(get_session)) -> ConversationDetail:
    return _detail(_get_conversation(conversation_id, session), session)


@app.get("/api/conversations/{conversation_id}/status")
def api_status(conversation_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    """Cheap poll target while the pipeline runs."""
    conversation = _get_conversation(conversation_id, session)
    return {
        "id": conversation.id,
        "status": conversation.status.value,
        "status_detail": conversation.status_detail,
        "error": conversation.error,
        "processing": is_processing(conversation.id),
        "updated_at": conversation.updated_at.isoformat(),
    }


@app.post("/api/conversations/{conversation_id}/reprocess", response_model=ConversationDetail)
def api_reprocess(conversation_id: str, session: Session = Depends(get_session)) -> ConversationDetail:
    """Re-run the pipeline. Fields the manager has edited are left alone."""
    conversation = _get_conversation(conversation_id, session)
    if is_processing(conversation.id):
        raise HTTPException(status_code=409, detail="This conversation is already being processed.")
    if not Path(conversation.audio_path or "").exists():
        raise HTTPException(status_code=410, detail="The audio for this conversation is no longer on disk.")
    record_event(session, conversation.id, "reprocess_requested", actor="manager")
    conversation.status = ProcessingStatus.UPLOADED
    conversation.status_detail = "Queued for reprocessing"
    conversation.error = None
    session.commit()
    submit(conversation.id, settings)
    session.refresh(conversation)
    return _detail(conversation, session)


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def api_delete(conversation_id: str, session: Session = Depends(get_session)) -> Response:
    conversation = _get_conversation(conversation_id, session)
    audio_path = Path(conversation.audio_path or "")
    session.delete(conversation)
    session.commit()
    # Only ever remove files this app put in its own audio directory.
    audio_dir = settings.audio_dir.resolve()
    if audio_path.exists() and audio_path.resolve().is_relative_to(audio_dir):
        audio_path.unlink(missing_ok=True)
    return Response(status_code=204)


# --------------------------------------------------------------------------
# Fields
# --------------------------------------------------------------------------

@app.patch("/api/conversations/{conversation_id}/fields/{field_id}", response_model=FieldOut)
def api_update_field(
    conversation_id: str,
    field_id: str,
    update: FieldUpdate,
    session: Session = Depends(get_session),
) -> FieldOut:
    conversation = _get_conversation(conversation_id, session)
    field = session.get(FormField, field_id)
    if field is None or field.conversation_id != conversation.id:
        raise HTTPException(status_code=404, detail="Field not found")

    previous = field.value
    field.value = update.value
    field.edited = field.value != field.draft_value
    field.edited_at = utcnow() if field.edited else None
    field.edited_by = normalise(update.edited_by) or conversation.manager_name or "manager"
    field.source = FieldSource.MANAGER if field.edited else _draft_source(conversation)
    record_event(
        session,
        conversation.id,
        "field_edited" if field.edited else "field_reverted",
        actor=field.edited_by,
        section_id=field.section_id,
        field_id=field.field_id,
        previous=previous,
        current=field.value,
    )
    session.commit()
    return _field_out(field, get_template(conversation.template_id))


@app.post("/api/conversations/{conversation_id}/fields/{field_id}/revert", response_model=FieldOut)
def api_revert_field(
    conversation_id: str, field_id: str, session: Session = Depends(get_session)
) -> FieldOut:
    conversation = _get_conversation(conversation_id, session)
    field = session.get(FormField, field_id)
    if field is None or field.conversation_id != conversation.id:
        raise HTTPException(status_code=404, detail="Field not found")
    previous = field.value
    field.value = field.draft_value
    field.edited = False
    field.edited_at = None
    field.edited_by = ""
    field.source = _draft_source(conversation)
    record_event(
        session,
        conversation.id,
        "field_reverted",
        actor="manager",
        section_id=field.section_id,
        field_id=field.field_id,
        previous=previous,
    )
    session.commit()
    return _field_out(field, get_template(conversation.template_id))


def _draft_source(conversation: Conversation) -> FieldSource:
    return FieldSource.CLAUDE if conversation.analysis_provider == "claude" else FieldSource.HEURISTIC


@app.get("/api/conversations/{conversation_id}/audit", response_model=list[AuditEventOut])
def api_audit(conversation_id: str, session: Session = Depends(get_session)) -> list[AuditEventOut]:
    conversation = _get_conversation(conversation_id, session)
    return [
        AuditEventOut(
            at=event.at.isoformat(), actor=event.actor, action=event.action, detail=event.detail
        )
        for event in conversation.events
    ]


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def _file_chunks(path: Path, start: int, length: int, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


@app.get("/api/conversations/{conversation_id}/audio")
def api_audio(
    conversation_id: str, request: Request, session: Session = Depends(get_session)
) -> Response:
    """Serve the recording, honouring Range requests so the player can seek."""
    conversation = _get_conversation(conversation_id, session)
    path = Path(conversation.audio_path or "")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    size = path.stat().st_size
    media_type = conversation.audio_mime or "application/octet-stream"
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(
            path,
            media_type=media_type,
            headers={"Accept-Ranges": "bytes", "Content-Length": str(size)},
        )

    match = _RANGE_RE.fullmatch(range_header.strip())
    if not match:
        raise HTTPException(status_code=416, detail="Malformed Range header")
    raw_start, raw_end = match.groups()
    if raw_start:
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1
    else:  # suffix range: last N bytes
        length = int(raw_end or 0)
        start = max(size - length, 0)
        end = size - 1
    if start >= size:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    end = min(end, size - 1)
    length = end - start + 1

    return StreamingResponse(
        _file_chunks(path, start, length),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        },
    )


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

@app.get("/api/conversations/{conversation_id}/export.md")
def api_export_markdown(
    conversation_id: str,
    transcript: bool = False,
    session: Session = Depends(get_session),
) -> Response:
    conversation = _get_conversation(conversation_id, session)
    template = get_template(conversation.template_id)
    fields = list(
        session.scalars(
            select(FormField).where(FormField.conversation_id == conversation.id).order_by(FormField.order)
        )
    )
    segments = list(
        session.scalars(
            select(Segment).where(Segment.conversation_id == conversation.id).order_by(Segment.index)
        )
    )
    body = export.to_markdown(conversation, template, fields, include_transcript=transcript, segments=segments)
    filename = f"{conversation.occurred_on or 'record'}-{conversation.report_name or 'conversation'}.md".replace(" ", "-")
    return Response(
        content=body,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/conversations/{conversation_id}/export.json")
def api_export_json(conversation_id: str, session: Session = Depends(get_session)) -> Response:
    conversation = _get_conversation(conversation_id, session)
    template = get_template(conversation.template_id)
    fields = list(
        session.scalars(
            select(FormField).where(FormField.conversation_id == conversation.id).order_by(FormField.order)
        )
    )
    segments = list(
        session.scalars(
            select(Segment).where(Segment.conversation_id == conversation.id).order_by(Segment.index)
        )
    )
    payload = export.to_dict(conversation, template, fields, segments)
    return Response(
        content=json.dumps(payload, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{conversation.id}.json"'},
    )


# --------------------------------------------------------------------------
# Web app
# --------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
