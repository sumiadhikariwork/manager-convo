"""Persistence model.

Three ideas drive the shape of these tables:

1. The raw transcript is immutable. It is written once, at ingest, and never
   rewritten - it is the evidence the record rests on.
2. Every drafted field keeps both what the machine wrote and what the manager
   left there, so the record shows plainly which is which.
3. Everything that touches a record is appended to an audit log.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ProcessingStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    TRANSCRIBING = "transcribing"
    ALIGNING = "aligning"
    DRAFTING = "drafting"
    READY = "ready"
    FAILED = "failed"


class SpeakerRole(str, enum.Enum):
    MANAGER = "manager"
    REPORT = "report"
    UNKNOWN = "unknown"


class FieldSource(str, enum.Enum):
    CLAUDE = "claude"
    HEURISTIC = "heuristic"
    MANAGER = "manager"


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(300), default="")
    template_id: Mapped[str] = mapped_column(String(100))
    manager_name: Mapped[str] = mapped_column(String(200), default="")
    report_name: Mapped[str] = mapped_column(String(200), default="")
    occurred_on: Mapped[str] = mapped_column(String(20), default="")

    #: Both people were told the conversation was being recorded.
    consent_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)

    audio_filename: Mapped[str] = mapped_column(String(400), default="")
    audio_path: Mapped[str] = mapped_column(String(700), default="")
    audio_mime: Mapped[str] = mapped_column(String(120), default="application/octet-stream")
    audio_bytes: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus, native_enum=False, length=20), default=ProcessingStatus.UPLOADED
    )
    status_detail: Mapped[str] = mapped_column(String(400), default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    speech_provider: Mapped[str] = mapped_column(String(60), default="")
    speech_model: Mapped[str] = mapped_column(String(120), default="")
    analysis_provider: Mapped[str] = mapped_column(String(60), default="")
    analysis_model: Mapped[str] = mapped_column(String(120), default="")
    language: Mapped[str] = mapped_column(String(20), default="")

    #: Immutable transcript exactly as returned by the speech provider.
    transcript_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    #: Derived conversation-level metrics (talk ratio, question counts, timing).
    analysis_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    segments: Mapped[list["Segment"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Segment.index",
    )
    fields: Mapped[list["FormField"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="FormField.order",
    )
    events: Mapped[list["AuditEvent"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="AuditEvent.at",
    )

    @property
    def is_terminal(self) -> bool:
        return self.status in (ProcessingStatus.READY, ProcessingStatus.FAILED)


class Segment(Base):
    """One utterance of the transcript, with its place on the timeline."""

    __tablename__ = "segments"
    __table_args__ = (Index("ix_segments_conversation_index", "conversation_id", "index"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    index: Mapped[int] = mapped_column(Integer)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)

    #: Raw label from the speech provider, when it does diarisation.
    speaker_label: Mapped[str] = mapped_column(String(60), default="")
    speaker_role: Mapped[SpeakerRole] = mapped_column(
        Enum(SpeakerRole, native_enum=False, length=12), default=SpeakerRole.UNKNOWN
    )
    speaker_confidence: Mapped[float] = mapped_column(Float, default=0.0)

    #: Which agenda item this stretch of the conversation belongs to.
    section_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    section_confidence: Mapped[float] = mapped_column(Float, default=0.0)

    conversation: Mapped[Conversation] = relationship(back_populates="segments")


class FormField(Base):
    """One box on the feedback form, as filled for one conversation."""

    __tablename__ = "form_fields"
    __table_args__ = (
        Index("ix_form_fields_lookup", "conversation_id", "section_id", "field_id", unique=True),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    section_id: Mapped[str] = mapped_column(String(100))
    field_id: Mapped[str] = mapped_column(String(100))
    order: Mapped[int] = mapped_column(Integer, default=0)
    kind: Mapped[str] = mapped_column(String(20), default="text")

    #: What the pipeline drafted. Never overwritten by an edit.
    draft_value: Mapped[Any] = mapped_column(JSON, nullable=True)
    #: What the record currently says.
    value: Mapped[Any] = mapped_column(JSON, nullable=True)

    source: Mapped[FieldSource] = mapped_column(
        Enum(FieldSource, native_enum=False, length=12), default=FieldSource.HEURISTIC
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    #: [{"segment_index": int, "start": float, "end": float, "quote": str}, ...]
    evidence_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)

    edited: Mapped[bool] = mapped_column(Boolean, default=False)
    edited_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    edited_by: Mapped[str] = mapped_column(String(200), default="")

    conversation: Mapped[Conversation] = relationship(back_populates="fields")


class AuditEvent(Base):
    """Append-only trail: who did what to this record, and when."""

    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    actor: Mapped[str] = mapped_column(String(200), default="system")
    action: Mapped[str] = mapped_column(String(80))
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    conversation: Mapped[Conversation] = relationship(back_populates="events")


def record_event(
    session, conversation_id: str, action: str, *, actor: str = "system", **detail: Any
) -> AuditEvent:
    event = AuditEvent(
        conversation_id=conversation_id, action=action, actor=actor, detail=detail or None
    )
    session.add(event)
    return event
