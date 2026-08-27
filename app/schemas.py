"""API request and response shapes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FieldSpecOut(BaseModel):
    id: str
    label: str
    kind: str
    placeholder: str = ""
    guidance: str = ""
    choices: list[str] = Field(default_factory=list)


class SectionSpecOut(BaseModel):
    id: str
    title: str
    minutes: int
    prompt: str
    kind: str
    fields: list[FieldSpecOut]


class TemplateOut(BaseModel):
    id: str
    name: str
    description: str
    planned_minutes: int
    sections: list[SectionSpecOut]


class EvidenceOut(BaseModel):
    segment_index: int
    start: float
    end: float
    quote: str


class FieldOut(BaseModel):
    id: str
    section_id: str
    field_id: str
    label: str
    kind: str
    placeholder: str = ""
    guidance: str = ""
    choices: list[str] = Field(default_factory=list)
    value: Any = None
    draft_value: Any = None
    source: str
    confidence: float
    edited: bool
    edited_at: str | None = None
    edited_by: str = ""
    evidence: list[EvidenceOut] = Field(default_factory=list)


class SegmentOut(BaseModel):
    index: int
    start: float
    end: float
    text: str
    speaker_label: str = ""
    speaker_role: str = "unknown"
    speaker_confidence: float = 0.0
    section_id: str | None = None
    section_confidence: float = 0.0


class ConversationSummary(BaseModel):
    id: str
    title: str
    template_id: str
    template_name: str
    manager_name: str
    report_name: str
    occurred_on: str
    status: str
    status_detail: str
    error: str | None = None
    duration_seconds: float | None = None
    audio_filename: str
    created_at: str
    updated_at: str
    processed_at: str | None = None
    analysis_provider: str = ""
    edited_field_count: int = 0
    field_count: int = 0


class ConversationDetail(ConversationSummary):
    consent_confirmed: bool = False
    speech_provider: str = ""
    speech_model: str = ""
    analysis_model: str = ""
    language: str = ""
    template: TemplateOut
    metrics: dict[str, Any] = Field(default_factory=dict)
    fields: list[FieldOut] = Field(default_factory=list)
    segments: list[SegmentOut] = Field(default_factory=list)


class FieldUpdate(BaseModel):
    value: Any = None
    edited_by: str = ""


class AuditEventOut(BaseModel):
    at: str
    actor: str
    action: str
    detail: dict[str, Any] | None = None
