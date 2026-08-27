"""Shareable renderings of a finished record.

The exported document is the point of the whole system: something the manager
and the person coached can both read, that says what was said, who said it, and
where in the recording to check.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.models import Conversation, FieldSource, FormField
from app.templates import ConversationTemplate, FormFieldSpec
from app.util import format_timestamp

_SOURCE_LABEL = {
    FieldSource.CLAUDE: "drafted from the recording",
    FieldSource.HEURISTIC: "extracted from the recording",
    FieldSource.MANAGER: "edited by the manager",
}


def _value_lines(spec: FormFieldSpec, value: Any) -> list[str]:
    if spec.kind == "actions":
        lines = []
        for action in value or []:
            bits = [f"- {action.get('action', '').strip()}"]
            meta = []
            if action.get("owner"):
                meta.append(f"owner: {action['owner']}")
            if action.get("due"):
                meta.append(f"due: {action['due']}")
            if action.get("support"):
                meta.append(f"support: {action['support']}")
            if meta:
                bits.append(f"  ({'; '.join(meta)})")
            lines.append("\n".join(bits))
        return lines or ["- _nothing recorded_"]
    if spec.kind == "list":
        return [f"- {item}" for item in (value or [])] or ["- _nothing recorded_"]
    text = (value or "").strip() if isinstance(value, str) else str(value or "")
    return [text or "_nothing recorded_"]


def to_markdown(
    conversation: Conversation,
    template: ConversationTemplate,
    fields: Sequence[FormField],
    include_transcript: bool = False,
    segments: Sequence[Any] = (),
) -> str:
    by_key = {(f.section_id, f.field_id): f for f in fields}
    analysis = conversation.analysis_json or {}
    talk = (analysis.get("talk_ratio") or {}).get("manager")

    out: list[str] = []
    title = conversation.title or "Coaching conversation"
    out.append(f"# {title}")
    out.append("")
    meta = [
        f"**Template:** {template.name}",
        f"**Manager:** {conversation.manager_name or '—'}",
        f"**Person:** {conversation.report_name or '—'}",
        f"**Date:** {conversation.occurred_on or conversation.created_at.date().isoformat()}",
        f"**Recording:** {conversation.audio_filename or '—'} ({format_timestamp(conversation.duration_seconds)})",
    ]
    if talk is not None:
        meta.append(f"**Manager talk time:** {round(talk * 100)}%")
    out.extend(meta)
    out.append("")
    out.append(
        "> Drafted automatically from the recording, then reviewed by the manager. "
        "Timestamps point at the moment in the recording each entry is drawn from."
    )
    out.append("")

    for section in template.sections:
        heading = section.title
        if section.kind == "agenda":
            actual = next(
                (s for s in analysis.get("sections", []) if s.get("section_id") == section.id), None
            )
            timing = f" · {section.minutes} min planned"
            if actual and actual.get("segment_count"):
                timing += (
                    f", {format_timestamp(actual['start'])}–{format_timestamp(actual['end'])} actual"
                )
            heading += timing
        out.append(f"## {heading}")
        if section.kind == "agenda":
            out.append(f"_{section.prompt}_")
        out.append("")

        for spec in section.fields:
            field = by_key.get((section.id, spec.id))
            out.append(f"### {spec.label}")
            value = field.value if field else None
            out.extend(_value_lines(spec, value))
            out.append("")
            if field and field.evidence_json:
                cites = ", ".join(
                    f"[{format_timestamp(e['start'])}]" for e in field.evidence_json[:6]
                )
                out.append(f"_Evidence: {cites}_")
                out.append("")
            if field:
                label = _SOURCE_LABEL.get(field.source, str(field.source))
                if field.edited:
                    label = "edited by the manager"
                out.append(f"<sub>{label}</sub>")
                out.append("")

    out.append("## Provenance")
    out.append("")
    out.append(f"- Speech to text: {conversation.speech_provider or '—'} ({conversation.speech_model or '—'})")
    out.append(f"- Alignment and drafting: {conversation.analysis_provider or '—'} ({conversation.analysis_model or '—'})")
    if analysis.get("degraded_reason"):
        out.append(f"- Fell back to offline drafting: {analysis['degraded_reason']}")
    edited = [f for f in fields if f.edited]
    out.append(f"- Fields changed by the manager: {len(edited)} of {len(fields)}")
    if edited:
        for field in edited:
            spec = template.field(field.section_id, field.field_id)
            out.append(f"  - {spec.label if spec else field.field_id} ({field.section_id})")
    out.append("")

    if include_transcript and segments:
        out.append("## Transcript")
        out.append("")
        current = None
        for seg in segments:
            if seg.section_id != current:
                spec = template.section(seg.section_id or "")
                out.append("")
                out.append(f"**{spec.title if spec else seg.section_id or 'Unassigned'}**")
                out.append("")
                current = seg.section_id
            role = getattr(seg.speaker_role, "value", str(seg.speaker_role))
            who = {"manager": conversation.manager_name or "Manager",
                   "report": conversation.report_name or "Person"}.get(role, seg.speaker_label or "Speaker")
            out.append(f"`{format_timestamp(seg.start)}` **{who}:** {seg.text}")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def to_dict(
    conversation: Conversation,
    template: ConversationTemplate,
    fields: Sequence[FormField],
    segments: Sequence[Any] = (),
) -> dict[str, Any]:
    by_key = {(f.section_id, f.field_id): f for f in fields}
    return {
        "conversation": {
            "id": conversation.id,
            "title": conversation.title,
            "template_id": conversation.template_id,
            "template_name": template.name,
            "manager": conversation.manager_name,
            "person": conversation.report_name,
            "occurred_on": conversation.occurred_on,
            "created_at": conversation.created_at.isoformat(),
            "duration_seconds": conversation.duration_seconds,
            "audio_filename": conversation.audio_filename,
            "consent_confirmed": conversation.consent_confirmed,
            "status": conversation.status.value,
        },
        "provenance": {
            "speech_provider": conversation.speech_provider,
            "speech_model": conversation.speech_model,
            "analysis_provider": conversation.analysis_provider,
            "analysis_model": conversation.analysis_model,
            "language": conversation.language,
        },
        "metrics": conversation.analysis_json or {},
        "sections": [
            {
                "id": section.id,
                "title": section.title,
                "kind": section.kind,
                "planned_minutes": section.minutes,
                "prompt": section.prompt,
                "fields": [
                    {
                        "id": spec.id,
                        "label": spec.label,
                        "kind": spec.kind,
                        "value": (by_key.get((section.id, spec.id)).value
                                  if (section.id, spec.id) in by_key else None),
                        "draft_value": (by_key.get((section.id, spec.id)).draft_value
                                        if (section.id, spec.id) in by_key else None),
                        "edited": bool((section.id, spec.id) in by_key
                                       and by_key[(section.id, spec.id)].edited),
                        "source": (by_key[(section.id, spec.id)].source.value
                                   if (section.id, spec.id) in by_key else None),
                        "confidence": (by_key[(section.id, spec.id)].confidence
                                       if (section.id, spec.id) in by_key else None),
                        "evidence": (by_key[(section.id, spec.id)].evidence_json
                                     if (section.id, spec.id) in by_key else None),
                    }
                    for spec in section.fields
                ],
            }
            for section in template.sections
        ],
        "transcript": [
            {
                "index": seg.index,
                "start": seg.start,
                "end": seg.end,
                "speaker_role": getattr(seg.speaker_role, "value", str(seg.speaker_role)),
                "speaker_label": seg.speaker_label,
                "section_id": seg.section_id,
                "text": seg.text,
            }
            for seg in segments
        ],
    }
