"""Conversation-level metrics derived from the aligned transcript.

These are counted, not judged: seconds spoken, questions asked, time each
agenda item actually took against what was planned.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.analysis.types import StructureResult
from app.templates import ConversationTemplate
from app.transcription.base import TranscriptSegment
from app.util import normalise


def _is_question(text: str) -> bool:
    return normalise(text).endswith("?")


def compute_metrics(
    segments: Sequence[TranscriptSegment],
    structure: StructureResult,
    template: ConversationTemplate,
    duration: float | None = None,
) -> dict[str, Any]:
    total = duration or (max((s.end for s in segments), default=0.0))
    seconds: dict[str, float] = {"manager": 0.0, "report": 0.0, "unknown": 0.0}
    questions: dict[str, int] = {"manager": 0, "report": 0, "unknown": 0}
    words: dict[str, int] = {"manager": 0, "report": 0, "unknown": 0}

    for i, seg in enumerate(segments):
        role = structure.speaker_roles[i] if i < len(structure.speaker_roles) else "unknown"
        role = role if role in seconds else "unknown"
        seconds[role] += seg.duration
        words[role] += len(normalise(seg.text).split())
        if _is_question(seg.text):
            questions[role] += 1

    spoken = seconds["manager"] + seconds["report"] + seconds["unknown"]
    attributed = seconds["manager"] + seconds["report"]

    sections: list[dict[str, Any]] = []
    for spec in template.agenda:
        indices = [
            i
            for i in range(len(segments))
            if i < len(structure.section_ids) and structure.section_ids[i] == spec.id
        ]
        if indices:
            start = segments[indices[0]].start
            end = segments[indices[-1]].end
            actual = sum(segments[i].duration for i in indices)
        else:
            start = end = actual = 0.0
        sections.append(
            {
                "section_id": spec.id,
                "title": spec.title,
                "planned_seconds": spec.minutes * 60,
                "actual_seconds": round(end - start, 1) if indices else 0.0,
                "spoken_seconds": round(actual, 1),
                "start": round(start, 2),
                "end": round(end, 2),
                "segment_count": len(indices),
                "first_segment": indices[0] if indices else None,
                "last_segment": indices[-1] if indices else None,
            }
        )

    return {
        "duration_seconds": round(total, 1),
        "spoken_seconds": round(spoken, 1),
        "talk_ratio": {
            "manager": round(seconds["manager"] / attributed, 3) if attributed else None,
            "report": round(seconds["report"] / attributed, 3) if attributed else None,
            "unattributed_seconds": round(seconds["unknown"], 1),
        },
        "seconds_by_role": {k: round(v, 1) for k, v in seconds.items()},
        "words_by_role": words,
        "questions_by_role": questions,
        "segment_count": len(segments),
        "sections": sections,
        "alignment_method": structure.method,
        "alignment_notes": structure.notes,
        "mean_section_confidence": (
            round(sum(structure.section_confidence) / len(structure.section_confidence), 3)
            if structure.section_confidence
            else 0.0
        ),
    }
