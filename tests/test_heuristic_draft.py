"""The offline drafter must never write anything that was not said."""

from __future__ import annotations

from app.analysis.heuristic import build_structure, draft_fields
from app.util import normalise


def _draft(transcript, template):
    structure = build_structure(transcript.segments, template, transcript.duration, "Priya", "Sam")
    return structure, draft_fields(transcript.segments, structure, template, "Priya", "Sam")


def _field(draft, section_id, field_id):
    return next(f for f in draft.fields if f.section_id == section_id and f.field_id == field_id)


def test_every_template_field_is_drafted(transcript, template):
    _, draft = _draft(transcript, template)
    expected = {(s.id, f.id) for s in template.sections for f in s.fields}
    assert {(f.section_id, f.field_id) for f in draft.fields} == expected


def test_drafted_text_is_verbatim_from_the_transcript(transcript, template):
    """The offline path quotes; it does not compose. Nothing may be invented."""
    _, draft = _draft(transcript, template)
    spoken = normalise(" ".join(s.text for s in transcript.segments)).lower()

    for field in draft.fields:
        if field.field_id == "tone":  # a counted judgement, not an excerpt
            continue
        candidates: list[str] = []
        if isinstance(field.value, str):
            candidates = [field.value]
        elif isinstance(field.value, list):
            for item in field.value:
                candidates.append(item if isinstance(item, str) else item.get("action", ""))

        for candidate in candidates:
            # Multi-quote fields put one excerpt per line.
            for line in candidate.split("\n"):
                line = normalise(line.replace("“", "").replace("”", "")).rstrip("…").lower()
                if len(line) < 12:
                    continue
                assert line in spoken, f"{field.section_id}.{field.field_id} is not verbatim: {line!r}"


def test_evidence_indices_are_real_segments(transcript, template):
    _, draft = _draft(transcript, template)
    for field in draft.fields:
        for index in field.evidence:
            assert 0 <= index < len(transcript.segments)


def test_specific_praise_is_picked_up(transcript, template):
    _, draft = _draft(transcript, template)
    strength = _field(draft, "goals", "strength_named")
    assert "migration weekend was excellent" in strength.value
    assert strength.evidence


def test_blockers_come_from_the_person_being_coached(transcript, template):
    structure, draft = _draft(transcript, template)
    blockers = _field(draft, "reality", "blockers")
    assert blockers.value, "the person named blockers out loud"
    assert any("blocked on design" in item for item in blockers.value)
    for index in blockers.evidence:
        assert structure.speaker_roles[index] in ("report", "unknown")


def test_commitments_are_attributed_to_a_named_person(transcript, template):
    _, draft = _draft(transcript, template)
    actions = _field(draft, "way_forward", "actions")
    assert actions.value
    owners = {a["owner"] for a in actions.value if a["owner"]}
    assert owners <= {"Priya", "Sam"}
    assert "Priya" in owners


def test_feedback_given_records_what_the_manager_actually_said(transcript, template):
    _, draft = _draft(transcript, template)
    feedback = _field(draft, "record", "feedback_given")
    assert any("migration" in item.lower() for item in feedback.value)


def test_confidence_stays_low_because_this_path_is_a_fallback(transcript, template):
    _, draft = _draft(transcript, template)
    assert draft.method == "heuristic"
    assert all(f.confidence <= 0.5 for f in draft.fields)


def test_empty_when_nothing_was_said_about_it(template):
    """A field with no supporting talk stays empty rather than being filled in."""
    from app.analysis.heuristic import build_structure as bs
    from app.transcription.base import TranscriptSegment

    segments = [
        TranscriptSegment(index=i, start=i * 5.0, end=i * 5.0 + 5, text="We talked about the weather.", speaker="Priya" if i % 2 else "Sam")
        for i in range(8)
    ]
    structure = bs(segments, template, 40.0, "Priya", "Sam")
    draft = draft_fields(segments, structure, template, "Priya", "Sam")
    strength = next(f for f in draft.fields if f.field_id == "strength_named")
    actions = next(f for f in draft.fields if f.field_id == "actions")
    assert strength.value == ""
    assert actions.value == []
