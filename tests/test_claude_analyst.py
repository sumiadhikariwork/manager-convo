"""The Claude analyst, driven by a fake client.

These tests are about the contract around the model, not the model: whatever
comes back is repaired into something monotonic, in range, and honest - and
quote text is never taken from the response.
"""

from __future__ import annotations

import pytest

from app.analysis.claude import (
    ClaudeAnalyst,
    DraftAction,
    DraftedFieldOutput,
    DraftOutput,
    SectionBoundary,
    SpeakerTurn,
    StructureOutput,
    _render_transcript,
)
from app.analysis.types import AnalysisError


class _Response:
    def __init__(self, parsed, stop_reason="end_turn", stop_details=None):
        self.parsed_output = parsed
        self.stop_reason = stop_reason
        self.stop_details = stop_details


class _Messages:
    def __init__(self, responses, recorder):
        self._responses = list(responses)
        self._recorder = recorder

    def parse(self, **kwargs):
        self._recorder.append(kwargs)
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeClient:
    def __init__(self, *responses):
        self.calls: list[dict] = []
        self.messages = _Messages(responses, self.calls)


def _analyst(*responses) -> tuple[ClaudeAnalyst, FakeClient]:
    client = FakeClient(*responses)
    return ClaudeAnalyst(model="claude-opus-5", client=client), client


# -- structure -------------------------------------------------------------

def test_boundaries_become_a_per_segment_assignment(transcript, template):
    analyst, _ = _analyst(_Response(StructureOutput(
        boundaries=[
            SectionBoundary(section_id="open", first_segment=0, confidence=0.9),
            SectionBoundary(section_id="goals", first_segment=5, confidence=0.85),
            SectionBoundary(section_id="reality", first_segment=15, confidence=0.8),
            SectionBoundary(section_id="way_forward", first_segment=27, confidence=0.9),
        ],
        speaker_turns=[SpeakerTurn(first_segment=0, role="manager"), SpeakerTurn(first_segment=1, role="report")],
        notes="Clear hand-offs throughout.",
    )))
    result = analyst.structure(transcript.segments, template, "Priya", "Sam", transcript.duration)

    assert len(result.section_ids) == len(transcript.segments)
    assert result.section_ids[0] == "open"
    assert result.section_ids[5] == "goals"
    assert result.section_ids[15] == "reality"
    assert result.section_ids[-1] == "way_forward"
    assert result.method == "claude"
    assert result.notes == "Clear hand-offs throughout."


def test_out_of_order_boundaries_are_repaired_to_agenda_order(transcript, template):
    """The agenda cannot run backwards, whatever the response says."""
    analyst, _ = _analyst(_Response(StructureOutput(
        boundaries=[
            SectionBoundary(section_id="way_forward", first_segment=8, confidence=0.5),
            SectionBoundary(section_id="open", first_segment=0, confidence=0.5),
            SectionBoundary(section_id="reality", first_segment=4, confidence=0.5),
            SectionBoundary(section_id="goals", first_segment=20, confidence=0.5),
        ],
        speaker_turns=[SpeakerTurn(first_segment=0, role="manager")],
        notes="",
    )))
    result = analyst.structure(transcript.segments, template, duration=transcript.duration)

    order = [s.id for s in template.agenda]
    seen = [result.section_ids[0]]
    for section_id in result.section_ids[1:]:
        if section_id != seen[-1]:
            seen.append(section_id)
    assert seen == [s for s in order if s in seen]
    assert seen == sorted(seen, key=order.index)


def test_out_of_range_boundaries_are_clamped(transcript, template):
    analyst, _ = _analyst(_Response(StructureOutput(
        boundaries=[
            SectionBoundary(section_id="open", first_segment=-5, confidence=0.5),
            SectionBoundary(section_id="goals", first_segment=4, confidence=0.5),
            SectionBoundary(section_id="reality", first_segment=9999, confidence=0.5),
            SectionBoundary(section_id="way_forward", first_segment=10001, confidence=0.5),
        ],
        speaker_turns=[SpeakerTurn(first_segment=0, role="manager")],
        notes="",
    )))
    result = analyst.structure(transcript.segments, template, duration=transcript.duration)
    assert len(result.section_ids) == len(transcript.segments)
    assert result.section_ids[0] == "open"


def test_unknown_section_ids_are_ignored(transcript, template):
    analyst, _ = _analyst(_Response(StructureOutput(
        boundaries=[
            SectionBoundary(section_id="open", first_segment=0, confidence=0.6),
            SectionBoundary(section_id="coffee_break", first_segment=3, confidence=0.9),
            SectionBoundary(section_id="goals", first_segment=6, confidence=0.6),
            SectionBoundary(section_id="reality", first_segment=16, confidence=0.6),
            SectionBoundary(section_id="way_forward", first_segment=28, confidence=0.6),
        ],
        speaker_turns=[SpeakerTurn(first_segment=0, role="manager")],
        notes="",
    )))
    result = analyst.structure(transcript.segments, template, duration=transcript.duration)
    assert "coffee_break" not in result.section_ids


def test_no_usable_boundaries_is_an_error_not_a_guess(transcript, template):
    analyst, _ = _analyst(_Response(StructureOutput(boundaries=[], speaker_turns=[], notes="")))
    with pytest.raises(AnalysisError):
        analyst.structure(transcript.segments, template, duration=transcript.duration)


def test_speaker_turns_fill_forward(transcript, template):
    analyst, _ = _analyst(_Response(StructureOutput(
        boundaries=[SectionBoundary(section_id=s.id, first_segment=i * 8, confidence=0.7)
                    for i, s in enumerate(template.agenda)],
        speaker_turns=[
            SpeakerTurn(first_segment=0, role="manager"),
            SpeakerTurn(first_segment=2, role="report"),
            SpeakerTurn(first_segment=5, role="manager"),
        ],
        notes="",
    )))
    result = analyst.structure(transcript.segments, template, duration=transcript.duration)
    assert result.speaker_roles[0] == "manager"
    assert result.speaker_roles[1] == "manager"
    assert result.speaker_roles[2] == "report"
    assert result.speaker_roles[4] == "report"
    assert result.speaker_roles[5] == "manager"
    assert result.speaker_roles[-1] == "manager"


def test_a_refusal_is_surfaced_not_swallowed(transcript, template):
    class _Details:
        category = "reasoning_extraction"

    analyst, _ = _analyst(_Response(None, stop_reason="refusal", stop_details=_Details()))
    with pytest.raises(AnalysisError, match="declined"):
        analyst.structure(transcript.segments, template, duration=transcript.duration)


def test_transport_failures_become_analysis_errors(transcript, template):
    analyst, _ = _analyst(RuntimeError("connection reset"))
    with pytest.raises(AnalysisError, match="connection reset"):
        analyst.structure(transcript.segments, template, duration=transcript.duration)


def test_empty_transcript_never_reaches_the_api(template):
    analyst, client = _analyst()
    with pytest.raises(AnalysisError):
        analyst.structure([], template)
    assert client.calls == []


def test_request_uses_the_configured_model_and_adaptive_thinking(transcript, template):
    analyst, client = _analyst(_Response(StructureOutput(
        boundaries=[SectionBoundary(section_id="open", first_segment=0, confidence=0.5)],
        speaker_turns=[SpeakerTurn(first_segment=0, role="manager")],
        notes="",
    )))
    analyst.structure(transcript.segments, template, duration=transcript.duration)
    call = client.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_format"] is StructureOutput


def test_transcript_is_rendered_with_indices_the_model_can_cite(transcript):
    rendered = _render_transcript(transcript.segments[:3])
    assert rendered.startswith("[0] 0:02 Priya:")
    assert "[1] " in rendered


# -- drafting --------------------------------------------------------------

def _structure(transcript, template):
    from app.analysis.heuristic import build_structure

    return build_structure(transcript.segments, template, transcript.duration, "Priya", "Sam")


def test_values_are_coerced_to_the_kind_the_template_declares(transcript, template):
    structure = _structure(transcript, template)
    analyst, _ = _analyst(_Response(DraftOutput(fields=[
        DraftedFieldOutput(
            section_id="goals", field_id="notes",
            text="  Two of four goals moved.  ", items=["ignored"], actions=[],
            evidence=[6, 7], confidence=0.8,
        ),
        DraftedFieldOutput(
            section_id="reality", field_id="blockers",
            text="ignored", items=["Design capacity", "", "  No analytics access  "], actions=[],
            evidence=[16], confidence=0.9,
        ),
        DraftedFieldOutput(
            section_id="way_forward", field_id="actions",
            text="", items=[],
            actions=[
                DraftAction(action="Get analytics access", owner="Priya", due="Friday", support=""),
                DraftAction(action="   ", owner="Sam", due="", support=""),
            ],
            evidence=[28], confidence=0.85,
        ),
        DraftedFieldOutput(
            section_id="record", field_id="tone",
            text="Supportive", items=[], actions=[], evidence=[], confidence=0.6,
        ),
    ])))

    draft = analyst.draft(transcript.segments, structure, template, "Priya", "Sam")
    by_key = {(f.section_id, f.field_id): f for f in draft.fields}

    assert by_key[("goals", "notes")].value == "Two of four goals moved."
    assert by_key[("reality", "blockers")].value == ["Design capacity", "No analytics access"]
    assert by_key[("way_forward", "actions")].value == [
        {"action": "Get analytics access", "owner": "Priya", "due": "Friday", "support": ""}
    ]
    assert by_key[("record", "tone")].value == "supportive"
    assert draft.method == "claude"


def test_a_choice_outside_the_allowed_set_is_dropped(transcript, template):
    structure = _structure(transcript, template)
    analyst, _ = _analyst(_Response(DraftOutput(fields=[
        DraftedFieldOutput(section_id="record", field_id="tone", text="jubilant",
                           items=[], actions=[], evidence=[], confidence=0.9),
    ])))
    draft = analyst.draft(transcript.segments, structure, template)
    tone = next(f for f in draft.fields if f.field_id == "tone")
    assert tone.value == ""


def test_missing_fields_come_back_empty_rather_than_absent(transcript, template):
    structure = _structure(transcript, template)
    analyst, _ = _analyst(_Response(DraftOutput(fields=[])))
    draft = analyst.draft(transcript.segments, structure, template)

    expected = {(s.id, f.id) for s in template.sections for f in s.fields}
    assert {(f.section_id, f.field_id) for f in draft.fields} == expected
    for field in draft.fields:
        assert field.value in ("", [])
        assert field.confidence == 0.0


def test_evidence_indices_outside_the_transcript_are_discarded(transcript, template):
    structure = _structure(transcript, template)
    analyst, _ = _analyst(_Response(DraftOutput(fields=[
        DraftedFieldOutput(section_id="goals", field_id="notes", text="Something.",
                           items=[], actions=[], evidence=[-3, 2, 99999, 2], confidence=0.7),
    ])))
    draft = analyst.draft(transcript.segments, structure, template)
    notes = next(f for f in draft.fields if (f.section_id, f.field_id) == ("goals", "notes"))
    assert notes.evidence == [2], "out-of-range and duplicate citations must be dropped"


def test_confidence_is_clamped(transcript, template):
    structure = _structure(transcript, template)
    analyst, _ = _analyst(_Response(DraftOutput(fields=[
        DraftedFieldOutput(section_id="goals", field_id="notes", text="x", items=[], actions=[],
                           evidence=[], confidence=17.0),
    ])))
    draft = analyst.draft(transcript.segments, structure, template)
    notes = next(f for f in draft.fields if (f.section_id, f.field_id) == ("goals", "notes"))
    assert notes.confidence == 1.0


def test_the_draft_prompt_carries_the_section_split_and_the_form_spec(transcript, template):
    structure = _structure(transcript, template)
    analyst, client = _analyst(_Response(DraftOutput(fields=[])))
    analyst.draft(transcript.segments, structure, template, "Priya", "Sam")

    prompt = client.calls[0]["messages"][0]["content"]
    assert "# Form specification" in prompt
    assert "=== Goals (goals) ===" in prompt
    assert "MANAGER (Priya)" in prompt
    assert "REPORT (Sam)" in prompt
    assert "field_id: strength_named" in prompt
