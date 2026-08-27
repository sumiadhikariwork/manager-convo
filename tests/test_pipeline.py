"""End to end: an audio file goes in, a filled-in record comes out."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.analysis.types import AnalysisError
from app.db import SessionLocal
from app.models import AuditEvent, Conversation, FieldSource, FormField, ProcessingStatus, Segment
from app.pipeline import process_conversation
from app.templates import GROW_MONTHLY
from app.util import normalise


def _make_conversation(audio_path, template_id=GROW_MONTHLY.id) -> str:
    with SessionLocal() as session:
        conversation = Conversation(
            title="Monthly check-in",
            template_id=template_id,
            manager_name="Priya",
            report_name="Sam",
            consent_confirmed=True,
            audio_filename=audio_path.name,
            audio_path=str(audio_path),
            audio_mime="audio/wav",
            audio_bytes=audio_path.stat().st_size,
        )
        session.add(conversation)
        session.commit()
        return conversation.id


def _load(conversation_id):
    with SessionLocal() as session:
        conversation = session.get(Conversation, conversation_id)
        segments = list(session.scalars(
            select(Segment).where(Segment.conversation_id == conversation_id).order_by(Segment.index)))
        fields = list(session.scalars(
            select(FormField).where(FormField.conversation_id == conversation_id).order_by(FormField.order)))
        return conversation, segments, fields


def _events(conversation_id):
    with SessionLocal() as session:
        return list(session.scalars(
            select(AuditEvent).where(AuditEvent.conversation_id == conversation_id).order_by(AuditEvent.at)))


def test_a_full_run_reaches_ready(settings, audio_with_sidecar):
    conversation_id = _make_conversation(audio_with_sidecar)
    process_conversation(conversation_id, settings)

    conversation, segments, fields = _load(conversation_id)
    assert conversation.status is ProcessingStatus.READY
    assert conversation.error is None
    assert conversation.processed_at is not None
    assert len(segments) > 20
    assert len(fields) == sum(len(s.fields) for s in GROW_MONTHLY.sections)


def test_the_raw_transcript_is_kept_alongside_the_record(settings, audio_with_sidecar):
    conversation_id = _make_conversation(audio_with_sidecar)
    process_conversation(conversation_id, settings)

    conversation, segments, _ = _load(conversation_id)
    stored = conversation.transcript_json
    assert stored["provider"] == "fixture"
    assert len(stored["segments"]) == len(segments)
    assert stored["segments"][0]["text"] == segments[0].text


def test_segments_carry_their_agenda_item_and_speaker(settings, audio_with_sidecar):
    conversation_id = _make_conversation(audio_with_sidecar)
    process_conversation(conversation_id, settings)

    _, segments, _ = _load(conversation_id)
    assert {s.section_id for s in segments} == {sec.id for sec in GROW_MONTHLY.agenda}
    roles = {s.speaker_role.value for s in segments}
    assert roles == {"manager", "report"}
    assert all(s.start <= s.end for s in segments)


def test_every_citation_matches_the_stored_transcript(settings, audio_with_sidecar):
    """Evidence is resolved from the transcript, so quotes cannot drift from it."""
    conversation_id = _make_conversation(audio_with_sidecar)
    process_conversation(conversation_id, settings)

    _, segments, fields = _load(conversation_id)
    by_index = {s.index: s for s in segments}
    checked = 0
    for field in fields:
        for evidence in field.evidence_json or []:
            segment = by_index[evidence["segment_index"]]
            assert normalise(evidence["quote"]).rstrip("…") in normalise(segment.text)
            assert evidence["start"] == pytest.approx(segment.start, abs=0.01)
            checked += 1
    assert checked > 5


def test_metrics_are_counted_from_the_aligned_transcript(settings, audio_with_sidecar):
    conversation_id = _make_conversation(audio_with_sidecar)
    process_conversation(conversation_id, settings)

    conversation, _, _ = _load(conversation_id)
    metrics = conversation.analysis_json
    assert 0.0 < metrics["talk_ratio"]["manager"] < 1.0
    assert metrics["questions_by_role"]["manager"] > 0
    assert len(metrics["sections"]) == len(GROW_MONTHLY.agenda)
    assert all(s["segment_count"] > 0 for s in metrics["sections"])


def test_a_missing_transcript_fails_the_record_rather_than_the_process(settings, silent_wav):
    """No sidecar and no model: the failure belongs on the record, visibly."""
    target = settings.audio_dir / "lonely.wav"
    target.write_bytes(silent_wav.read_bytes())
    conversation_id = _make_conversation(target)

    process_conversation(conversation_id, settings)

    conversation, segments, fields = _load(conversation_id)
    assert conversation.status is ProcessingStatus.FAILED
    assert "sidecar" in conversation.error
    assert segments == []
    assert any(e.action == "failed" for e in _events(conversation_id))


def test_the_run_degrades_to_the_offline_analyst_when_claude_is_unavailable(
    settings, audio_with_sidecar, monkeypatch
):
    class Unavailable:
        name = "claude"
        model = "claude-opus-5"

        def structure(self, *args, **kwargs):
            raise AnalysisError("Claude request failed: no credentials")

        def draft(self, *args, **kwargs):
            raise AnalysisError("Claude request failed: no credentials")

    monkeypatch.setattr("app.pipeline.get_analyst", lambda *_: Unavailable())
    conversation_id = _make_conversation(audio_with_sidecar)
    process_conversation(conversation_id, settings)

    conversation, _, fields = _load(conversation_id)
    assert conversation.status is ProcessingStatus.READY, "a missing model must not lose the record"
    assert conversation.analysis_provider == "heuristic"
    assert "no credentials" in conversation.analysis_json["degraded_reason"]
    assert all(f.source is FieldSource.HEURISTIC for f in fields)


def test_reprocessing_keeps_what_the_manager_wrote(settings, audio_with_sidecar):
    conversation_id = _make_conversation(audio_with_sidecar)
    process_conversation(conversation_id, settings)

    with SessionLocal() as session:
        field = session.scalars(
            select(FormField).where(
                FormField.conversation_id == conversation_id,
                FormField.field_id == "strength_named",
            )
        ).one()
        field.value = "Wrote the rollback plan before anyone asked."
        field.edited = True
        field.source = FieldSource.MANAGER
        field_id = field.id
        session.commit()

    process_conversation(conversation_id, settings)

    with SessionLocal() as session:
        field = session.get(FormField, field_id)
        assert field.value == "Wrote the rollback plan before anyone asked."
        assert field.edited is True
        assert field.source is FieldSource.MANAGER
        # The fresh draft is still recorded next to it, for comparison.
        assert field.draft_value != field.value


def test_reprocessing_refreshes_untouched_fields(settings, audio_with_sidecar):
    conversation_id = _make_conversation(audio_with_sidecar)
    process_conversation(conversation_id, settings)

    with SessionLocal() as session:
        field = session.scalars(
            select(FormField).where(
                FormField.conversation_id == conversation_id, FormField.field_id == "headline"
            )
        ).one()
        field.value = "stale"
        field_id = field.id
        session.commit()

    process_conversation(conversation_id, settings)

    with SessionLocal() as session:
        assert session.get(FormField, field_id).value != "stale"


def test_the_audit_trail_records_each_stage(settings, audio_with_sidecar):
    conversation_id = _make_conversation(audio_with_sidecar)
    process_conversation(conversation_id, settings)

    actions = [e.action for e in _events(conversation_id)]
    assert actions == ["transcribed", "aligned", "drafted"]


def test_a_different_template_produces_that_template_s_fields(settings, audio_with_sidecar):
    from app.templates import SBI_FEEDBACK

    conversation_id = _make_conversation(audio_with_sidecar, template_id=SBI_FEEDBACK.id)
    process_conversation(conversation_id, settings)

    _, segments, fields = _load(conversation_id)
    assert {(f.section_id, f.field_id) for f in fields} == {
        (s.id, f.id) for s in SBI_FEEDBACK.sections for f in s.fields
    }
    assert {s.section_id for s in segments} == {s.id for s in SBI_FEEDBACK.agenda}
