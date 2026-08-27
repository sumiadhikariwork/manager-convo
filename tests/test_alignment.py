"""The offline aligner: does each stretch of talk land on the right agenda item?"""

from __future__ import annotations


from app.analysis.heuristic import align_segments, build_structure, infer_speaker_roles
from app.templates import GROW_MONTHLY, SBI_FEEDBACK
from app.transcription.base import TranscriptSegment


def _boundaries(section_ids: list[str]) -> dict[str, int]:
    """First segment index of each section, in order of first appearance."""
    starts: dict[str, int] = {}
    for i, section_id in enumerate(section_ids):
        starts.setdefault(section_id, i)
    return starts


def test_every_segment_is_assigned(transcript, template):
    section_ids, confidence = align_segments(transcript.segments, template, transcript.duration)
    assert len(section_ids) == len(transcript.segments)
    assert len(confidence) == len(transcript.segments)
    assert all(section_ids)


def test_assignment_is_monotonic_and_covers_the_agenda(transcript, template):
    """The agenda runs in order and does not repeat, so neither may the output."""
    section_ids, _ = align_segments(transcript.segments, template, transcript.duration)
    order = [s.id for s in template.agenda]
    seen = [section_ids[0]]
    for section_id in section_ids[1:]:
        if section_id != seen[-1]:
            seen.append(section_id)
    assert seen == order, "sections must appear once each, in agenda order"


def test_boundaries_land_on_the_spoken_cues(transcript, template):
    """The manager's hand-offs are audible; the aligner should find them."""
    section_ids, _ = align_segments(transcript.segments, template, transcript.duration)
    starts = _boundaries(section_ids)
    texts = [s.text.lower() for s in transcript.segments]

    # "Let's pull up the Goal Charter and walk it together."
    goals_cue = next(i for i, t in enumerate(texts) if "goal charter" in t)
    # "Let's move into what is actually in the way."
    reality_cue = next(i for i, t in enumerate(texts) if "in the way" in t)
    # "Let's agree the way forward."
    forward_cue = next(i for i, t in enumerate(texts) if "agree the way forward" in t)

    assert abs(starts["goals"] - goals_cue) <= 1
    assert abs(starts["reality"] - reality_cue) <= 2
    assert abs(starts["way_forward"] - forward_cue) <= 2


def test_confidence_is_reported_per_segment(transcript, template):
    _, confidence = align_segments(transcript.segments, template, transcript.duration)
    assert all(0.0 < c <= 1.0 for c in confidence)


def test_handles_fewer_segments_than_agenda_items(template):
    segments = [
        TranscriptSegment(index=0, start=0.0, end=2.0, text="Hello."),
        TranscriptSegment(index=1, start=2.0, end=4.0, text="Fine, thanks."),
    ]
    section_ids, confidence = align_segments(segments, template, 4.0)
    assert section_ids == ["open", "goals"]
    assert len(confidence) == 2


def test_handles_empty_transcript(template):
    assert align_segments([], template, 0.0) == ([], [])


def test_works_for_a_different_template(transcript):
    """Templates are data - the aligner must not be specific to one agenda."""
    section_ids, _ = align_segments(transcript.segments, SBI_FEEDBACK, transcript.duration)
    assert set(section_ids) == {s.id for s in SBI_FEEDBACK.agenda}


def test_record_sections_are_not_part_of_alignment(transcript):
    section_ids, _ = align_segments(transcript.segments, GROW_MONTHLY, transcript.duration)
    assert "record" not in section_ids


def test_manager_is_identified_from_the_speaker_labels(transcript):
    roles, confidence = infer_speaker_roles(transcript.segments, "Priya", "Sam")
    by_label = {
        seg.speaker: role for seg, role in zip(transcript.segments, roles)
    }
    assert by_label["Priya"] == "manager"
    assert by_label["Sam"] == "report"
    assert all(c > 0.5 for c in confidence)


def test_manager_is_inferred_when_no_names_are_given(transcript):
    """Without names, the coach is the one asking the questions."""
    roles, _ = infer_speaker_roles(transcript.segments)
    by_label = {seg.speaker: role for seg, role in zip(transcript.segments, roles)}
    assert by_label["Priya"] == "manager"


def test_unlabelled_speakers_are_reported_as_unknown_not_guessed():
    segments = [
        TranscriptSegment(index=i, start=float(i), end=float(i + 1), text="Something was said.")
        for i in range(4)
    ]
    roles, confidence = infer_speaker_roles(segments, "Priya", "Sam")
    assert roles == ["unknown"] * 4
    assert confidence == [0.0] * 4


def test_build_structure_returns_parallel_lists(transcript, template):
    structure = build_structure(transcript.segments, template, transcript.duration, "Priya", "Sam")
    n = len(transcript.segments)
    assert len(structure.section_ids) == n
    assert len(structure.section_confidence) == n
    assert len(structure.speaker_roles) == n
    assert len(structure.speaker_confidence) == n
    assert structure.method == "heuristic"
