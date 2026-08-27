"""Claude-backed analysis: agenda alignment, speaker attribution, form drafting.

Two calls per conversation, each with a structured output schema:

1. **Structure** - where each agenda item begins, and which speaker is the
   manager. Small output: boundary indices, not a re-run of the transcript.
2. **Draft** - the form fields themselves, each citing the segments it rests on.

The model never supplies quote text. It returns segment *indices*, and the
application resolves those to the stored transcript, so a citation on the
record is always something that was genuinely said.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from pydantic import BaseModel, Field

from app.analysis.types import AnalysisError, DraftedField, DraftResult, StructureResult
from app.templates import ConversationTemplate
from app.transcription.base import TranscriptSegment
from app.util import format_timestamp, normalise

logger = logging.getLogger(__name__)

MAX_TOKENS = 16000


# --------------------------------------------------------------------------
# Structured output schemas
# --------------------------------------------------------------------------

class SectionBoundary(BaseModel):
    section_id: str = Field(description="Agenda item id, exactly as given in the agenda.")
    first_segment: int = Field(description="Index of the first transcript segment of this agenda item.")
    confidence: float = Field(description="0.0-1.0 confidence that this is where the agenda item begins.")


class SpeakerTurn(BaseModel):
    first_segment: int = Field(description="Index of the segment where this speaker starts talking.")
    role: str = Field(description="Either 'manager' or 'report'.")


class StructureOutput(BaseModel):
    boundaries: list[SectionBoundary] = Field(
        description="One entry per agenda item, in agenda order, with increasing first_segment."
    )
    speaker_turns: list[SpeakerTurn] = Field(
        description=(
            "Speaker changes only. The role applies from first_segment until the next entry. "
            "The first entry must have first_segment 0."
        )
    )
    notes: str = Field(description="Anything about the alignment a reader should know. May be empty.")


class DraftAction(BaseModel):
    action: str = Field(description="The commitment, as agreed.")
    owner: str = Field(description="Person named in the conversation who owns it. Empty if never named.")
    due: str = Field(description="When it is due, as spoken. Empty if not agreed.")
    support: str = Field(description="Support the manager offered for it. Empty if none.")


class DraftedFieldOutput(BaseModel):
    section_id: str = Field(description="Section id from the form specification.")
    field_id: str = Field(description="Field id from the form specification.")
    text: str = Field(description="Value for a text or choice field. Empty for list and actions fields.")
    items: list[str] = Field(description="Values for a list field. Empty for other kinds.")
    actions: list[DraftAction] = Field(description="Values for an actions field. Empty for other kinds.")
    evidence: list[int] = Field(
        description="Segment indices this answer rests on. Empty only when the answer is empty."
    )
    confidence: float = Field(description="0.0-1.0 confidence in this answer.")


class DraftOutput(BaseModel):
    fields: list[DraftedFieldOutput] = Field(description="One entry per field in the form specification.")


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

TRANSPARENCY_RULES = """\
This record is shared with the person who was coached. Accuracy matters more than completeness.

- Report only what was actually said. Never infer, embellish, soften or fill a gap.
- If something was not discussed, leave the value empty. An empty field is a correct answer; \
a plausible invention is not.
- Prefer the speakers' own words and their own framing. Keep figures, dates and names exactly as spoken.
- Describe what was said. Do not assess the person's character, attitude or potential.
- Record criticism as plainly as praise. Do not round either one off.
- Cite evidence by segment index only. Never write a quotation - the application resolves \
your indices against the stored transcript."""

STRUCTURE_SYSTEM = f"""\
You are aligning a recorded workplace coaching conversation to the agenda the manager was working from.

The agenda runs in order and does not repeat. Your job is to find where each agenda item begins.
Real conversations drift: an item may start early, run long, or be barely covered at all. Follow what \
actually happened in the room, not the planned timings - the timings are a hint, nothing more.

You also decide which speaker is the manager (the one coaching) and which is the report (the one being \
coached). The manager usually asks the questions, gives the feedback, and offers support.

{TRANSPARENCY_RULES}"""

DRAFT_SYSTEM = f"""\
You are filling in a manager's feedback form from the transcript of a recorded coaching conversation, \
so that the record of what was said is accurate and available to both people.

You are given the form specification and the transcript, already split by agenda item. Fill every field \
in the specification. Answer each one only from the part of the conversation it belongs to, except for \
fields in a section marked "whole conversation", which draw on everything.

{TRANSPARENCY_RULES}

Output rules:
- For a `text` field, put the answer in `text` and leave `items` and `actions` empty.
- For a `choice` field, put one of the listed choices in `text`.
- For a `list` field, put the answer in `items` and leave `text` empty.
- For an `actions` field, put the answer in `actions` and leave `text` and `items` empty.
- `evidence` lists the segment indices the answer rests on: the ones a reader would need in order to \
check you. Two or three is usually right. Leave it empty only when the answer itself is empty."""


def _render_transcript(
    segments: Sequence[TranscriptSegment], roles: Sequence[str] | None = None
) -> str:
    lines = []
    for i, seg in enumerate(segments):
        stamp = format_timestamp(seg.start)
        if roles is not None and i < len(roles) and roles[i] in ("manager", "report"):
            who = roles[i].upper()
        elif seg.speaker:
            who = seg.speaker
        else:
            who = "?"
        lines.append(f"[{i}] {stamp} {who}: {normalise(seg.text)}")
    return "\n".join(lines)


def _render_agenda(template: ConversationTemplate) -> str:
    return "\n".join(
        f"- {s.id} | {s.title} | planned {s.minutes} min | {s.prompt}" for s in template.agenda
    )


def _render_form_spec(template: ConversationTemplate) -> str:
    blocks = []
    for section in template.sections:
        scope = "whole conversation" if section.kind == "record" else f"agenda item '{section.id}'"
        blocks.append(f"## Section {section.id} — {section.title} ({scope})")
        for spec in section.fields:
            line = f"- field_id: {spec.id} | kind: {spec.kind} | label: {spec.label}"
            if spec.choices:
                line += f" | choices: {', '.join(spec.choices)}"
            if spec.guidance:
                line += f"\n  what to record: {spec.guidance}"
            blocks.append(line)
    return "\n".join(blocks)


# --------------------------------------------------------------------------
# Repair of model output
# --------------------------------------------------------------------------

def _boundaries_to_sections(
    boundaries: Sequence[SectionBoundary], template: ConversationTemplate, n_segments: int
) -> tuple[list[str], list[float]]:
    """Turn boundary indices into a per-segment assignment.

    The agenda is ordered and non-repeating, so the assignment is repaired to be
    monotonic no matter what came back.
    """
    order = [s.id for s in template.agenda]
    by_id = {b.section_id: b for b in boundaries if b.section_id in order}

    starts: list[tuple[int, str, float]] = []
    cursor = 0
    for section_id in order:
        boundary = by_id.get(section_id)
        if boundary is None:
            continue
        start = max(cursor, min(int(boundary.first_segment), max(n_segments - 1, 0)))
        confidence = min(max(float(boundary.confidence or 0.0), 0.0), 1.0)
        starts.append((start, section_id, confidence))
        cursor = start

    if not starts:
        raise AnalysisError("Alignment returned no usable agenda boundaries.")
    # The conversation has to start somewhere.
    starts[0] = (0, starts[0][1], starts[0][2])

    section_ids = [starts[0][1]] * n_segments
    confidence = [starts[0][2] or 0.5] * n_segments
    for idx, (start, section_id, conf) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else n_segments
        for i in range(start, end):
            section_ids[i] = section_id
            confidence[i] = conf or 0.5
    return section_ids, confidence


def _turns_to_roles(turns: Sequence[SpeakerTurn], n_segments: int) -> tuple[list[str], list[float]]:
    clean = sorted(
        (
            (max(0, min(int(t.first_segment), n_segments - 1)), t.role.strip().lower())
            for t in turns
            if t.role.strip().lower() in ("manager", "report")
        ),
        key=lambda row: row[0],
    )
    if not clean:
        return ["unknown"] * n_segments, [0.0] * n_segments

    roles = ["unknown"] * n_segments
    confidence = [0.0] * n_segments
    current = clean[0][1]
    pointer = 0
    for i in range(n_segments):
        while pointer < len(clean) and clean[pointer][0] <= i:
            current = clean[pointer][1]
            pointer += 1
        roles[i] = current
        confidence[i] = 0.8
    return roles, confidence


def _coerce_value(kind: str, out: DraftedFieldOutput, choices: Sequence[str]) -> Any:
    if kind == "actions":
        return [
            {
                "action": normalise(a.action),
                "owner": normalise(a.owner),
                "due": normalise(a.due),
                "support": normalise(a.support),
            }
            for a in out.actions
            if normalise(a.action)
        ]
    if kind == "list":
        return [normalise(item) for item in out.items if normalise(item)]
    if kind == "choice":
        value = normalise(out.text).lower()
        return value if not choices or value in choices else ""
    return normalise(out.text)


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------

class ClaudeAnalyst:
    """Alignment and drafting through the Claude API."""

    name = "claude"

    def __init__(self, model: str = "claude-opus-5", api_key: str = "", client: Any = None):
        self.model = model
        self._client = client
        self._api_key = api_key

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise AnalysisError("The `anthropic` package is not installed.") from exc
            try:
                # An unset ANTHROPIC_API_KEY is not the same as no credentials:
                # the SDK also resolves ANTHROPIC_AUTH_TOKEN and an `ant auth
                # login` profile, so let it do its own resolution when we have
                # nothing explicit to hand it.
                self._client = (
                    anthropic.Anthropic(api_key=self._api_key)
                    if self._api_key
                    else anthropic.Anthropic()
                )
            except Exception as exc:
                raise AnalysisError(f"Could not create an Anthropic client: {exc}") from exc
        return self._client

    def _parse(self, *, system: str, prompt: str, output_format: type[BaseModel]):
        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=MAX_TOKENS,
                thinking={"type": "adaptive"},
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_format=output_format,
            )
        except Exception as exc:
            raise AnalysisError(f"Claude request failed: {exc}") from exc

        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise AnalysisError(f"Claude declined to process this conversation ({category}).")

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise AnalysisError("Claude returned no structured output.")
        return parsed

    # -- step 1 ------------------------------------------------------------
    def structure(
        self,
        segments: Sequence[TranscriptSegment],
        template: ConversationTemplate,
        manager_name: str = "",
        report_name: str = "",
        duration: float | None = None,
    ) -> StructureResult:
        if not segments:
            raise AnalysisError("Nothing to align: the transcript is empty.")

        who = []
        if manager_name:
            who.append(f"The manager is {manager_name}.")
        if report_name:
            who.append(f"The person being coached is {report_name}.")

        prompt = (
            f"# Agenda ({template.name}, planned {template.planned_minutes} min)\n"
            f"{_render_agenda(template)}\n\n"
            f"# Recording\n"
            f"Length: {format_timestamp(duration)}. Segments: {len(segments)} (indices 0-{len(segments) - 1}).\n"
            + (" ".join(who) + "\n" if who else "")
            + "\n# Transcript\n"
            f"{_render_transcript(segments)}\n\n"
            "Return one boundary per agenda item, in agenda order, with the index of the segment "
            "where that item begins. The first boundary starts at segment 0. Then return the speaker "
            "turns: an entry each time the speaker changes, starting at segment 0."
        )

        parsed: StructureOutput = self._parse(
            system=STRUCTURE_SYSTEM, prompt=prompt, output_format=StructureOutput
        )
        section_ids, section_conf = _boundaries_to_sections(parsed.boundaries, template, len(segments))
        roles, role_conf = _turns_to_roles(parsed.speaker_turns, len(segments))
        return StructureResult(
            section_ids=section_ids,
            section_confidence=section_conf,
            speaker_roles=roles,
            speaker_confidence=role_conf,
            method="claude",
            notes=normalise(parsed.notes),
        )

    # -- step 2 ------------------------------------------------------------
    def draft(
        self,
        segments: Sequence[TranscriptSegment],
        structure: StructureResult,
        template: ConversationTemplate,
        manager_name: str = "",
        report_name: str = "",
    ) -> DraftResult:
        if not segments:
            raise AnalysisError("Nothing to draft from: the transcript is empty.")

        section_titles = {s.id: s.title for s in template.agenda}
        marked_lines = []
        previous = None
        for i, seg in enumerate(segments):
            section_id = structure.section_ids[i] if i < len(structure.section_ids) else ""
            if section_id != previous:
                marked_lines.append(f"\n=== {section_titles.get(section_id, section_id)} ({section_id}) ===")
                previous = section_id
            role = structure.speaker_roles[i] if i < len(structure.speaker_roles) else "unknown"
            who = role.upper() if role in ("manager", "report") else (seg.speaker or "?")
            if role == "manager" and manager_name:
                who = f"MANAGER ({manager_name})"
            elif role == "report" and report_name:
                who = f"REPORT ({report_name})"
            marked_lines.append(f"[{i}] {format_timestamp(seg.start)} {who}: {normalise(seg.text)}")

        prompt = (
            f"# Form specification\n{_render_form_spec(template)}\n\n"
            f"# Transcript, split by agenda item\n{chr(10).join(marked_lines).strip()}\n\n"
            "Fill in every field in the specification, in the order it is listed."
        )

        parsed: DraftOutput = self._parse(
            system=DRAFT_SYSTEM, prompt=prompt, output_format=DraftOutput
        )

        by_key = {(f.section_id, f.field_id): f for f in parsed.fields}
        drafted: list[DraftedField] = []
        n = len(segments)
        for section in template.sections:
            for spec in section.fields:
                out = by_key.get((section.id, spec.id))
                if out is None:
                    drafted.append(
                        DraftedField(
                            section_id=section.id,
                            field_id=spec.id,
                            value=[] if spec.kind in ("list", "actions") else "",
                            confidence=0.0,
                        )
                    )
                    continue
                evidence = sorted({i for i in out.evidence if isinstance(i, int) and 0 <= i < n})
                drafted.append(
                    DraftedField(
                        section_id=section.id,
                        field_id=spec.id,
                        value=_coerce_value(spec.kind, out, spec.choices),
                        confidence=min(max(float(out.confidence or 0.0), 0.0), 1.0),
                        evidence=evidence,
                    )
                )

        return DraftResult(fields=drafted, method="claude", model=self.model)
