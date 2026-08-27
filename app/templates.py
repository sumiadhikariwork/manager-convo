"""Conversation templates.

A template is the machine-readable version of the paper form a manager would
otherwise fill in by hand: an ordered agenda, how long each part is meant to
take, the coaching prompt for that part, and the fields the manager is expected
to record. The pipeline aligns the audio to the agenda and then drafts every
field in it.

Templates are plain data so an organisation can add its own without touching
the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FieldKind = Literal["text", "list", "actions", "choice", "ratio"]
SectionKind = Literal["agenda", "record"]


@dataclass(frozen=True)
class FormFieldSpec:
    """One box on the feedback form."""

    id: str
    label: str
    kind: FieldKind = "text"
    placeholder: str = ""
    #: Shown to the manager, and given to the model as the drafting brief.
    guidance: str = ""
    choices: tuple[str, ...] = ()
    required: bool = False


@dataclass(frozen=True)
class SectionSpec:
    """One agenda item, e.g. "Goals - 5 min"."""

    id: str
    title: str
    minutes: int
    prompt: str
    fields: tuple[FormFieldSpec, ...]
    kind: SectionKind = "agenda"
    #: Words that signal this part of the conversation has started. Used by the
    #: offline aligner; the Claude aligner reads the prompt instead.
    cues: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConversationTemplate:
    id: str
    name: str
    description: str
    sections: tuple[SectionSpec, ...]

    @property
    def agenda(self) -> tuple[SectionSpec, ...]:
        """Sections the audio is aligned against, in order."""
        return tuple(s for s in self.sections if s.kind == "agenda")

    @property
    def record_sections(self) -> tuple[SectionSpec, ...]:
        """Sections summarising the whole conversation rather than a slice."""
        return tuple(s for s in self.sections if s.kind == "record")

    @property
    def planned_minutes(self) -> int:
        return sum(s.minutes for s in self.agenda)

    def section(self, section_id: str) -> SectionSpec | None:
        for s in self.sections:
            if s.id == section_id:
                return s
        return None

    def field(self, section_id: str, field_id: str) -> FormFieldSpec | None:
        section = self.section(section_id)
        if section is None:
            return None
        for f in section.fields:
            if f.id == field_id:
                return f
        return None


NOTES_PLACEHOLDER = "Recorded by the coach, in the room."


GROW_MONTHLY = ConversationTemplate(
    id="grow-monthly-15",
    name="Monthly coaching conversation (GROW, 15 min)",
    description=(
        "The standing monthly one-to-one: open, review the Goal Charter, "
        "surface what is in the way, and agree the way forward."
    ),
    sections=(
        SectionSpec(
            id="open",
            title="Open",
            minutes=2,
            prompt="Set an easy tone. How has the month been? Let them speak first.",
            cues=(
                "how have you been", "how has the month", "how are things", "good to see you",
                "how was your", "thanks for making time", "how's it going", "how have things been",
                "before we start", "let's get started", "how are you doing",
            ),
            fields=(
                FormFieldSpec(
                    id="notes",
                    label="Notes",
                    kind="text",
                    placeholder=NOTES_PLACEHOLDER,
                    guidance=(
                        "How the person described their month in their own framing, and the "
                        "tone the conversation opened on. Two or three sentences."
                    ),
                ),
            ),
        ),
        SectionSpec(
            id="goals",
            title="Goals",
            minutes=5,
            prompt=(
                "Walk the Goal Charter together: what moved, what the numbers say. "
                "Name one specific thing done well."
            ),
            cues=(
                "goal charter", "target", "numbers", "the goal", "objective", "kpi",
                "quarter", "we said", "last month you", "you hit", "against plan",
                "pipeline", "metric", "progress", "moved",
            ),
            fields=(
                FormFieldSpec(
                    id="notes",
                    label="Notes",
                    kind="text",
                    placeholder=NOTES_PLACEHOLDER,
                    guidance=(
                        "What moved against the Goal Charter and what the numbers actually "
                        "said, as discussed. Keep figures exactly as spoken."
                    ),
                ),
                FormFieldSpec(
                    id="strength_named",
                    label="One specific thing done well",
                    kind="text",
                    guidance=(
                        "The single concrete piece of praise the coach gave, in the coach's "
                        "own terms. Leave empty if no specific praise was given - do not invent one."
                    ),
                ),
            ),
        ),
        SectionSpec(
            id="reality",
            title="Reality",
            minutes=4,
            prompt="What is in the way? What have you already tried? What would make it easier?",
            cues=(
                "in the way", "blocked", "blocker", "struggling", "difficult", "problem",
                "challenge", "i tried", "we tried", "didn't work", "hard part",
                "what would make it easier", "stuck", "frustrating", "obstacle",
            ),
            fields=(
                FormFieldSpec(
                    id="notes",
                    label="Notes",
                    kind="text",
                    placeholder=NOTES_PLACEHOLDER,
                    guidance="How the person described the situation and what is genuinely in the way.",
                ),
                FormFieldSpec(
                    id="blockers",
                    label="What is in the way",
                    kind="list",
                    guidance="Each obstacle named in the conversation, one per item, in plain language.",
                ),
                FormFieldSpec(
                    id="tried_already",
                    label="Already tried",
                    kind="list",
                    guidance="Things the person says they have already attempted, one per item.",
                ),
            ),
        ),
        SectionSpec(
            id="way_forward",
            title="Way Forward",
            minutes=4,
            prompt="Agree specific actions, owners, and the support you will arrange.",
            cues=(
                "so the plan", "next step", "action", "by friday", "by the end of",
                "i'll", "you'll", "let's agree", "who will", "i will set up",
                "support", "follow up", "check in", "deadline", "before we next meet",
            ),
            fields=(
                FormFieldSpec(
                    id="notes",
                    label="Notes",
                    kind="text",
                    placeholder=NOTES_PLACEHOLDER,
                    guidance="How the way forward was agreed, and anything left deliberately open.",
                ),
                FormFieldSpec(
                    id="actions",
                    label="Agreed actions",
                    kind="actions",
                    guidance=(
                        "Every commitment made out loud. Owner must be a person named in the "
                        "conversation. Leave due/support empty when they were not agreed."
                    ),
                ),
            ),
        ),
        SectionSpec(
            id="record",
            title="Conversation record",
            minutes=0,
            kind="record",
            prompt="Summary of the conversation as a whole, for the shared record.",
            fields=(
                FormFieldSpec(
                    id="headline",
                    label="Headline",
                    kind="text",
                    guidance="One sentence a third party could read to know what this conversation was about.",
                ),
                FormFieldSpec(
                    id="feedback_given",
                    label="Feedback given to the person",
                    kind="list",
                    guidance=(
                        "Every piece of feedback the coach actually delivered, praise and "
                        "criticism alike, phrased as the person would have heard it. This is "
                        "the transparency record - do not soften and do not add feedback that "
                        "was not given."
                    ),
                ),
                FormFieldSpec(
                    id="commitments_by_manager",
                    label="What the manager committed to",
                    kind="list",
                    guidance="Support or actions the coach took on themselves.",
                ),
                FormFieldSpec(
                    id="tone",
                    label="Tone",
                    kind="choice",
                    choices=("supportive", "neutral", "directive", "tense", "mixed"),
                    guidance="The overall tone of the exchange, judged from how both people spoke.",
                ),
                FormFieldSpec(
                    id="follow_up",
                    label="Follow-up agreed",
                    kind="text",
                    guidance="When and how they agreed to pick this up again. Empty if not agreed.",
                ),
            ),
        ),
    ),
)


SBI_FEEDBACK = ConversationTemplate(
    id="sbi-feedback-20",
    name="Direct feedback conversation (SBI, 20 min)",
    description=(
        "A focused feedback conversation using Situation - Behaviour - Impact, "
        "closing on what changes next."
    ),
    sections=(
        SectionSpec(
            id="context",
            title="Context",
            minutes=3,
            prompt="Say why you are meeting and check they are ready to hear it.",
            cues=("wanted to talk", "reason we're", "is now a good time", "want to raise"),
            fields=(
                FormFieldSpec(id="notes", label="Notes", kind="text", placeholder=NOTES_PLACEHOLDER,
                             guidance="How the conversation was framed and how the person received the framing."),
            ),
        ),
        SectionSpec(
            id="situation_behaviour",
            title="Situation & Behaviour",
            minutes=6,
            prompt="Describe the specific situation and the observable behaviour. No labels, no character.",
            cues=("on monday", "in the meeting", "last week", "when you", "i noticed", "what happened"),
            fields=(
                FormFieldSpec(id="notes", label="Notes", kind="text", placeholder=NOTES_PLACEHOLDER,
                             guidance="The situation and behaviour as described by the coach."),
                FormFieldSpec(id="observations", label="Observable behaviours cited", kind="list",
                              guidance="Each specific, observable behaviour the coach cited. Not interpretations."),
            ),
        ),
        SectionSpec(
            id="impact",
            title="Impact",
            minutes=5,
            prompt="Explain the impact, then stop and listen to their account.",
            cues=("the impact", "the effect", "meant that", "as a result", "how do you see it"),
            fields=(
                FormFieldSpec(id="notes", label="Notes", kind="text", placeholder=NOTES_PLACEHOLDER,
                             guidance="The impact described, and the person's own account of events."),
                FormFieldSpec(id="their_response", label="Their response", kind="text",
                              guidance="How the person responded, in their framing. Include disagreement plainly."),
            ),
        ),
        SectionSpec(
            id="change",
            title="What changes",
            minutes=6,
            prompt="Agree what will be different, by when, and what support you will give.",
            cues=("going forward", "next time", "what changes", "i'll support", "let's agree", "by "),
            fields=(
                FormFieldSpec(id="notes", label="Notes", kind="text", placeholder=NOTES_PLACEHOLDER,
                             guidance="How the change was agreed."),
                FormFieldSpec(id="actions", label="Agreed actions", kind="actions",
                              guidance="Every commitment made out loud, with the owner named in the conversation."),
            ),
        ),
        SectionSpec(
            id="record",
            title="Conversation record",
            minutes=0,
            kind="record",
            prompt="Summary of the conversation as a whole, for the shared record.",
            fields=(
                FormFieldSpec(id="headline", label="Headline", kind="text",
                              guidance="One sentence a third party could read to know what this was about."),
                FormFieldSpec(id="feedback_given", label="Feedback given to the person", kind="list",
                              guidance="Every piece of feedback actually delivered, in the words the person would have heard."),
                FormFieldSpec(id="commitments_by_manager", label="What the manager committed to", kind="list",
                              guidance="Support or actions the coach took on themselves."),
                FormFieldSpec(id="tone", label="Tone", kind="choice",
                              choices=("supportive", "neutral", "directive", "tense", "mixed"),
                              guidance="The overall tone of the exchange."),
                FormFieldSpec(id="follow_up", label="Follow-up agreed", kind="text",
                              guidance="When and how they agreed to pick this up again."),
            ),
        ),
    ),
)


TEMPLATES: dict[str, ConversationTemplate] = {
    t.id: t for t in (GROW_MONTHLY, SBI_FEEDBACK)
}

DEFAULT_TEMPLATE_ID = GROW_MONTHLY.id


def get_template(template_id: str | None) -> ConversationTemplate:
    """Look up a template, falling back to the default."""
    if not template_id:
        return TEMPLATES[DEFAULT_TEMPLATE_ID]
    try:
        return TEMPLATES[template_id]
    except KeyError as exc:
        raise KeyError(f"Unknown conversation template: {template_id!r}") from exc


def list_templates() -> list[ConversationTemplate]:
    return list(TEMPLATES.values())
