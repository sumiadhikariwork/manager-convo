"""Offline analysis: agenda alignment, speaker roles, and an extractive draft.

This path needs no API key and no network. It exists for two reasons: it is the
fallback when the Claude call is unavailable or declines, and it is the honest
floor for what the system can claim - everything it writes into the form is
lifted verbatim from the transcript, never generated.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Sequence

from app.analysis.types import DraftedField, DraftResult, StructureResult
from app.templates import ConversationTemplate, SectionSpec
from app.transcription.base import TranscriptSegment
from app.util import normalise, split_sentences, truncate

# How much the planned running order matters relative to the spoken cues.
TIME_PRIOR_WEIGHT = 1.4
CUE_WEIGHT = 1.0
#: Fraction of the timeline either side of a section's planned slot that still
#: counts as "roughly on schedule".
PRIOR_TOLERANCE = 0.18

QUESTION_RE = re.compile(r"\?\s*$")

COMMITMENT_RE = re.compile(
    r"\b(i'?ll|i will|we'?ll|we will|you'?ll|you will|let'?s|i'?m going to|"
    r"i can|i'?ll set up|by (?:monday|tuesday|wednesday|thursday|friday|next|the end)|"
    r"before we (?:next )?meet)\b",
    re.IGNORECASE,
)
BLOCKER_RE = re.compile(
    r"\b(blocked|blocker|in the way|stuck|struggl\w*|difficult|hard to|problem|"
    r"issue|slow(?:ing)? (?:me|us) down|waiting on|bottleneck|can'?t|couldn'?t|"
    r"no time|short(?:-| )staffed)\b",
    re.IGNORECASE,
)
TRIED_RE = re.compile(
    r"\b(i tried|we tried|i'?ve tried|we'?ve tried|i attempted|already (?:tried|did|"
    r"asked|spoke)|last time i|i did try|that didn'?t work)\b",
    re.IGNORECASE,
)
PRAISE_RE = re.compile(
    r"\b(well done|really good|great (?:job|work)|impressed|nice work|credit to you|"
    r"you (?:handled|did|nailed|managed)|strong(?:est)?|pleased|proud|thank you for)\b",
    re.IGNORECASE,
)
CRITIQUE_RE = re.compile(
    r"\b(needs? to (?:improve|change)|not (?:good )?enough|concerned|worried|"
    r"disappoint\w*|below|slipped|missed|has to change|i need you to|"
    r"that can'?t (?:keep )?happen\w*)\b",
    re.IGNORECASE,
)
SUPPORT_RE = re.compile(
    r"\b(?:i'?ll|i will|i am going to|i'?m going to|i can)\s+"
    r"(?:set up|arrange|talk to|speak to|go to|get|find|cover|introduce|ask|raise|chase|"
    r"come back|take|sort|unblock|free up|check)\b|"
    r"\bleave (?:that|it) with me\b|\bthat one is on me\b|\bthat is on me\b",
    re.IGNORECASE,
)
POSITIVE_TONE_RE = re.compile(
    r"\b(thanks|thank you|great|good|glad|happy|pleased|appreciate|well done|nice)\b", re.IGNORECASE
)
TENSE_TONE_RE = re.compile(
    r"\b(disagree|unfair|frustrat\w*|angry|upset|not acceptable|excuse|defensive|"
    r"that'?s not|no, )\b",
    re.IGNORECASE,
)
DIRECTIVE_TONE_RE = re.compile(
    r"\b(you need to|you must|i need you to|this has to|non-negotiable|expect(?:ation)?s? (?:are|is))\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Agenda alignment
# --------------------------------------------------------------------------

def _cue_scores(segments: Sequence[TranscriptSegment], sections: Sequence[SectionSpec]) -> list[list[float]]:
    scores: list[list[float]] = []
    for seg in segments:
        text = f" {seg.text.lower()} "
        row = []
        for section in sections:
            hits = sum(1 for cue in section.cues if cue in text)
            row.append(min(hits, 3) / 3.0)
        scores.append(row)
    return scores


def _time_prior(
    segments: Sequence[TranscriptSegment], sections: Sequence[SectionSpec], duration: float
) -> list[list[float]]:
    """How well each segment's position matches each section's planned slot."""
    total_minutes = sum(max(s.minutes, 1) for s in sections)
    bounds: list[tuple[float, float]] = []
    cursor = 0.0
    for section in sections:
        share = max(section.minutes, 1) / total_minutes
        bounds.append((cursor, cursor + share))
        cursor += share

    span = duration if duration > 0 else 1.0
    prior: list[list[float]] = []
    for seg in segments:
        position = min(max(((seg.start + seg.end) / 2.0) / span, 0.0), 1.0)
        row = []
        for lo, hi in bounds:
            if lo <= position <= hi:
                row.append(1.0)
            else:
                distance = lo - position if position < lo else position - hi
                row.append(max(0.0, 1.0 - distance / PRIOR_TOLERANCE))
        prior.append(row)
    return prior


def _softmax_confidence(row: Sequence[float], chosen: int) -> float:
    top = max(row)
    exps = [math.exp((v - top) * 3.0) for v in row]
    total = sum(exps) or 1.0
    return round(min(max(exps[chosen] / total, 0.05), 0.95), 3)


def align_segments(
    segments: Sequence[TranscriptSegment],
    template: ConversationTemplate,
    duration: float | None = None,
) -> tuple[list[str], list[float]]:
    """Assign every segment to an agenda item.

    The agenda runs in order and does not repeat, so this is a monotonic
    segmentation problem: pick the boundaries that best fit the spoken cues and
    the planned running order. Solved exactly with a small dynamic program,
    constrained so each agenda item gets at least one segment.
    """
    sections = template.agenda
    if not sections:
        return [], []
    if not segments:
        return [], []

    n, k = len(segments), len(sections)
    if n < k:
        # Fewer utterances than agenda items: lay them out in order.
        ids = [sections[min(i, k - 1)].id for i in range(n)]
        return ids, [0.2] * n

    span = duration or max(s.end for s in segments)
    cue = _cue_scores(segments, sections)
    prior = _time_prior(segments, sections, span)
    score = [
        [CUE_WEIGHT * cue[i][s] + TIME_PRIOR_WEIGHT * prior[i][s] for s in range(k)]
        for i in range(n)
    ]

    neg = float("-inf")
    dp = [[neg] * k for _ in range(n)]
    back = [[-1] * k for _ in range(n)]
    dp[0][0] = score[0][0]

    for i in range(1, n):
        remaining = k - 1 - (n - 1 - i)  # cannot be further along than the tail allows
        for s in range(k):
            if s > i or s < remaining:
                continue
            stay = dp[i - 1][s]
            advance = dp[i - 1][s - 1] if s > 0 else neg
            if stay >= advance:
                if stay == neg:
                    continue
                dp[i][s] = stay + score[i][s]
                back[i][s] = s
            else:
                dp[i][s] = advance + score[i][s]
                back[i][s] = s - 1

    path = [0] * n
    path[n - 1] = k - 1
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i][path[i]]

    section_ids = [sections[s].id for s in path]
    confidence = [_softmax_confidence(score[i], path[i]) for i in range(n)]
    return section_ids, confidence


# --------------------------------------------------------------------------
# Speaker roles
# --------------------------------------------------------------------------

def infer_speaker_roles(
    segments: Sequence[TranscriptSegment],
    manager_name: str = "",
    report_name: str = "",
) -> tuple[list[str], list[float]]:
    """Work out which diarisation label is the manager.

    Uses two signals that hold up well in coaching conversations: the coach asks
    most of the questions, and the coach speaks less than the person being
    coached. Without diarisation labels we say "unknown" rather than guess -
    the Claude path attributes turns from the content instead.
    """
    labels = [s.speaker.strip() for s in segments]
    distinct = [l for l in dict.fromkeys(labels) if l]
    if not distinct:
        return ["unknown"] * len(segments), [0.0] * len(segments)

    def named(label: str, name: str) -> bool:
        if not name:
            return False
        a, b = label.lower(), name.lower()
        return a == b or a in b or b in a or a == b.split()[0]

    stats: dict[str, dict[str, float]] = {
        label: {"questions": 0.0, "turns": 0.0, "seconds": 0.0} for label in distinct
    }
    for seg in segments:
        label = seg.speaker.strip()
        if not label:
            continue
        stats[label]["turns"] += 1
        stats[label]["seconds"] += seg.duration
        if QUESTION_RE.search(seg.text.strip()):
            stats[label]["questions"] += 1

    total_seconds = sum(v["seconds"] for v in stats.values()) or 1.0
    ranking: list[tuple[float, str]] = []
    for label, v in stats.items():
        question_rate = v["questions"] / max(v["turns"], 1.0)
        talk_share = v["seconds"] / total_seconds
        ranking.append((question_rate * 2.0 - talk_share, label))
    ranking.sort(reverse=True)

    manager_label = ranking[0][1]
    for label in distinct:
        if named(label, manager_name):
            manager_label = label
            break
    else:
        for label in distinct:
            if named(label, report_name) and len(distinct) == 2:
                manager_label = next(l for l in distinct if l != label)
                break

    explicit = any(named(l, manager_name) or named(l, report_name) for l in distinct)
    base_confidence = 0.95 if explicit else 0.6

    roles, confidence = [], []
    for seg in segments:
        label = seg.speaker.strip()
        if not label:
            roles.append("unknown")
            confidence.append(0.0)
        elif label == manager_label:
            roles.append("manager")
            confidence.append(base_confidence)
        else:
            roles.append("report")
            confidence.append(base_confidence)
    return roles, confidence


def build_structure(
    segments: Sequence[TranscriptSegment],
    template: ConversationTemplate,
    duration: float | None = None,
    manager_name: str = "",
    report_name: str = "",
) -> StructureResult:
    section_ids, section_conf = align_segments(segments, template, duration)
    roles, role_conf = infer_speaker_roles(segments, manager_name, report_name)
    return StructureResult(
        section_ids=section_ids,
        section_confidence=section_conf,
        speaker_roles=roles,
        speaker_confidence=role_conf,
        method="heuristic",
        notes="Aligned from spoken cues and the planned running order.",
    )


# --------------------------------------------------------------------------
# Extractive drafting
# --------------------------------------------------------------------------

def _section_segments(
    segments: Sequence[TranscriptSegment], section_ids: Sequence[str], section_id: str
) -> list[tuple[int, TranscriptSegment]]:
    return [
        (i, seg)
        for i, seg in enumerate(segments)
        if i < len(section_ids) and section_ids[i] == section_id
    ]


def _matching(
    pairs: Iterable[tuple[int, TranscriptSegment]],
    pattern: re.Pattern[str],
    roles: Sequence[str] | None = None,
    want_role: str | None = None,
    limit: int = 6,
) -> tuple[list[str], list[int]]:
    out: list[str] = []
    evidence: list[int] = []
    for i, seg in pairs:
        if want_role and roles is not None and i < len(roles) and roles[i] not in (want_role, "unknown"):
            continue
        for sentence in split_sentences(seg.text):
            if pattern.search(sentence):
                out.append(truncate(sentence, 200))
                evidence.append(i)
                break
        if len(out) >= limit:
            break
    return out, evidence


def _representative_quotes(
    pairs: Sequence[tuple[int, TranscriptSegment]], section: SectionSpec, limit: int = 3
) -> tuple[list[str], list[int]]:
    """Pick the lines that best characterise a stretch of conversation."""
    if not pairs:
        return [], []
    scored: list[tuple[float, int, str]] = []
    for i, seg in pairs:
        text = normalise(seg.text)
        if len(text) < 25:
            continue
        lowered = f" {text.lower()} "
        cue_hits = sum(1 for cue in section.cues if cue in lowered)
        length_score = min(len(text) / 220.0, 1.0)
        scored.append((cue_hits + length_score, i, text))
    if not scored:
        scored = [(0.0, i, normalise(seg.text)) for i, seg in pairs]
    scored.sort(key=lambda row: (-row[0], row[1]))
    picked = sorted(scored[:limit], key=lambda row: row[1])
    return [truncate(text, 240) for _, _, text in picked], [i for _, i, _ in picked]


def _actions_from(
    pairs: Sequence[tuple[int, TranscriptSegment]],
    roles: Sequence[str],
    manager_name: str,
    report_name: str,
) -> tuple[list[dict[str, str]], list[int]]:
    actions: list[dict[str, str]] = []
    evidence: list[int] = []
    for i, seg in pairs:
        for sentence in split_sentences(seg.text):
            if len(sentence) < 30 or not COMMITMENT_RE.search(sentence):
                continue
            speaker_role = roles[i] if i < len(roles) else "unknown"
            lowered = sentence.lower()
            if re.search(r"\b(i'?ll|i will|i'?m going to|i can)\b", lowered):
                owner = manager_name if speaker_role == "manager" else report_name
            elif re.search(r"\byou'?ll|you will\b", lowered):
                owner = report_name if speaker_role == "manager" else manager_name
            else:
                owner = ""
            due = ""
            due_match = re.search(
                r"\bby (the end of [\w ]+|next [\w]+|monday|tuesday|wednesday|thursday|friday|"
                r"the \d{1,2}(?:st|nd|rd|th)?[\w ]*)",
                sentence,
                re.IGNORECASE,
            )
            if due_match:
                due = due_match.group(1).strip()
            actions.append(
                {
                    "action": truncate(sentence, 200),
                    "owner": owner,
                    "due": due,
                    "support": "",
                }
            )
            evidence.append(i)
            break
        if len(actions) >= 8:
            break
    return actions, evidence


def _tone(segments: Sequence[TranscriptSegment]) -> str:
    text = " ".join(s.text for s in segments)
    tense = len(TENSE_TONE_RE.findall(text))
    directive = len(DIRECTIVE_TONE_RE.findall(text))
    positive = len(POSITIVE_TONE_RE.findall(text))
    if tense >= 3 and tense > positive:
        return "tense"
    if directive >= 3 and directive > positive:
        return "directive"
    if positive >= 3 and tense == 0:
        return "supportive"
    if positive and (tense or directive):
        return "mixed"
    return "neutral"


def draft_fields(
    segments: Sequence[TranscriptSegment],
    structure: StructureResult,
    template: ConversationTemplate,
    manager_name: str = "",
    report_name: str = "",
) -> DraftResult:
    """Fill the form from the transcript using only verbatim excerpts."""
    section_ids = structure.section_ids
    roles = structure.speaker_roles
    all_pairs = list(enumerate(segments))
    drafted: list[DraftedField] = []

    for section in template.sections:
        pairs = (
            all_pairs
            if section.kind == "record"
            else _section_segments(segments, section_ids, section.id)
        )
        for spec in section.fields:
            value: Any
            evidence: list[int] = []
            confidence = 0.3

            if spec.kind == "actions":
                value, evidence = _actions_from(pairs, roles, manager_name, report_name)
                confidence = 0.4 if value else 0.15
            elif spec.id == "blockers":
                items, evidence = _matching(pairs, BLOCKER_RE, roles, "report")
                value, confidence = items, (0.4 if items else 0.15)
            elif spec.id == "tried_already":
                items, evidence = _matching(pairs, TRIED_RE, roles, "report")
                value, confidence = items, (0.4 if items else 0.15)
            elif spec.id == "feedback_given":
                praise, ev_a = _matching(pairs, PRAISE_RE, roles, "manager", limit=4)
                critique, ev_b = _matching(pairs, CRITIQUE_RE, roles, "manager", limit=4)
                value = praise + critique
                evidence = ev_a + ev_b
                confidence = 0.4 if value else 0.15
            elif spec.id == "commitments_by_manager":
                items, evidence = _matching(pairs, SUPPORT_RE, roles, "manager")
                value, confidence = items, (0.4 if items else 0.15)
            elif spec.id == "strength_named":
                items, evidence = _matching(pairs, PRAISE_RE, roles, "manager", limit=1)
                value = items[0] if items else ""
                confidence = 0.4 if items else 0.15
            elif spec.kind == "choice":
                value = _tone(segments) if spec.id == "tone" else (spec.choices[0] if spec.choices else "")
                confidence = 0.3
            elif spec.id == "headline":
                quotes, evidence = _representative_quotes(pairs, section, limit=1)
                value = quotes[0] if quotes else ""
                confidence = 0.25
            elif spec.id == "follow_up":
                items, evidence = _matching(
                    list(reversed(list(pairs))),
                    re.compile(
                        r"\b(same time next|next (?:month|week)|catch up|check in|follow up|"
                        r"speak (?:again|next)|before we next meet|book (?:a|another))\b",
                        re.IGNORECASE,
                    ),
                    limit=1,
                )
                value = items[0] if items else ""
                confidence = 0.3 if items else 0.15
            elif spec.kind == "list":
                quotes, evidence = _representative_quotes(pairs, section, limit=3)
                value, confidence = quotes, (0.25 if quotes else 0.1)
            else:  # free text
                quotes, evidence = _representative_quotes(pairs, section, limit=3)
                value = "\n".join(f"“{q}”" for q in quotes)
                confidence = 0.25 if quotes else 0.1

            drafted.append(
                DraftedField(
                    section_id=section.id,
                    field_id=spec.id,
                    value=value,
                    confidence=confidence,
                    evidence=sorted(dict.fromkeys(evidence)),
                )
            )

    return DraftResult(fields=drafted, method="heuristic", model="extractive")
