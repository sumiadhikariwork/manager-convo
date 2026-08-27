"""Types exchanged between the pipeline and its analysis providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StructureResult:
    """Where each transcript segment sits: which agenda item, and who spoke.

    All three lists are parallel to the transcript's segment list.
    """

    section_ids: list[str]
    section_confidence: list[float]
    speaker_roles: list[str]
    speaker_confidence: list[float]
    method: str = "heuristic"
    notes: str = ""


@dataclass
class DraftedField:
    """One drafted answer, with the segments it is drawn from.

    ``evidence`` holds *segment indices only*. Quote text is resolved from the
    stored transcript afterwards, so a citation can never be a paraphrase.
    """

    section_id: str
    field_id: str
    value: Any
    confidence: float = 0.0
    evidence: list[int] = field(default_factory=list)


@dataclass
class DraftResult:
    fields: list[DraftedField] = field(default_factory=list)
    method: str = "heuristic"
    model: str = ""


class AnalysisError(RuntimeError):
    """Raised when a provider cannot produce a usable result."""
