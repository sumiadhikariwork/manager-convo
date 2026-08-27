"""Turning a transcript into an aligned, drafted feedback record."""

from __future__ import annotations

import logging
from typing import Sequence

from app.analysis import heuristic
from app.analysis.claude import ClaudeAnalyst
from app.analysis.metrics import compute_metrics
from app.analysis.types import AnalysisError, DraftedField, DraftResult, StructureResult
from app.config import Settings, get_settings
from app.templates import ConversationTemplate
from app.transcription.base import TranscriptSegment

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisError",
    "DraftResult",
    "DraftedField",
    "StructureResult",
    "ClaudeAnalyst",
    "HeuristicAnalyst",
    "compute_metrics",
    "get_analyst",
]


class HeuristicAnalyst:
    """Offline alignment and extractive drafting. Always available."""

    name = "heuristic"
    model = "extractive"

    def structure(
        self,
        segments: Sequence[TranscriptSegment],
        template: ConversationTemplate,
        manager_name: str = "",
        report_name: str = "",
        duration: float | None = None,
    ) -> StructureResult:
        return heuristic.build_structure(segments, template, duration, manager_name, report_name)

    def draft(
        self,
        segments: Sequence[TranscriptSegment],
        structure: StructureResult,
        template: ConversationTemplate,
        manager_name: str = "",
        report_name: str = "",
    ) -> DraftResult:
        return heuristic.draft_fields(segments, structure, template, manager_name, report_name)


def get_analyst(settings: Settings | None = None):
    settings = settings or get_settings()
    if settings.claude_available():
        return ClaudeAnalyst(model=settings.claude_model, api_key=settings.anthropic_api_key)
    return HeuristicAnalyst()


def get_fallback_analyst() -> HeuristicAnalyst:
    return HeuristicAnalyst()
