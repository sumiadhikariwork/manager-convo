"""Small shared helpers."""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def format_timestamp(seconds: float | None) -> str:
    """Seconds to m:ss (or h:mm:ss past an hour)."""
    if seconds is None:
        return "--:--"
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def normalise(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def split_sentences(text: str) -> list[str]:
    """Cheap sentence split that keeps the terminator."""
    text = normalise(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def truncate(text: str, limit: int = 240) -> str:
    text = normalise(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
