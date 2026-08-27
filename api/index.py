"""Vercel entrypoint.

Vercel's Python runtime looks for an ASGI application in `api/`. Everything
real lives in the `app` package; this only puts the repository root on the
import path and re-exports it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402,F401

# Vercel invokes the module-level `app`.
__all__ = ["app"]
