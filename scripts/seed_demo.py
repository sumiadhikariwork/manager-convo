#!/usr/bin/env python3
"""Load the demo conversation so the app has something to show.

Generates a silent WAV of the right length, drops the fixture transcript beside
it as a sidecar, and runs the pipeline against it with SPEECH_PROVIDER=fixture.
The audio is silent - it exists so the player, the seeking and the timeline are
real. The words come from tests/fixtures/monthly_checkin.txt.

    python scripts/seed_demo.py
"""

from __future__ import annotations

import os
import struct
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "tests" / "fixtures" / "monthly_checkin.txt"
DURATION_SECONDS = 445
SAMPLE_RATE = 8000


def write_silent_wav(path: Path, seconds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(struct.pack("<h", 0) * SAMPLE_RATE * seconds)


def main() -> int:
    if not FIXTURE.exists():
        print(f"Missing fixture: {FIXTURE}", file=sys.stderr)
        return 1

    # The fixture provider reads a sidecar rather than decoding audio.
    os.environ["SPEECH_PROVIDER"] = "fixture"
    os.environ.setdefault("STORAGE_BACKEND", "local")

    from app.config import get_settings
    from app.db import init_db, session_scope
    from app.models import Conversation, record_event
    from app.pipeline import process_conversation
    from app.templates import GROW_MONTHLY

    settings = get_settings()
    settings.ensure_dirs()
    init_db()

    with session_scope() as session:
        conversation = Conversation(
            title="Monthly check-in (demo)",
            template_id=GROW_MONTHLY.id,
            manager_name="Priya",
            report_name="Sam",
            occurred_on="2026-08-24",
            consent_confirmed=True,
            audio_filename="monthly_checkin.wav",
            audio_mime="audio/wav",
        )
        session.add(conversation)
        session.flush()
        conversation_id = conversation.id

        audio_key = f"audio/{conversation_id}.wav"
        audio_path = settings.audio_dir / audio_key
        write_silent_wav(audio_path, DURATION_SECONDS)
        audio_path.with_suffix(".wav.txt").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

        conversation.audio_key = audio_key
        conversation.audio_path = str(audio_path)
        conversation.audio_bytes = audio_path.stat().st_size
        record_event(session, conversation_id, "uploaded", actor="Priya", filename="monthly_checkin.wav")

    print(f"Seeded conversation {conversation_id}; running the pipeline…")
    process_conversation(conversation_id, settings)

    with session_scope() as session:
        conversation = session.get(Conversation, conversation_id)
        print(f"Status: {conversation.status.value} — {conversation.status_detail}")
        if conversation.error:
            print(f"Error: {conversation.error}", file=sys.stderr)
            return 1

    print("Done. Start the app with: uvicorn app.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
