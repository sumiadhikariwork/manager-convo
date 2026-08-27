"""Shared test setup.

The database engine is built at import time from the settings, so the
environment has to be pointed at a throwaway directory before anything under
``app`` is imported.
"""

from __future__ import annotations

import os
import struct
import tempfile
import wave
from pathlib import Path

_TEST_DATA = Path(tempfile.mkdtemp(prefix="manager-convo-tests-"))
os.environ["DATA_DIR"] = str(_TEST_DATA)
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DATA / 'test.sqlite3'}"
os.environ["SPEECH_PROVIDER"] = "fixture"
os.environ["ANALYSIS_PROVIDER"] = "heuristic"
os.environ["ANTHROPIC_API_KEY"] = ""

import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import SessionLocal, engine, init_db  # noqa: E402
from app.models import Base  # noqa: E402
from app.templates import GROW_MONTHLY  # noqa: E402
from app.transcription.fixture import _script_to_transcript  # noqa: E402

FIXTURE_SCRIPT = Path(__file__).parent / "fixtures" / "monthly_checkin.txt"
DEMO_DURATION = 445.0


@pytest.fixture(scope="session", autouse=True)
def _database():
    get_settings().ensure_dirs()
    init_db()
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture(autouse=True)
def _clean_tables():
    yield
    with SessionLocal() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def script() -> str:
    return FIXTURE_SCRIPT.read_text(encoding="utf-8")


@pytest.fixture
def transcript(script):
    """The fixture conversation as a Transcript, with speaker labels."""
    return _script_to_transcript(script, DEMO_DURATION)


@pytest.fixture
def template():
    return GROW_MONTHLY


@pytest.fixture
def silent_wav(tmp_path) -> Path:
    """A real, playable WAV of the right length, with no sound in it."""
    path = tmp_path / "conversation.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(struct.pack("<h", 0) * 8000 * 5)
    return path


@pytest.fixture
def audio_with_sidecar(settings, script, silent_wav) -> Path:
    """A WAV in the app's audio directory with its fixture transcript beside it."""
    target = settings.audio_dir / silent_wav.name
    target.write_bytes(silent_wav.read_bytes())
    target.with_suffix(".wav.txt").write_text(script, encoding="utf-8")
    return target
