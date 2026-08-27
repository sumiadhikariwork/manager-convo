"""Runtime configuration, read from the environment (or a local .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Speech to text
    speech_provider: str = "faster_whisper"
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"

    # Understanding
    analysis_provider: str = "claude"
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-5"

    # Storage
    data_dir: Path = Path("./data")
    database_url: str = "sqlite:///./data/manager_convo.sqlite3"
    max_upload_mb: int = 200

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def claude_available(self) -> bool:
        """True when a Claude-backed analysis run is actually possible.

        An unset ANTHROPIC_API_KEY does not by itself mean there are no
        credentials - the SDK also resolves ANTHROPIC_AUTH_TOKEN and an
        `ant auth login` profile - so only the explicit "heuristic" setting
        turns the Claude path off.
        """
        return self.analysis_provider.lower() == "claude"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
