"""Runtime configuration, read from the environment (or a local .env file)."""

from __future__ import annotations

import os
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

    # Hosted transcription (speech_provider=assemblyai)
    assemblyai_api_key: str = ""
    assemblyai_language: str = ""
    assemblyai_speakers_expected: int = 2

    # Understanding
    analysis_provider: str = "claude"
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-5"

    # Storage
    data_dir: Path = Path("./data")
    database_url: str = "sqlite:///./data/manager_convo.sqlite3"
    max_upload_mb: int = 200

    #: "local" (a directory on disk) or "s3" (any S3-compatible bucket).
    storage_backend: str = "local"
    storage_bucket: str = ""
    storage_region: str = "auto"
    storage_endpoint_url: str = ""
    storage_access_key_id: str = ""
    storage_secret_access_key: str = ""
    #: Set when the bucket is served from a CDN or public domain; otherwise
    #: playback uses a short-lived presigned URL.
    storage_public_base_url: str = ""

    # Execution
    #: "thread" runs the pipeline in a background thread (a long-lived server).
    #: "deferred" advances it one stage per request, driven by webhooks and the
    #: resume endpoint - the only shape that survives a serverless host.
    job_runner: str = "thread"
    #: Public origin of this deployment, needed so transcription webhooks can
    #: reach us. On Vercel, VERCEL_URL supplies it automatically.
    public_base_url: str = ""
    #: Shared secret in the webhook URL, so only the transcription service can
    #: advance a conversation.
    webhook_secret: str = ""
    #: Create tables at startup. Convenient locally; on a serverless host run
    #: scripts/migrate.py once instead of paying for this on every cold start.
    auto_create_tables: bool = True

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        """Create the storage directories, with a legible error if we cannot.

        Called explicitly at startup rather than at import, so that merely
        importing the app on a read-only filesystem does not kill the process
        before it can report anything.
        """
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.audio_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot write to DATA_DIR ({self.data_dir}): {exc}. "
                "This app stores recordings and its database on disk, so it needs a "
                "writable, persistent directory. On a read-only or ephemeral host, "
                "point DATA_DIR at a mounted volume."
            ) from exc

    @property
    def base_url(self) -> str:
        """Public origin of this deployment, without a trailing slash."""
        if self.public_base_url:
            return self.public_base_url.rstrip("/")
        # Vercel exposes the deployment host but not the scheme.
        vercel = os.environ.get("VERCEL_PROJECT_PRODUCTION_URL") or os.environ.get("VERCEL_URL")
        if vercel:
            return f"https://{vercel.rstrip('/')}"
        return ""

    @property
    def is_serverless(self) -> bool:
        return self.job_runner.lower() == "deferred"

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
    """Read configuration. Deliberately free of side effects.

    Directory creation happens in ensure_dirs(), called from the application's
    startup hook - reading configuration must never touch the filesystem.
    """
    return Settings()
