"""Hosted transcription through AssemblyAI.

Chosen for two properties that matter to a serverless deployment more than
accuracy does: it fetches the recording from a URL itself, and it reports
completion by webhook. Neither the audio nor the wait touches our functions.

It also does speaker diarisation, so on this path the speaker labels are
measured rather than inferred - a straight improvement on the local Whisper
path, where roles have to be guessed from who asks the questions.
"""

from __future__ import annotations

import logging
from typing import Any

# The Anthropic SDK brings httpx2 with it, so this costs nothing extra in the
# deployment bundle.
import httpx2 as httpx

from app.transcription.base import Transcript, TranscriptionError, TranscriptSegment
from app.util import normalise

logger = logging.getLogger(__name__)

API_ROOT = "https://api.assemblyai.com"
WEBHOOK_HEADER = "X-Conversation-Records-Secret"
TIMEOUT = 30.0


class AssemblyAISpeechProvider:
    name = "assemblyai"

    def __init__(
        self,
        api_key: str,
        language: str = "",
        speakers_expected: int = 2,
        api_root: str = API_ROOT,
        client: Any = None,
    ):
        if not api_key:
            raise TranscriptionError(
                "ASSEMBLYAI_API_KEY is required when SPEECH_PROVIDER=assemblyai."
            )
        self.api_key = api_key
        self.language = language
        self.speakers_expected = speakers_expected
        self.api_root = api_root.rstrip("/")
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = httpx.Client(timeout=TIMEOUT)
        return self._client

    @property
    def _headers(self) -> dict[str, str]:
        return {"authorization": self.api_key, "content-type": "application/json"}

    # -- submit ------------------------------------------------------------
    def submit(self, audio_url: str, webhook_url: str, webhook_secret: str = "") -> str:
        payload: dict[str, Any] = {
            "audio_url": audio_url,
            "speaker_labels": True,
            "punctuate": True,
            "format_text": True,
        }
        if self.speakers_expected > 0:
            # A coaching conversation is two people; saying so stops the model
            # splitting one voice across several labels.
            payload["speaker_options"] = {
                "min_speakers_expected": self.speakers_expected,
                "max_speakers_expected": self.speakers_expected,
            }
        if self.language:
            payload["language_code"] = self.language
        else:
            payload["language_detection"] = True
        if webhook_url:
            payload["webhook_url"] = webhook_url
            if webhook_secret:
                payload["webhook_auth_header_name"] = WEBHOOK_HEADER
                payload["webhook_auth_header_value"] = webhook_secret

        try:
            response = self.client.post(
                f"{self.api_root}/v2/transcript", headers=self._headers, json=payload
            )
        except Exception as exc:
            raise TranscriptionError(f"Could not reach AssemblyAI: {exc}") from exc

        if response.status_code >= 400:
            raise TranscriptionError(
                f"AssemblyAI rejected the recording ({response.status_code}): {response.text[:300]}"
            )

        body = response.json()
        job_id = body.get("id")
        if not job_id:
            raise TranscriptionError(f"AssemblyAI returned no job id: {body}")
        logger.info("Submitted %s to AssemblyAI as %s", audio_url.split("?")[0], job_id)
        return str(job_id)

    # -- collect -----------------------------------------------------------
    def fetch(self, job_id: str) -> Transcript:
        try:
            response = self.client.get(
                f"{self.api_root}/v2/transcript/{job_id}", headers=self._headers
            )
        except Exception as exc:
            raise TranscriptionError(f"Could not reach AssemblyAI: {exc}") from exc

        if response.status_code >= 400:
            raise TranscriptionError(
                f"AssemblyAI would not return transcript {job_id} "
                f"({response.status_code}): {response.text[:300]}"
            )

        body = response.json()
        status = body.get("status")
        if status == "error":
            raise TranscriptionError(f"AssemblyAI failed to transcribe: {body.get('error')}")
        if status != "completed":
            raise TranscriptionError(f"Transcript {job_id} is not ready yet (status: {status}).")

        return self._to_transcript(body)

    def _to_transcript(self, body: dict[str, Any]) -> Transcript:
        utterances = body.get("utterances") or []
        segments: list[TranscriptSegment] = []
        for utterance in utterances:
            text = normalise(utterance.get("text", ""))
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    index=len(segments),
                    # AssemblyAI reports milliseconds.
                    start=round(float(utterance.get("start", 0)) / 1000.0, 3),
                    end=round(float(utterance.get("end", 0)) / 1000.0, 3),
                    text=text,
                    speaker=str(utterance.get("speaker") or ""),
                )
            )

        if not segments:
            # Diarisation off or a single unbroken block: fall back to the
            # whole-transcript text rather than losing the recording.
            text = normalise(body.get("text") or "")
            if not text:
                raise TranscriptionError("AssemblyAI found no speech in the recording.")
            segments = [
                TranscriptSegment(
                    index=0, start=0.0, end=float(body.get("audio_duration") or 0.0), text=text
                )
            ]

        return Transcript(
            segments=segments,
            language=str(body.get("language_code") or ""),
            provider=self.name,
            model="universal",
            duration=float(body.get("audio_duration") or segments[-1].end),
        )
