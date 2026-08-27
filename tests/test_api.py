"""The HTTP surface, exercised through a real client."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import pipeline
from app.config import get_settings
from app.db import SessionLocal
from app.main import app
from app.models import Conversation
from app.templates import GROW_MONTHLY


@pytest.fixture
def client(monkeypatch):
    """A client whose uploads are processed synchronously, so tests are deterministic."""
    settings = get_settings()
    monkeypatch.setattr(
        "app.main.submit",
        lambda conversation_id, _settings=None: pipeline.process_conversation(conversation_id, settings) or True,
    )
    with TestClient(app) as test_client:
        yield test_client


def _upload(client, script, name="check-in.wav", **form):
    """Upload audio and drop the fixture transcript beside it, as the app would find it."""
    payload = {
        "title": "Monthly check-in",
        "template_id": GROW_MONTHLY.id,
        "manager_name": "Priya",
        "report_name": "Sam",
        "occurred_on": "2026-08-24",
        "consent_confirmed": "true",
    }
    payload.update(form)

    # The upload is written under the new conversation's id, so the sidecar has
    # to be planted after the file lands. Patch the fixture provider's lookup
    # by writing the script for every .wav in the audio directory instead.
    import wave, struct, io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(struct.pack("<h", 0) * 8000 * 5)
    buffer.seek(0)

    original = pipeline.get_speech_provider

    def provider(settings_arg=None):
        speech = original(settings_arg)
        real_transcribe = speech.transcribe

        def transcribe(path):
            path.with_suffix(".wav.txt").write_text(script, encoding="utf-8")
            return real_transcribe(path)

        speech.transcribe = transcribe
        return speech

    pipeline.get_speech_provider = provider
    try:
        return client.post(
            "/api/conversations",
            data=payload,
            files={"audio": (name, buffer.read(), "audio/wav")},
        )
    finally:
        pipeline.get_speech_provider = original


# -- templates and config --------------------------------------------------

def test_templates_are_listed_with_their_fields(client):
    response = client.get("/api/templates")
    assert response.status_code == 200
    templates = response.json()
    grow = next(t for t in templates if t["id"] == GROW_MONTHLY.id)
    assert [s["title"] for s in grow["sections"]][:4] == ["Open", "Goals", "Reality", "Way Forward"]
    assert grow["planned_minutes"] == 15


def test_config_reports_what_this_deployment_runs_on(client):
    body = client.get("/api/config").json()
    assert body["speech_provider"] == "fixture"
    assert body["speech_model"] == "sidecar"
    assert body["analysis_provider"] == "heuristic"
    assert body["job_runner"] == "thread"


# -- upload ----------------------------------------------------------------

def test_upload_requires_consent(client, script):
    response = _upload(client, script, consent_confirmed="false")
    assert response.status_code == 400
    assert "recorded" in response.json()["detail"]


def test_upload_rejects_a_non_audio_file(client):
    response = client.post(
        "/api/conversations",
        data={"consent_confirmed": "true"},
        files={"audio": ("notes.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported audio type" in response.json()["detail"]


def test_upload_rejects_an_empty_file(client):
    response = client.post(
        "/api/conversations",
        data={"consent_confirmed": "true"},
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400


def test_upload_rejects_an_unknown_template(client, script):
    response = _upload(client, script, template_id="does-not-exist")
    assert response.status_code == 400


def test_upload_processes_and_returns_a_filled_in_record(client, script):
    response = _upload(client, script)
    assert response.status_code == 201
    conversation_id = response.json()["id"]

    detail = client.get(f"/api/conversations/{conversation_id}").json()
    assert detail["status"] == "ready"
    assert detail["consent_confirmed"] is True
    assert len(detail["segments"]) > 20
    assert len(detail["fields"]) == sum(len(s.fields) for s in GROW_MONTHLY.sections)

    strength = next(f for f in detail["fields"] if f["field_id"] == "strength_named")
    assert "migration weekend" in strength["value"]
    assert strength["evidence"]
    assert strength["edited"] is False


def test_status_endpoint_reports_progress(client, script):
    conversation_id = _upload(client, script).json()["id"]
    body = client.get(f"/api/conversations/{conversation_id}/status").json()
    assert body["status"] == "ready"
    assert body["processing"] is False


def test_conversations_are_listed_newest_first(client, script):
    first = _upload(client, script, title="First").json()["id"]
    second = _upload(client, script, title="Second").json()["id"]
    listed = client.get("/api/conversations").json()
    assert [c["id"] for c in listed][:2] == [second, first]


# -- editing ---------------------------------------------------------------

def test_editing_a_field_marks_it_and_keeps_the_draft(client, script):
    detail = _upload(client, script).json()
    field = next(f for f in detail["fields"] if f["field_id"] == "strength_named")
    original_draft = field["draft_value"]

    response = client.patch(
        f"/api/conversations/{detail['id']}/fields/{field['id']}",
        json={"value": "Wrote the rollback plan unprompted.", "edited_by": "Priya"},
    )
    assert response.status_code == 200
    updated = response.json()
    assert updated["value"] == "Wrote the rollback plan unprompted."
    assert updated["edited"] is True
    assert updated["source"] == "manager"
    assert updated["edited_by"] == "Priya"
    assert updated["draft_value"] == original_draft, "the draft must survive the edit"


def test_reverting_restores_the_draft(client, script):
    detail = _upload(client, script).json()
    field = next(f for f in detail["fields"] if f["field_id"] == "strength_named")
    client.patch(f"/api/conversations/{detail['id']}/fields/{field['id']}", json={"value": "changed"})

    reverted = client.post(
        f"/api/conversations/{detail['id']}/fields/{field['id']}/revert"
    ).json()
    assert reverted["value"] == field["draft_value"]
    assert reverted["edited"] is False


def test_setting_a_field_back_to_the_draft_is_not_an_edit(client, script):
    detail = _upload(client, script).json()
    field = next(f for f in detail["fields"] if f["field_id"] == "strength_named")
    updated = client.patch(
        f"/api/conversations/{detail['id']}/fields/{field['id']}",
        json={"value": field["draft_value"]},
    ).json()
    assert updated["edited"] is False


def test_a_field_from_another_conversation_is_not_reachable(client, script):
    first = _upload(client, script).json()
    second = _upload(client, script).json()
    stolen = second["fields"][0]["id"]
    response = client.patch(
        f"/api/conversations/{first['id']}/fields/{stolen}", json={"value": "x"}
    )
    assert response.status_code == 404


def test_editing_is_written_to_the_audit_trail(client, script):
    detail = _upload(client, script).json()
    field = next(f for f in detail["fields"] if f["field_id"] == "strength_named")
    client.patch(
        f"/api/conversations/{detail['id']}/fields/{field['id']}",
        json={"value": "changed", "edited_by": "Priya"},
    )
    events = client.get(f"/api/conversations/{detail['id']}/audit").json()
    actions = [e["action"] for e in events]
    assert actions[0] == "uploaded"
    assert "field_edited" in actions
    edit = next(e for e in events if e["action"] == "field_edited")
    assert edit["actor"] == "Priya"
    assert edit["detail"]["field_id"] == "strength_named"
    assert edit["detail"]["current"] == "changed"


# -- audio -----------------------------------------------------------------

def test_audio_is_served_whole(client, script):
    conversation_id = _upload(client, script).json()["id"]
    response = client.get(f"/api/conversations/{conversation_id}/audio")
    assert response.status_code == 200
    assert response.headers["accept-ranges"] == "bytes"


def test_audio_honours_range_requests_so_the_player_can_seek(client, script):
    conversation_id = _upload(client, script).json()["id"]
    response = client.get(
        f"/api/conversations/{conversation_id}/audio", headers={"Range": "bytes=100-199"}
    )
    assert response.status_code == 206
    assert response.headers["content-length"] == "100"
    assert response.headers["content-range"].startswith("bytes 100-199/")
    assert len(response.content) == 100


def test_a_range_past_the_end_is_rejected(client, script):
    conversation_id = _upload(client, script).json()["id"]
    response = client.get(
        f"/api/conversations/{conversation_id}/audio", headers={"Range": "bytes=99999999-"}
    )
    assert response.status_code == 416


# -- export ----------------------------------------------------------------

def test_markdown_export_reads_as_a_shareable_record(client, script):
    detail = _upload(client, script).json()
    body = client.get(f"/api/conversations/{detail['id']}/export.md").text

    assert "# Monthly check-in" in body
    assert "**Manager:** Priya" in body
    assert "## Open · 2 min planned" in body
    assert "### One specific thing done well" in body
    assert "## Provenance" in body
    assert "Fields changed by the manager: 0 of" in body


def test_markdown_export_can_include_the_transcript(client, script):
    detail = _upload(client, script).json()
    body = client.get(f"/api/conversations/{detail['id']}/export.md?transcript=true").text
    assert "## Transcript" in body
    assert "**Priya:**" in body


def test_markdown_export_names_the_fields_the_manager_changed(client, script):
    detail = _upload(client, script).json()
    field = next(f for f in detail["fields"] if f["field_id"] == "strength_named")
    client.patch(f"/api/conversations/{detail['id']}/fields/{field['id']}", json={"value": "changed"})

    body = client.get(f"/api/conversations/{detail['id']}/export.md").text
    assert "Fields changed by the manager: 1 of" in body
    assert "edited by the manager" in body


def test_json_export_carries_the_record_and_the_transcript(client, script):
    detail = _upload(client, script).json()
    payload = json.loads(client.get(f"/api/conversations/{detail['id']}/export.json").text)

    assert payload["conversation"]["manager"] == "Priya"
    assert payload["provenance"]["speech_provider"] == "fixture"
    assert len(payload["transcript"]) > 20
    section_ids = [s["id"] for s in payload["sections"]]
    assert section_ids == [s.id for s in GROW_MONTHLY.sections]


# -- lifecycle -------------------------------------------------------------

def test_reprocess_reruns_the_pipeline(client, script):
    detail = _upload(client, script).json()
    response = client.post(f"/api/conversations/{detail['id']}/reprocess")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    actions = [e["action"] for e in client.get(f"/api/conversations/{detail['id']}/audit").json()]
    assert actions.count("drafted") == 2


def test_delete_removes_the_record_and_the_recording(client, script):
    from app.storage import get_storage

    storage = get_storage(get_settings())
    detail = _upload(client, script).json()
    key = f"audio/{detail['id']}.wav"
    assert storage.exists(key)

    assert client.delete(f"/api/conversations/{detail['id']}").status_code == 204
    assert client.get(f"/api/conversations/{detail['id']}").status_code == 404
    assert not storage.exists(key)


def test_unknown_conversation_is_a_404(client):
    assert client.get("/api/conversations/nope").status_code == 404
    assert client.get("/api/conversations/nope/audit").status_code == 404
    assert client.get("/api/conversations/nope/export.md").status_code == 404


def test_the_web_app_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Conversation records" in response.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/healthz").json() == {"status": "ok"}


# -- direct upload and callbacks -------------------------------------------

class _Bucket:
    """Storage that takes bytes directly, the way S3 or R2 would."""

    name = "s3"
    supports_direct_upload = True

    def __init__(self):
        self.objects: dict[str, int] = {}

    def upload_ticket(self, key, content_type):
        from app.storage import UploadTicket

        return UploadTicket(
            key=key,
            upload_url=f"https://bucket.example/{key}?sig=abc",
            method="PUT",
            headers={"Content-Type": content_type},
        )

    def put(self, key, stream, content_type):
        self.objects[key] = len(stream.read())
        return self.objects[key]

    def exists(self, key):
        return key in self.objects

    def size(self, key):
        return self.objects[key]

    def delete(self, key):
        self.objects.pop(key, None)

    def playback_url(self, key, expires_in=3600):
        return f"https://bucket.example/{key}?sig=abc"

    def local_path(self, key):
        return None


@pytest.fixture
def bucket(monkeypatch):
    """Swap in remote storage everywhere the app reaches for it."""
    from app import main as main_module

    store = _Bucket()
    monkeypatch.setattr(main_module, "get_storage", lambda *_: store)
    monkeypatch.setattr("app.pipeline.get_storage", lambda *_: store)
    return store


def test_upload_ticket_hands_back_a_direct_url(client, bucket):
    response = client.post(
        "/api/uploads", data={"filename": "check-in.m4a", "content_type": "audio/mp4"}
    )
    assert response.status_code == 200
    ticket = response.json()
    assert ticket["direct"] is True
    assert ticket["upload_url"].startswith("https://bucket.example/audio/")
    assert ticket["key"].startswith("audio/") and ticket["key"].endswith(".m4a")
    assert ticket["headers"]["Content-Type"] == "audio/mp4"


def test_upload_ticket_rejects_a_non_audio_filename(client, bucket):
    response = client.post("/api/uploads", data={"filename": "notes.txt"})
    assert response.status_code == 400


def test_local_storage_reports_no_direct_upload(client):
    """The browser uses this to decide which upload path to take."""
    body = client.get("/api/config").json()
    assert body["storage_backend"] == "local"
    assert body["direct_upload"] is False


def test_completing_an_upload_creates_the_record(client, bucket, script, monkeypatch):
    from app import pipeline

    ticket = client.post("/api/uploads", data={"filename": "a.wav"}).json()
    bucket.objects[ticket["key"]] = 4096  # the browser's PUT landed

    class Stub:
        name = "assemblyai"

        def submit(self, audio_url, webhook_url, webhook_secret=""):
            return "job-xyz"

        def fetch(self, job_id):
            from app.transcription.fixture import _script_to_transcript

            return _script_to_transcript(script, 445.0)

    monkeypatch.setattr(pipeline, "get_speech_provider", lambda *_: Stub())
    monkeypatch.setattr(get_settings(), "job_runner", "deferred")
    monkeypatch.setattr(get_settings(), "public_base_url", "https://records.example")

    response = client.post(
        "/api/conversations/complete",
        json={
            "key": ticket["key"],
            "audio_filename": "a.wav",
            "manager_name": "Priya",
            "report_name": "Sam",
            "consent_confirmed": True,
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "transcribing", "the request must not wait for transcription"
    assert body["audio_filename"] == "a.wav"


def test_completing_an_upload_still_requires_consent(client, bucket):
    ticket = client.post("/api/uploads", data={"filename": "a.wav"}).json()
    bucket.objects[ticket["key"]] = 10
    response = client.post(
        "/api/conversations/complete",
        json={"key": ticket["key"], "audio_filename": "a.wav", "consent_confirmed": False},
    )
    assert response.status_code == 400


def test_completing_an_upload_that_never_landed_is_a_conflict(client, bucket):
    response = client.post(
        "/api/conversations/complete",
        json={"key": "audio/never.wav", "audio_filename": "never.wav", "consent_confirmed": True},
    )
    assert response.status_code == 409


def test_a_forged_storage_key_is_rejected(client, bucket):
    bucket.objects["secrets/keys.wav"] = 10
    response = client.post(
        "/api/conversations/complete",
        json={"key": "secrets/keys.wav", "audio_filename": "keys.wav", "consent_confirmed": True},
    )
    assert response.status_code == 400


def test_remote_audio_redirects_to_the_bucket(client, bucket, script):
    """Never proxy a recording through a function billed by the second."""
    detail = _upload(client, script).json()
    with SessionLocal() as session:
        conversation = session.get(Conversation, detail["id"])
        conversation.audio_key = "audio/remote.wav"
        session.commit()
    bucket.objects["audio/remote.wav"] = 100

    response = client.get(
        f"/api/conversations/{detail['id']}/audio", follow_redirects=False
    )
    assert response.status_code == 307
    assert response.headers["location"].startswith("https://bucket.example/audio/remote.wav")


def test_the_webhook_needs_the_shared_secret(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "webhook_secret", "s3cret")
    response = client.post(
        "/api/webhooks/transcription", json={"transcript_id": "job-1", "status": "completed"}
    )
    assert response.status_code == 401


def test_the_webhook_ignores_a_job_it_does_not_know(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "webhook_secret", "")
    response = client.post(
        "/api/webhooks/transcription", json={"transcript_id": "nope", "status": "completed"}
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True, "matched": False}


def test_the_webhook_rejects_a_body_with_no_job_id(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "webhook_secret", "")
    assert client.post("/api/webhooks/transcription", json={"status": "completed"}).status_code == 400


def test_the_webhook_skips_intermediate_statuses(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "webhook_secret", "")
    response = client.post(
        "/api/webhooks/transcription", json={"transcript_id": "job-1", "status": "processing"}
    )
    assert response.json()["ignored"] == "processing"


def test_resume_is_safe_on_a_finished_conversation(client, script):
    detail = _upload(client, script).json()
    response = client.post(f"/api/conversations/{detail['id']}/resume")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
