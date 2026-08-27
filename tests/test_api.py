"""The HTTP surface, exercised through a real client."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import pipeline
from app.config import get_settings
from app.main import app
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
    assert body["analysis_provider"] == "heuristic"


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
    settings = get_settings()
    detail = _upload(client, script).json()
    audio_files = list(settings.audio_dir.glob(f"{detail['id']}.*"))
    assert audio_files

    assert client.delete(f"/api/conversations/{detail['id']}").status_code == 204
    assert client.get(f"/api/conversations/{detail['id']}").status_code == 404
    assert not (settings.audio_dir / f"{detail['id']}.wav").exists()


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
