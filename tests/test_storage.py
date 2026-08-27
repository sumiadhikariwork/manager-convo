"""Storage backends: local disk, and S3-compatible with presigned URLs."""

from __future__ import annotations

import io

import pytest

from app.config import Settings
from app.storage import (
    LocalStorage,
    S3Storage,
    StorageError,
    get_storage,
    guess_content_type,
)


class FakeS3Client:
    """Stands in for boto3's S3 client, recording what it was asked to sign."""

    def __init__(self, objects: dict[str, int] | None = None):
        self.objects = objects if objects is not None else {}
        self.calls: list[tuple[str, dict]] = []
        self.deleted: list[str] = []

    def generate_presigned_url(self, operation, Params, ExpiresIn):  # noqa: N803
        self.calls.append((operation, dict(Params)))
        return f"https://s3.example/{Params['Bucket']}/{Params['Key']}?op={operation}&x={ExpiresIn}"

    def head_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise RuntimeError("NoSuchKey")
        return {"ContentLength": self.objects[Key]}

    def upload_fileobj(self, stream, Bucket, Key, ExtraArgs=None):  # noqa: N803
        self.objects[Key] = len(stream.read())

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.deleted.append(Key)
        self.objects.pop(Key, None)


def make_s3(objects=None, public_base_url="") -> tuple[S3Storage, FakeS3Client]:
    client = FakeS3Client(objects)
    storage = S3Storage.__new__(S3Storage)
    storage.bucket = "recordings"
    storage.public_base_url = public_base_url
    storage._client = client
    return storage, client


# -- local -----------------------------------------------------------------

def test_local_round_trip(tmp_path):
    storage = LocalStorage(tmp_path)
    written = storage.put("audio/a.wav", io.BytesIO(b"0123456789"), "audio/wav")
    assert written == 10
    assert storage.exists("audio/a.wav")
    assert storage.size("audio/a.wav") == 10
    assert storage.local_path("audio/a.wav").read_bytes() == b"0123456789"

    storage.delete("audio/a.wav")
    assert not storage.exists("audio/a.wav")


def test_local_has_no_direct_upload_and_no_public_url(tmp_path):
    """The application serves its own files, so there is nothing to hand out."""
    storage = LocalStorage(tmp_path)
    assert storage.supports_direct_upload is False
    assert storage.upload_ticket("audio/a.wav", "audio/wav").direct is False
    assert storage.playback_url("audio/a.wav") is None


def test_local_refuses_to_escape_its_root(tmp_path):
    storage = LocalStorage(tmp_path)
    with pytest.raises(StorageError, match="outside the storage root"):
        storage.exists("../../../etc/passwd")


def test_local_path_is_none_for_anything_that_is_not_a_file(tmp_path):
    storage = LocalStorage(tmp_path)
    assert storage.local_path("") is None
    assert storage.local_path("missing.wav") is None
    (tmp_path / "adir").mkdir()
    assert storage.local_path("adir") is None


def test_deleting_something_absent_is_not_an_error(tmp_path):
    LocalStorage(tmp_path).delete("never-existed.wav")


# -- s3 --------------------------------------------------------------------

def test_upload_ticket_is_a_signed_put(tmp_path):
    storage, client = make_s3()
    ticket = storage.upload_ticket("audio/a.wav", "audio/mpeg")

    assert ticket.direct is True
    assert ticket.method == "PUT"
    assert ticket.upload_url.startswith("https://s3.example/recordings/audio/a.wav")

    operation, params = client.calls[0]
    assert operation == "put_object"
    # The signature covers Content-Type, so the browser has to send exactly this.
    assert params["ContentType"] == "audio/mpeg"
    assert ticket.headers == {"Content-Type": "audio/mpeg"}


def test_playback_url_is_signed_and_time_limited():
    storage, client = make_s3()
    url = storage.playback_url("audio/a.wav", expires_in=900)
    assert "op=get_object" in url and "x=900" in url


def test_a_public_base_url_skips_signing():
    """A bucket behind a CDN is served by plain URL, not a signature."""
    storage, client = make_s3(public_base_url="https://cdn.example/")
    assert storage.playback_url("audio/a.wav") == "https://cdn.example/audio/a.wav"
    assert client.calls == []


def test_size_and_existence_come_from_the_bucket():
    storage, _ = make_s3({"audio/a.wav": 4242})
    assert storage.exists("audio/a.wav")
    assert storage.size("audio/a.wav") == 4242
    assert not storage.exists("audio/missing.wav")


def test_asking_the_size_of_a_missing_object_explains_itself():
    storage, _ = make_s3()
    with pytest.raises(StorageError, match="not in the bucket"):
        storage.size("audio/missing.wav")


def test_remote_storage_has_no_local_path():
    """This is what stops a local speech model being pointed at remote audio."""
    storage, _ = make_s3({"audio/a.wav": 1})
    assert storage.local_path("audio/a.wav") is None


def test_server_side_upload_still_works():
    storage, client = make_s3()
    assert storage.put("audio/a.wav", io.BytesIO(b"12345"), "audio/wav") == 5
    assert client.objects["audio/a.wav"] == 5


def test_delete_removes_the_object():
    storage, client = make_s3({"audio/a.wav": 1})
    storage.delete("audio/a.wav")
    assert client.deleted == ["audio/a.wav"]


# -- selection -------------------------------------------------------------

def test_backend_selection_follows_the_setting(tmp_path):
    assert get_storage(Settings(storage_backend="local", data_dir=tmp_path)).name == "local"
    with pytest.raises(StorageError, match="Unknown STORAGE_BACKEND"):
        get_storage(Settings(storage_backend="carrier-pigeon"))


def test_s3_without_a_bucket_says_so():
    with pytest.raises(StorageError, match="STORAGE_BUCKET"):
        get_storage(Settings(storage_backend="s3", storage_bucket=""))


def test_content_type_guessing():
    assert guess_content_type("a.mp3") == "audio/mpeg"
    assert guess_content_type("a.wav") in ("audio/wav", "audio/x-wav")
    assert guess_content_type("a.unknownext") == "application/octet-stream"
