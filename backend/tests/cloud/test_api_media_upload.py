import io


def _dropzone(client):
    return client.post("/dropzones/", json={"name": "DZ"}).json()


def test_upload_registers_media_and_stores_file(client, app):
    dz = _dropzone(client)
    resp = client.post(
        f"/dropzones/{dz['id']}/media",
        files={"file": ("handcam_001.mp4", io.BytesIO(b"video-bytes"), "video/mp4")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["filename"] == "handcam_001.mp4"
    assert body["status"] == "registered"
    assert len(body["locations"]) == 1

    # The bytes actually landed in the app's storage backend.
    storage = app.state.storage
    key = f"dropzones/{dz['id']}/media/{body['id']}/handcam_001.mp4"
    assert storage.get(key) == b"video-bytes"


def test_get_media_by_id(client):
    dz = _dropzone(client)
    created = client.post(
        f"/dropzones/{dz['id']}/media",
        files={"file": ("a.mp4", io.BytesIO(b"x"), "video/mp4")},
    ).json()
    got = client.get(f"/dropzones/{dz['id']}/media/{created['id']}")
    assert got.status_code == 200
    assert got.json()["id"] == created["id"]


def test_upload_enqueues_analysis(client, monkeypatch):
    calls = {}

    def fake_delay(media_id):
        calls["media_id"] = media_id

    import tandemista.worker.tasks as tasks
    monkeypatch.setattr(tasks.analyze_media, "delay", staticmethod(fake_delay))

    dz = _dropzone(client)
    created = client.post(
        f"/dropzones/{dz['id']}/media",
        files={"file": ("a.mp4", io.BytesIO(b"x"), "video/mp4")},
    ).json()
    assert calls["media_id"] == created["id"]
