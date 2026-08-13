import io

from sqlalchemy.orm import Session as SASession


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


def test_media_row_is_committed_before_task_is_enqueued(client, monkeypatch):
    """I1: a real worker can dequeue and look up the row before the API's
    own transaction would otherwise commit (in get_db, after the response),
    so the row must be committed BEFORE analyze_media.delay() is called.

    We assert this via call ordering rather than "does a fresh session see
    the row", because the test DB is SQLite with a StaticPool -- there is
    only one underlying connection for the whole test, so an uncommitted
    flush is already visible to every "fresh" session too. Only ordering
    of Session.commit() vs. delay() actually distinguishes the fixed
    behavior (commit, then enqueue) from the bug (enqueue, then commit
    only after the response via get_db).
    """
    events = []

    original_commit = SASession.commit

    def tracking_commit(self):
        events.append("commit")
        return original_commit(self)

    def fake_delay(media_id):
        events.append("delay")

    monkeypatch.setattr(SASession, "commit", tracking_commit)

    import tandemista.worker.tasks as tasks
    monkeypatch.setattr(tasks.analyze_media, "delay", staticmethod(fake_delay))

    dz = _dropzone(client)  # own commit(s), not part of the assertion below
    events.clear()

    client.post(
        f"/dropzones/{dz['id']}/media",
        files={"file": ("a.mp4", io.BytesIO(b"x"), "video/mp4")},
    )

    assert "commit" in events
    assert "delay" in events
    assert events.index("commit") < events.index("delay"), events


def test_upload_sanitizes_path_traversal_filename(client, app):
    dz = _dropzone(client)
    resp = client.post(
        f"/dropzones/{dz['id']}/media",
        files={"file": ("../../evil.mp4", io.BytesIO(b"video-bytes"), "video/mp4")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["filename"] == "evil.mp4"

    # The storage key must agree with the DB filename -- no None/mismatch.
    storage = app.state.storage
    key = f"dropzones/{dz['id']}/media/{body['id']}/evil.mp4"
    assert storage.get(key) == b"video-bytes"
