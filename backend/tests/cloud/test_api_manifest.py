def _make_dropzone(client, name="Skyranch"):
    return client.post("/dropzones/", json={"name": name}).json()


def test_create_and_list_dropzone(client):
    created = client.post("/dropzones/", json={"name": "Skyranch"})
    assert created.status_code == 201
    body = created.json()
    assert body["name"] == "Skyranch"
    assert body["currency"] == "USD"
    assert len(body["enabled_variants"]) == 4

    listed = client.get("/dropzones/")
    assert listed.status_code == 200
    assert any(d["id"] == body["id"] for d in listed.json())


def test_create_device_under_dropzone(client):
    dz = _make_dropzone(client)
    resp = client.post(
        f"/dropzones/{dz['id']}/devices",
        json={"name": "Hero12 #1", "kind": "gopro", "role": "handcam"},
    )
    assert resp.status_code == 201
    assert resp.json()["active"] is True

    devices = client.get(f"/dropzones/{dz['id']}/devices").json()
    assert len(devices) == 1


def test_create_jump_and_load(client):
    dz = _make_dropzone(client)
    load = client.post(f"/dropzones/{dz['id']}/loads", json={"name": "Load 1"})
    assert load.status_code == 201
    cust = client.post(f"/dropzones/{dz['id']}/customers", json={"name": "Jane"}).json()
    jump = client.post(
        f"/dropzones/{dz['id']}/jumps",
        json={"customer_id": cust["id"], "load_id": load.json()["id"], "video_package": True},
    )
    assert jump.status_code == 201
    assert jump.json()["video_package"] is True
    assert len(client.get(f"/dropzones/{dz['id']}/jumps").json()) == 1


def test_unknown_dropzone_404(client):
    import uuid

    resp = client.post(f"/dropzones/{uuid.uuid4()}/loads", json={"name": "x"})
    assert resp.status_code == 404
