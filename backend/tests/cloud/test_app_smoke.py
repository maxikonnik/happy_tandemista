import io


def test_end_to_end_upload_flow(client, app):
    dz = client.post("/dropzones/", json={"name": "DZ"}).json()
    client.post(
        f"/dropzones/{dz['id']}/devices",
        json={"name": "H12", "kind": "gopro", "role": "handcam"},
    )
    load = client.post(f"/dropzones/{dz['id']}/loads", json={"name": "L1"}).json()
    cust = client.post(f"/dropzones/{dz['id']}/customers", json={"name": "Jane"}).json()
    jump = client.post(
        f"/dropzones/{dz['id']}/jumps",
        json={"customer_id": cust["id"], "load_id": load["id"]},
    ).json()
    media = client.post(
        f"/dropzones/{dz['id']}/media",
        files={"file": ("handcam_001.mp4", io.BytesIO(b"bytes"), "video/mp4")},
        data={"jump_id": jump["id"]},
    )
    assert media.status_code == 201
    assert media.json()["jump_id"] == jump["id"]
