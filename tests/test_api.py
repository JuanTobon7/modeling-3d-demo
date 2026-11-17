"""API: validación de fotos, token, ciclo de vida y un trabajo completo de punta a punta."""
import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scanner3d.app import create_app

TOKEN = "secreto"
H = {"X-Token": TOKEN}


def jpeg(color=(200, 50, 50), size=(64, 48)) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "JPEG")
    return b.getvalue()


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path, token=TOKEN)) as c:
        yield c


def test_token_required(client):
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/api/jobs", headers={"X-Token": "otro"}).status_code == 401
    assert client.get("/api/jobs", headers=H).status_code == 200
    assert client.get(f"/api/jobs?token={TOKEN}").status_code == 200
    assert client.get("/api/health").status_code == 200   # público
    assert client.get("/").status_code == 200              # la PWA es pública


def test_upload_validation_and_delete(client):
    job = client.post("/api/jobs", json={"name": "prueba"}, headers=H).json()
    files = [("files", ("a.jpg", jpeg(), "image/jpeg")),
             ("files", ("b.txt", b"hola", "text/plain")),
             ("files", ("c.png", jpeg(), "image/png"))]
    r = client.post(f"/api/jobs/{job['id']}/images", files=files, headers=H).json()
    assert r["accepted"] == 2 and r["total"] == 2
    assert r["rejected"][0]["file"] == "b.txt"
    # No arranca con menos de 3 fotos.
    assert client.post(f"/api/jobs/{job['id']}/start", headers=H).status_code == 400
    assert client.get("/api/jobs/../../etc", headers=H).status_code == 404
    assert client.get(f"/api/jobs/{job['id']}/files/..%2Fjob.json", headers=H).status_code == 404
    assert client.delete(f"/api/jobs/{job['id']}", headers=H).status_code == 204
    assert client.get(f"/api/jobs/{job['id']}", headers=H).status_code == 404


def test_bad_options_rejected(client):
    assert client.post("/api/jobs", json={"engine": "rm -rf"}, headers=H).status_code == 422


def test_cancel_running_job(client):
    job = client.post("/api/jobs", json={}, headers=H).json()
    files = [("files", (f"{k}.jpg", jpeg((k * 40, 90, 160)), "image/jpeg")) for k in range(4)]
    client.post(f"/api/jobs/{job['id']}/images", files=files, headers=H)
    assert client.post(f"/api/jobs/{job['id']}/start", headers=H).json()["status"] == "queued"
    r = client.post(f"/api/jobs/{job['id']}/cancel", headers=H).json()
    assert r["status"] in ("cancelled", "failed")


def test_end_to_end(client, tmp_path):
    from scanner3d.synthetic import make_dataset
    d = tmp_path / "imgs"
    make_dataset(d, views=20, width=640, height=480)
    job = client.post("/api/jobs", json={"quality": "fast"}, headers=H).json()
    paths = sorted(d.iterdir())
    for i in range(0, len(paths), 5):
        files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in paths[i:i + 5]]
        assert client.post(f"/api/jobs/{job['id']}/images", files=files, headers=H).json()["rejected"] == []
    client.post(f"/api/jobs/{job['id']}/start", headers=H)
    t0 = time.time()
    while time.time() - t0 < 400:
        j = client.get(f"/api/jobs/{job['id']}", headers=H).json()
        if j["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(1)
    assert j["status"] == "done", j["message"]
    assert "model.glb" in j["files"]
    r = client.get(f"/api/jobs/{job['id']}/files/model.glb", headers=H)
    assert r.status_code == 200 and r.content[:4] == b"glTF"
