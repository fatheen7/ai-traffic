import os

from fastapi.testclient import TestClient

from api.index import app

client = TestClient(app)
MEDIA = os.path.join(os.path.dirname(__file__), "..", "public", "media")


def test_health():
    assert client.get("/api/health").json()["ok"] is True


def test_plan():
    r = client.post("/api/plan", json={"counts": [24, 14, 10, 2]})
    assert r.status_code == 200 and r.json()["greens"] == [58, 34, 24, 10]


def test_plan_rejects_negative_counts():
    assert client.post("/api/plan", json={"counts": [5, -1]}).status_code == 400


def test_count_sample_clip():
    with open(os.path.join(MEDIA, "sample-north-20s.mp4"), "rb") as f:
        r = client.post("/api/count", files={"video": ("clip.mp4", f, "video/mp4")})
    body = r.json()
    assert r.status_code == 200 and body["count"] > 0 and len(body["keyframes"]) == 3


def test_siren_sample_recording():
    with open(os.path.join(MEDIA, "sample-siren-35s.wav"), "rb") as f:
        r = client.post("/api/siren", files={"audio": ("s.wav", f, "audio/wav")})
    assert r.status_code == 200 and len(r.json()["intervals"]) == 1


def test_bad_video_is_a_clear_error():
    r = client.post("/api/count", files={"video": ("x.mp4", b"not a video", "video/mp4")})
    assert r.status_code == 400
