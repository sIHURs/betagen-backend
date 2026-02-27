from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_get_unknown_job_returns_404() -> None:
    response = client.get("/api/v1/jobs/not-found")
    assert response.status_code == 404


def test_upload_rejects_non_video_extension() -> None:
    response = client.post(
        "/api/v1/videos/upload",
        files={"file": ("bad.txt", b"not-video", "text/plain")},
    )
    assert response.status_code == 400
    assert "Only .mp4 and .mov" in response.json()["detail"]


def test_process_unknown_video_returns_404() -> None:
    response = client.post("/api/v1/videos/missing-id/process")
    assert response.status_code == 404
