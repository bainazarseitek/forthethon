import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

KEY = "test-key"


def make_vfr(path: Path, seconds: int = 4, drop_every: int = 7) -> Path:
    vf = (
        "format=yuv420p,"
        "geq=lum='if(lt(X,64)*lt(Y,64),16+mod(N,25)*8,lum(X,Y))':cb='cb(X,Y)':cr='cr(X,Y)',"
        f"select='not(eq(mod(n,{drop_every}),3))'"
    )
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={seconds}",
            "-vf", vf, "-fps_mode", "vfr",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-video_track_timescale", "30000",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture(scope="session")
def videos(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("videos")
    vfr = make_vfr(root / "vfr.mov")
    rotated = root / "rotated.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(vfr), "-c", "copy", str(rotated)], check=True)
    return {"vfr": vfr, "rotated": rotated}


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def factory(**overrides) -> TestClient:
        settings = Settings(**{"api_key": KEY, "data_dir": tmp_path / "data", **overrides})
        client = TestClient(create_app(settings))
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


def upload(client: TestClient, path: Path, key: str | None = KEY, url: str = "/api/v1/analyze", name: str | None = None):
    headers = {"X-API-Key": key} if key is not None else {}
    with open(path, "rb") as f:
        return client.post(url, headers=headers, files={"file": (name or path.name, f, "video/quicktime")})


def wait_done(client: TestClient, job_id: str, url: str = "/api/v1/jobs/{}", timeout: float = 30) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(url.format(job_id), headers={"X-API-Key": KEY}).json()
        if body["status"] != "processing":
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still processing")


def assert_manual_review(body: dict, error: str) -> None:
    assert body["status"] == "done"
    assert body["verdict"] == "violation"
    assert body["error"] == error
    [violation] = body["violations"]
    assert violation["type_id"] == "other"
    assert violation["type"] == "Требуется ручная проверка"
    assert "ручной проверки" in violation["description"]
