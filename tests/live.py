import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ps() -> list[tuple[int, int, str]]:
    out = subprocess.run(["ps", "-A", "-o", "pid=,pgid=,stat="], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        pid, pgid, stat = line.split(None, 2)
        rows.append((int(pid), int(pgid), stat.strip()))
    return rows


def alive(pid: int) -> bool:
    return any(p == pid and not stat.startswith(("Z", "X")) for p, _, stat in _ps())


def group_members(pgid: int) -> list[int]:
    return [p for p, g, stat in _ps() if g == pgid and not stat.startswith(("Z", "X"))]


def get_json(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@contextmanager
def live_server(tmp_path: Path, **env):
    port = free_port()
    full_env = {**os.environ, "API_KEY": "live-key", "DATA_DIR": str(tmp_path / "data"), **{k: str(v) for k, v in env.items()}}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:create_app", "--factory", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
        env=full_env,
        stdout=subprocess.DEVNULL,
        stderr=open(tmp_path / "server.log", "w"),
        start_new_session=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                if get_json(f"{base}/healthz")[0] == 200:
                    break
            except OSError:
                pass
            if proc.poll() is not None:
                raise AssertionError((tmp_path / "server.log").read_text()[-2000:])
            time.sleep(0.1)
        else:
            raise AssertionError("live server did not become healthy")
        yield proc, base, port
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
