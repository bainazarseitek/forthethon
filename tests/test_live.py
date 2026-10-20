import os
import select
import signal
import socket
import subprocess
import sys
import time

import httpx

from tests.live import ROOT, alive, get_json, group_members, live_server


def wait_until(predicate, timeout: float) -> float:
    started = time.time()
    while time.time() - started < timeout:
        if predicate():
            return time.time() - started
        time.sleep(0.05)
    raise AssertionError("condition not met")


def describe(pids: list[int]) -> str:
    out = []
    for pid in pids:
        cmd = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        out.append(f"  {pid} {cmd[:90]}")
    return "\n".join(out)


def test_sigkill_of_server_leaves_no_detector_process(tmp_path, videos):
    pidfile = tmp_path / "grandchild.pid"
    with live_server(tmp_path, DETECTOR="tests.forking_detector", GRANDCHILD_PIDFILE=pidfile) as (server, base, _):
        detector_pid = get_json(f"{base}/healthz")[1]["detector_pid"]
        with open(videos["vfr"], "rb") as f:
            r = httpx.post(f"{base}/api/v1/analyze", headers={"X-API-Key": "live-key"}, files={"file": ("vfr.mov", f)})
        assert r.status_code == 200
        wait_until(pidfile.exists, 10)
        wait_until(lambda: pidfile.read_text().strip() != "", 5)
        grandchild = int(pidfile.read_text())
        members = group_members(detector_pid)
        print(f"\nserver pid {server.pid}, detector pid {detector_pid}, detector group before SIGKILL:\n{describe(members)}")
        assert detector_pid in members and grandchild in members and len(members) >= 3
        os.kill(server.pid, signal.SIGKILL)
        server.wait(timeout=5)
        elapsed = wait_until(lambda: not group_members(detector_pid) and not alive(detector_pid) and not alive(grandchild), 5)
        print(f"SIGKILL sent to {server.pid}; detector group empty after {elapsed:.2f}s")
        print(f"detector {detector_pid} alive={alive(detector_pid)}, grandchild {grandchild} alive={alive(grandchild)}, "
              f"group members={group_members(detector_pid)}")


def test_parent_watchdog_without_pdeathsig(tmp_path):
    script = (
        "import sys, time\n"
        "from app.detector_proc import DetectorProcess\n"
        "d = DetectorProcess('stub', pdeathsig=False)\n"
        "d.start()\n"
        "assert d.wait_ready(30)\n"
        "print(d.pid, flush=True)\n"
        "time.sleep(300)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", script], cwd=ROOT, stdout=subprocess.PIPE, text=True)
    try:
        detector_pid = int(parent.stdout.readline())
        members = group_members(detector_pid)
        assert detector_pid in members and len(members) == 2
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=5)
        elapsed = wait_until(lambda: not group_members(detector_pid), 3)
        print(f"\nwithout PDEATHSIG: detector group {members} gone {elapsed:.2f}s after SIGKILL of the parent")
    finally:
        if parent.poll() is None:
            parent.kill()


def test_chunked_upload_is_cut_off_while_streaming(tmp_path):
    spool = tmp_path / "tmp"
    spool.mkdir()
    with live_server(tmp_path, MAX_UPLOAD_MB=1, TMPDIR=spool) as (_, _, port):
        sock = socket.create_connection(("127.0.0.1", port))
        sock.sendall(
            b"POST /app/analyze HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n"
            b"Content-Type: multipart/form-data; boundary=b\r\n\r\n"
        )
        head = b'--b\r\nContent-Disposition: form-data; name="file"; filename="big.mov"\r\nContent-Type: video/quicktime\r\n\r\n'
        sock.sendall(b"%x\r\n%s\r\n" % (len(head), head))
        chunk = b"\0" * 65536
        sent, response = 0, b""
        try:
            while sent < 1200 * 1024 * 1024:
                if select.select([sock], [], [], 0)[0]:
                    response = sock.recv(65536)
                    break
                sock.sendall(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                sent += len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        if not response:
            sock.settimeout(5)
            response = sock.recv(65536)
        sock.close()
        spooled = sum(f.stat().st_size for f in spool.rglob("*") if f.is_file())
        print(f"\nchunked upload: server answered after {sent / 2**20:.1f} MB sent: {response.split(b'\r\n')[0].decode()}; "
              f"bytes left in TMPDIR: {spooled}")
        assert response.startswith(b"HTTP/1.1 413")
        assert b'"error":"file_too_large"' in response
        assert sent < 32 * 1024 * 1024
        assert spooled == 0
