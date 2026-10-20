import ctypes
import inspect
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from multiprocessing.connection import Connection

log = logging.getLogger("detector")

PR_SET_PDEATHSIG = 1
WATCHDOG_INTERVAL_SEC = 0.5


class DetectorTimeout(Exception):
    pass


class DetectorNotReady(DetectorTimeout):
    pass


class DetectorError(Exception):
    pass


def _kill_own_group() -> None:
    try:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    finally:
        os._exit(1)


def _watch_parent(parent_pid: int) -> None:
    while True:
        if os.getppid() != parent_pid:
            _kill_own_group()
        time.sleep(WATCHDOG_INTERVAL_SEC)


def _start_sentinel(conn: Connection) -> None:
    detector_pid = os.getpid()
    if os.fork() != 0:
        return
    try:
        conn.close()
        while os.getppid() == detector_pid:
            time.sleep(WATCHDOG_INTERVAL_SEC)
    finally:
        _kill_own_group()


def _bind_to_parent(parent_pid: int, pdeathsig: bool, conn: Connection) -> None:
    try:
        os.setpgid(0, 0)
    except OSError:
        pass
    if pdeathsig and sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    if os.getppid() != parent_pid:
        _kill_own_group()
    _start_sentinel(conn)
    threading.Thread(target=_watch_parent, args=(parent_pid,), name="parent-watchdog", daemon=True).start()


def _serve(name: str, conn: Connection, parent_pid: int, pdeathsig: bool) -> None:
    _bind_to_parent(parent_pid, pdeathsig, conn)
    from app.pipeline import load_detector

    try:
        analyze = load_detector(name)
    except BaseException as exc:
        conn.send(("load_error", f"{type(exc).__name__}: {exc}"))
        return
    conn.send(("ready", None))
    try:
        takes_budget = "budget_sec" in inspect.signature(analyze).parameters
    except (TypeError, ValueError):
        takes_budget = False
    while True:
        try:
            path, budget = conn.recv()
        except EOFError:
            return
        try:
            conn.send(("ok", analyze(path, budget_sec=budget) if takes_budget else analyze(path)))
        except BaseException as exc:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))


class DetectorProcess:
    def __init__(self, name: str, pdeathsig: bool = True):
        self.name = name
        self.pdeathsig = pdeathsig
        self._ctx = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._proc = None
        self._conn: Connection | None = None
        self._ready = False
        self._closed = False

    def start(self) -> None:
        if self._closed:
            return
        parent, child = self._ctx.Pipe()
        proc = self._ctx.Process(target=_serve, args=(self.name, child, os.getpid(), self.pdeathsig), name="detector", daemon=True)
        proc.start()
        child.close()
        try:
            os.setpgid(proc.pid, proc.pid)
        except OSError:
            pass
        self._proc, self._conn, self._ready = proc, parent, False

    @staticmethod
    def _kill_group(proc) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.kill()
        proc.join(timeout=5)

    def kill(self) -> None:
        if self._proc is not None:
            self._kill_group(self._proc)
        if self._conn is not None:
            self._conn.close()
        self._proc, self._conn, self._ready = None, None, False

    def stop(self) -> None:
        self._closed = True
        proc = self._proc
        if proc is not None:
            self._kill_group(proc)

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    @property
    def loaded(self) -> bool:
        if self._ready or not self.alive or not self._lock.acquire(blocking=False):
            return self.alive and self._ready
        try:
            self._check_ready()
            return self._ready
        finally:
            self._lock.release()

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def _check_ready(self, timeout: float = 0) -> None:
        if self._ready or self._conn is None or not self._conn.poll(timeout):
            return
        try:
            kind, payload = self._conn.recv()
        except EOFError:
            self.kill()
            return
        if kind == "ready":
            self._ready = True
        else:
            log.error("detector failed to load: %s", payload)
            self.kill()

    def wait_ready(self, timeout: float) -> bool:
        with self._lock:
            self._check_ready(timeout)
            return self._ready

    def _receive(self, deadline: float) -> tuple[str, object]:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._conn.poll(remaining):
            raise DetectorTimeout()
        try:
            return self._conn.recv()
        except EOFError as exc:
            raise DetectorError(f"detector process exited with code {self._proc.exitcode}") from exc

    def analyze(self, path: str, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        with self._lock:
            if not self.alive:
                self.kill()
                self.start()
            if self._proc is None:
                raise DetectorError("detector is stopped")
            self._check_ready(max(0.0, deadline - time.monotonic()))
            if self._proc is None:
                raise DetectorError("detector failed to load")
            if not self._ready:
                raise DetectorNotReady()
            try:
                self._conn.send((path, max(0.0, deadline - time.monotonic())))
                kind, payload = self._receive(deadline)
            except BaseException:
                self.kill()
                self.start()
                raise
            if kind == "error":
                raise DetectorError(payload)
            return payload
