import os
import time

from app import detector_stub


def analyze(video_path: str) -> dict:
    flag = os.environ.get("SLOW_FLAG", "")
    end = time.monotonic() + 30
    while flag and os.path.exists(flag) and time.monotonic() < end:
        time.sleep(0.02)
    return detector_stub.analyze(video_path)
