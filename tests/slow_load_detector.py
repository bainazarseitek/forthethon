import os
import time

from app import detector_stub


def load() -> None:
    time.sleep(float(os.environ.get("LOAD_DELAY_SEC", "3")))


def analyze(video_path: str) -> dict:
    return detector_stub.analyze(video_path)
