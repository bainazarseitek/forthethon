import os
import subprocess
import time


def analyze(video_path: str) -> dict:
    child = subprocess.Popen(["sleep", "300"])
    with open(os.environ["GRANDCHILD_PIDFILE"], "w") as f:
        f.write(str(child.pid))
    time.sleep(300)
    return {"events": []}
