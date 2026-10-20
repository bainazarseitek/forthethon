import json
import sys
import time

from app.detector_vlm.detector import analyze, load

load()
started = time.monotonic()
result = analyze(sys.argv[1])
result["latency_sec"] = round(time.monotonic() - started, 2)
json.dump(result, sys.stdout, ensure_ascii=False, indent=1)
