"""Low-level pixel motion features for the 160 train/val windows (CPU, numpy + cv2).

Input: the exact 16 prepared frames per window (review/generated/action-pilot/inputs/<video>.npz,
224 square, aspect-preserving pad; sha256 checked against action-5090/prepared.json), indexed by
window["frames"]. Test videos are never opened (only videos named by train/val windows).

Pre-registered constants (fixed before any evaluation):
  - grayscale, content area = cols 51:173 (source 464x848 scaled by 224/848, centered pad)
  - desk/hand region = data/action_crop.json [0,240,440,640] source px -> rows 63:169, cols 51:167
  - frame diff = |g[t+1]-g[t]| / 255 over the 15 consecutive pairs
  - Farneback flow (pyr 0.5, 3 levels, win 15, 3 iters, poly 5/1.2), magnitude in px
Features (d=38), all log1p'd:
  diff_grid[16]  4x4 grid over content area, time-mean of cell mean |diff|
  flow_grid[16]  4x4 grid over content area, time-mean of cell mean flow magnitude
  desk[6]        desk |diff|: time mean, time std, time max; desk flow mag: time mean, time std,
                 time-mean fraction of pixels with magnitude > 1 px
Output: review/generated/autoresearch/features/pixmotion.npz key "feat" (160, 38) float32,
in prepared.json window order.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "review/generated/action-5090/prepared.json"
INPUTS = ROOT / "review/generated/action-pilot"
OUT = ROOT / "review/generated/autoresearch/features/pixmotion.npz"
TEST = {"X-4", "X-5", "X-26", "X-27", "X-28", "X-29", "X-34", "X-35", "X-37", "X-47", "X-48"}

SCALE = 224 / 848
C0 = round((224 - 464 * SCALE) / 2)  # 51
CONTENT = (slice(0, 224), slice(C0, C0 + 122))
x0, y0, x1, y1 = json.loads((ROOT / "data/action_crop.json").read_text())["crop_xyxy"]
DESK = (slice(round(y0 * SCALE), round(y1 * SCALE)), slice(C0 + round(x0 * SCALE), C0 + round(x1 * SCALE)))


def grid(a, n=4):
    """a: (T, H, W) -> (T, n*n) cell means."""
    hs = np.array_split(np.arange(a.shape[1]), n)
    ws = np.array_split(np.arange(a.shape[2]), n)
    return np.stack([a[:, h][:, :, w].mean((1, 2)) for h in hs for w in ws], 1)


def window_features(rgb):
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in rgb]).astype(np.float32)
    diff = np.abs(np.diff(g, axis=0)) / 255
    mag = np.stack([
        np.linalg.norm(cv2.calcOpticalFlowFarneback(g[t].astype(np.uint8), g[t + 1].astype(np.uint8),
                                                    None, 0.5, 3, 15, 3, 5, 1.2, 0), axis=-1)
        for t in range(len(g) - 1)])
    dc, mc = diff[:, CONTENT[0], CONTENT[1]], mag[:, CONTENT[0], CONTENT[1]]
    dd, md = diff[:, DESK[0], DESK[1]], mag[:, DESK[0], DESK[1]]
    dd_t, md_t = dd.mean((1, 2)), md.mean((1, 2))
    desk = [dd_t.mean(), dd_t.std(), dd_t.max(), md_t.mean(), md_t.std(), (md > 1).mean((1, 2)).mean()]
    return np.log1p(np.concatenate([grid(dc).mean(0), grid(mc).mean(0), desk]))


def main():
    data = json.loads(SRC.read_text())
    windows = data["windows"]
    assert len(windows) == 160 and {w["split"] for w in windows} == {"train", "validation"}
    out, current, frames = [], None, None
    for w in windows:
        if w["video"] != current:
            assert w["video"] not in TEST
            item = data["inputs"][w["video"]]
            path = INPUTS / item["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
            frames = np.load(path)["frames"]
            current = w["video"]
        assert len(w["frames"]) == 16
        out.append(window_features(frames[w["frames"]]))
    feat = np.stack(out).astype(np.float32)
    assert feat.shape == (160, 38) and np.isfinite(feat).all()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, feat=feat)
    print(OUT, feat.shape, "content", CONTENT, "desk", DESK)


if __name__ == "__main__":
    main()
