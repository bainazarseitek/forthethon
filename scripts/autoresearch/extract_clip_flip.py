"""Per-frame L2 CLIP ViT-L/14 image projections of HORIZONTALLY FLIPPED frames, 160 train/val windows (GPU host).

Same encoder path as extract_clip_frames.py / run_action_pilot.infer: the prepared full-frame 224 inputs
(review/generated/action-pilot/inputs, sha256 checked against review/generated/action-5090/prepared.json),
/255, native CLIP mean/std, visual_projection(pooler_output), L2 per frame, float32, TF32 off.

Flip: the aspect-preserving padding is NOT left/right symmetric. The 464x848 sources scale to 123 content
columns and ffmpeg pads at x = floor((224-123)/2) = 50, so there are 50 pad columns on the left and 51 on
the right. Flipping the whole padded array would shift the content 1 px right. Instead the content
columns [L:R) (found per video from the exact pad colour 0x7b7468; all 28 videos give [50:173)) are
mirrored in place, which equals scale+pad of the horizontally flipped source.

Outputs go to a new dir review/generated/autoresearch/clip-flip/ (never overwritten; written to a temp
dir, then renamed):
  clipflip.npz  frames (160, 16, 768) float32, flipped, in prepared.json window order
  sanity.npz    rows (28,) first window (list position) of each video; orig (28, 16, 768) unflipped
                re-encode (compare with the 5090 clipframes.npz); naive (28, 16, 768) whole-array flip
  meta.json     provenance
Test windows/videos are never loaded.

Usage (remote): python3 scripts/autoresearch/extract_clip_flip.py
"""
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "review/generated/action-5090"
INPUTS = ROOT / "review/generated/action-pilot"
OUT = ROOT / "review/generated/autoresearch/clip-flip"
PAD = np.array([0x7B, 0x74, 0x68], np.uint8)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def content_bounds(frames):
    """[L, R): the columns that are not exactly the pad colour in every frame and row."""
    pad = (frames == PAD).all(axis=(0, 1, 3))
    cols = np.where(~pad)[0]
    left, right = int(cols[0]), int(cols[-1]) + 1
    assert pad[:left].all() and pad[right:].all() and not pad[left:right].any()
    return left, right


def flip(frames, left, right):
    out = frames.copy()
    out[:, :, left:right] = frames[:, :, left:right][:, :, ::-1]
    return out


def main():
    import torch
    import transformers
    from transformers import CLIPModel

    assert not OUT.exists(), "never overwrite an output dir"
    data = json.loads((SRC / "prepared.json").read_text())
    windows = data["windows"]
    assert len(windows) == 160 and {w["split"] for w in windows} == {"train", "validation"}
    assert all(s["split"] != "test" for s in data["sources"].values())
    assert set(data["inputs"]) == {w["video"] for w in windows} and len(data["inputs"]) == 28
    spec = data["config"]["models"]["clip"]
    pinned = {m["name"]: m for m in json.loads((ROOT / "data/gpu_models.json").read_text())}["clip"]
    assert pinned["repo"] == spec["id"] and pinned["revision"] == spec["revision"]
    path = Path.home() / ".cache/huggingface/hub" / ("models--" + spec["id"].replace("/", "--")) / "snapshots" / spec["revision"]
    weights_sha = digest(path / "model.safetensors")
    assert weights_sha == pinned["sha256"]
    torch.manual_seed(20261017)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = CLIPModel.from_pretrained(path, local_files_only=True, use_safetensors=True).eval().requires_grad_(False).to("cuda")
    proc = json.loads((path / "preprocessor_config.json").read_text())
    mean = torch.tensor(proc["image_mean"], device="cuda")[None, :, None, None]
    std = torch.tensor(proc["image_std"], device="cuda")[None, :, None, None]

    def encode(x):  # (16, 224, 224, 3) uint8 -> (16, 768) float32, L2 per frame
        pixels = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255
        pixels = (pixels - mean) / std
        with torch.inference_mode():
            image = model.visual_projection(model.vision_model(pixel_values=pixels).pooler_output)
            return torch.nn.functional.normalize(image, dim=-1).float().cpu().numpy()

    started = time.monotonic()
    bounds, out, sanity, current, frames = {}, [], {}, None, None
    for row, w in enumerate(windows):
        if w["video"] != current:
            item = data["inputs"][w["video"]]
            assert digest(INPUTS / item["file"]) == item["sha256"]
            frames = np.load(INPUTS / item["file"])["frames"]
            assert frames.shape == (item["frames"], 224, 224, 3) and frames.dtype == np.uint8
            current = w["video"]
            bounds[current] = content_bounds(frames)
        x = frames[w["frames"]]
        out.append(encode(flip(x, *bounds[current])))
        if current not in sanity:
            sanity[current] = (row, encode(x), encode(x[:, :, ::-1]))
    seconds = time.monotonic() - started
    arr = np.stack(out).astype(np.float32)
    assert arr.shape == (160, 16, 768) and np.isfinite(arr).all()
    rows = np.array([s[0] for s in sanity.values()])
    orig = np.stack([s[1] for s in sanity.values()]).astype(np.float32)
    naive = np.stack([s[2] for s in sanity.values()]).astype(np.float32)

    def l2mean(a):
        m = a.mean(1)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    flip_vs_orig = (l2mean(arr[rows]) * l2mean(orig)).sum(1)
    naive_vs_flip = (naive * arr[rows]).sum(2)
    tmp = Path(tempfile.mkdtemp(dir=OUT.parent, prefix=".clip-flip-tmp-"))
    np.savez_compressed(tmp / "clipflip.npz", frames=arr)
    np.savez_compressed(tmp / "sanity.npz", rows=rows, videos=np.array(list(sanity)), orig=orig, naive=naive)
    meta = dict(shape=list(arr.shape), content_columns={v: list(b) for v, b in bounds.items()},
                pad_left_right={v: [b[0], 224 - b[1]] for v, b in bounds.items()},
                flip="content columns [L:R) mirrored in place (pad layout kept); equals scale+pad of the flipped source",
                sanity_rows=rows.tolist(),
                sanity_window_mean_cos_flip_vs_orig_min=float(flip_vs_orig.min()),
                sanity_window_mean_cos_flip_vs_orig_mean=float(flip_vs_orig.mean()),
                naive_whole_array_flip_vs_content_flip_frame_cos_min=float(naive_vs_flip.min()),
                naive_whole_array_flip_vs_content_flip_frame_cos_mean=float(naive_vs_flip.mean()),
                encode_seconds=seconds, prepared_sha256=digest(SRC / "prepared.json"),
                model=spec, weights_sha256=weights_sha, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                transformers=transformers.__version__, script_sha256=digest(__file__),
                npz_sha256=digest(tmp / "clipflip.npz"), sanity_sha256=digest(tmp / "sanity.npz"),
                test_loaded=False, test_evaluated=False)
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
    os.rename(tmp, OUT)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
