"""Per-frame SigLIP so400m-patch14-384 image embeddings for the 160 train/val windows (GPU host).

Frames: exactly the 16 source frames per window of prepared.json (w["source_frames"], decoded
frame index n; re-checked here against ffprobe PTS == w["source_pts"]). The prepared 224 inputs
hold a 122x224 content area, so SigLIP's 384 input would upsample them 1.7x. Instead the same
frames are re-decoded from the source video (464x848) with run_action_pilot.prepare's ffmpeg
chain at 384: select by index, bicubic aspect-preserving downscale (content 210x384), centred pad
to 384x384 with the contract pad colour 0x7b7468, rgb24. Alignment check: the same chain at 224 is
compared with the prepared 224 inputs.
Normalization: SigLIP's own preprocessor_config.json (x * 1/255, mean 0.5, std 0.5), cross-checked
against SiglipImageProcessor on the padded frames (its own resize would squash, not pad).
Embedding: get_image_features(...).pooler_output (vision attention-pool head, 1152-d), float32,
TF32 off, eager attention (fp32 SDPA on this GPU was not repeatable: max|diff| 0.04 at norm ~22),
L2-normalised per frame. Output (160, 16, 1152) float32 in prepared.json window order, key
"frames". Test windows/videos are never loaded.
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
import transformers
from transformers import SiglipImageProcessor, SiglipModel

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "review/generated/action-5090"
OUT = ROOT / "review/generated/autoresearch/siglip-frames"
MODEL_ID = "google/siglip-so400m-patch14-384"
REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
WEIGHTS_SHA256 = "ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a"  # == Hub LFS sha256
PAD = "0x7b7468"
TEST = {"X-4", "X-5", "X-26", "X-27", "X-28", "X-29", "X-34", "X-35", "X-37", "X-47", "X-48"}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def decode(source, indexes, size):
    expression = "+".join(f"eq(n\\,{i})" for i in indexes)
    filters = (f"select='{expression}',scale={size}:{size}:force_original_aspect_ratio=decrease:flags=bicubic,"
               f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color={PAD},format=rgb24")
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(source), "-vf", filters,
                                   "-fps_mode", "vfr", "-f", "rawvideo", "pipe:1"])
    frames = np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)
    assert len(frames) == len(indexes), (source, len(frames), len(indexes))
    return frames


def main():
    started = time.monotonic()
    assert not OUT.exists(), "never overwrite an output dir"
    data = json.loads((SRC / "prepared.json").read_text())
    windows = data["windows"]
    assert len(windows) == 160 and {w["split"] for w in windows} == {"train", "validation"}
    videos = sorted({w["video"] for w in windows}, key=lambda v: int(v[2:]))
    assert len(videos) == 28 and not set(videos) & TEST
    clips = {c["id"]: c for c in json.loads((ROOT / "review/generated/manifest.json").read_text())["clips"]}
    path = Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / REVISION
    assert digest(path / "model.safetensors") == WEIGHTS_SHA256
    proc = json.loads((path / "preprocessor_config.json").read_text())
    size = proc["size"]["height"]
    assert size == proc["size"]["width"] == json.loads((path / "config.json").read_text())["vision_config"]["image_size"]
    torch.manual_seed(20261017)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = SiglipModel.from_pretrained(path, local_files_only=True, use_safetensors=True,
                                        attn_implementation="eager").eval().requires_grad_(False).to("cuda")
    mean = torch.tensor(proc["image_mean"], device="cuda")[None, :, None, None]
    std = torch.tensor(proc["image_std"], device="cuda")[None, :, None, None]

    def pixels(rgb):
        x = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) * proc["rescale_factor"]
        return (x - mean) / std

    def embed(rgb):
        with torch.inference_mode():
            return model.get_image_features(pixel_values=pixels(rgb)).pooler_output.float().cpu().numpy()

    raw = np.zeros((len(windows), 16, model.config.vision_config.hidden_size), np.float32)
    align, first = {}, None
    for video in videos:
        assert video not in TEST
        source = ROOT / clips[video]["source"]
        assert digest(source) == data["sources"][video]["sha256"], video
        probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(source)])
        pts = np.array([float(f["best_effort_timestamp_time"]) for f in json.loads(probe)["frames"]])
        rows = [i for i, w in enumerate(windows) if w["video"] == video]
        for i in rows:
            assert np.allclose(pts[windows[i]["source_frames"]], windows[i]["source_pts"], atol=1e-6), (video, i)
        indexes = sorted({s for i in rows for s in windows[i]["source_frames"]})
        hi, lo = decode(source, indexes, size), decode(source, indexes, 224)
        item = data["inputs"][video]
        assert digest(SRC / "v1-clip" / item["file"]) == item["sha256"]
        prepared = np.load(SRC / "v1-clip" / item["file"])["frames"]
        assert prepared.shape == lo.shape
        d = np.abs(prepared.astype(np.int16) - lo)
        align[video] = dict(frames=len(indexes), max_abs=int(d.max()), mean_abs=float(d.mean()), exact=float((d == 0).mean()))
        mapping = {s: k for k, s in enumerate(indexes)}
        for i in rows:
            sel = [mapping[s] for s in windows[i]["source_frames"]]
            assert sel == windows[i]["frames"]
            raw[i] = embed(hi[sel])
            if i == 0:
                first = hi[sel]
        print(video, align[video], flush=True)
    assert np.isfinite(raw).all()
    frames = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
    repeat = float(np.abs(embed(first) - raw[0]).max())
    ref = SiglipImageProcessor.from_pretrained(path, local_files_only=True)(images=list(first), return_tensors="pt")
    proc_check = float((ref["pixel_values"].to("cuda") - pixels(first)).abs().max())
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT / "siglip.npz", frames=frames.astype(np.float32))
    meta = dict(
        model=MODEL_ID, revision=REVISION, weights_sha256=WEIGHTS_SHA256, shape=list(frames.shape), key="frames",
        embedding="get_image_features(...).pooler_output, L2 per frame", attn_implementation="eager",
        input=f"source frames re-decoded with ffmpeg at {size}: bicubic aspect-preserving downscale, pad {PAD}",
        normalization=dict(rescale=proc["rescale_factor"], mean=proc["image_mean"], std=proc["image_std"]),
        processor_pixel_max_abs_diff=proc_check, repeat_window0_max_abs_diff=repeat,
        raw_norm_mean=float(np.linalg.norm(raw, axis=-1).mean()),
        align_224_vs_prepared=dict(max_abs=max(a["max_abs"] for a in align.values()),
                                   mean_abs=float(np.mean([a["mean_abs"] for a in align.values()])),
                                   min_exact=min(a["exact"] for a in align.values()), per_video=align),
        ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
        prepared_sha256=digest(SRC / "prepared.json"), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        transformers=transformers.__version__, seconds=round(time.monotonic() - started, 1),
        script_sha256=digest(__file__), npz_sha256=digest(OUT / "siglip.npz"), test_evaluated=False)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "align_224_vs_prepared"} |
                     {"align": {k: v for k, v in meta["align_224_vs_prepared"].items() if k != "per_video"}}, indent=1))


if __name__ == "__main__":
    main()
