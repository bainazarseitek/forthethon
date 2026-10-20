"""Per-video CLIP "scene embedding" for the 28 train/val videos (GPU host).

Label-free: frames sampled uniformly over the WHOLE video at 2 fps (targets t = 0, 0.5, 1, ...
< duration), floor lookup on real source PTS (ffprobe best_effort_timestamp_time), ignoring
annotations. Preprocessing identical to run_action_pilot.prepare/infer: ffmpeg select by frame
index, bicubic aspect-preserving scale to 224, pad color 0x7b7468, rgb24, /255, native CLIP
mean/std, visual_projection(pooler_output), L2 per frame.
Saves per-video mean (of L2 frame embeddings, not re-normalized) and std, plus a (160, 768)
array mapping each window (prepared.json order) to its video's scene mean.
Only videos referenced by prepared.json windows (train/validation) are decoded; test never.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from transformers import CLIPModel

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "review/generated/action-5090"
OUT = ROOT / "review/generated/autoresearch/clip-scene"
FPS = 2.0
SIZE = 224
TEST = {"X-4", "X-5", "X-26", "X-27", "X-28", "X-29", "X-34", "X-35", "X-37", "X-47", "X-48"}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), "never overwrite an output dir"
    data = json.loads((SRC / "prepared.json").read_text())
    windows = data["windows"]
    assert len(windows) == 160 and {w["split"] for w in windows} == {"train", "validation"}
    videos = sorted({w["video"] for w in windows}, key=lambda v: int(v[2:]))
    assert len(videos) == 28 and not set(videos) & TEST
    clips = {c["id"]: c for c in json.loads((ROOT / "review/generated/manifest.json").read_text())["clips"]}
    spec = data["config"]["models"]["clip"]
    path = Path.home() / ".cache/huggingface/hub" / ("models--" + spec["id"].replace("/", "--")) / "snapshots" / spec["revision"]
    expected = {m["name"]: m["sha256"] for m in json.loads((ROOT / "data/gpu_models.json").read_text())}
    assert digest(path / "model.safetensors") == expected["clip"]
    torch.manual_seed(20261017)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = CLIPModel.from_pretrained(path, local_files_only=True, use_safetensors=True).eval().requires_grad_(False).to("cuda")
    proc = json.loads((path / "preprocessor_config.json").read_text())
    mean = torch.tensor(proc["image_mean"], device="cuda")[None, :, None, None]
    std = torch.tensor(proc["image_std"], device="cuda")[None, :, None, None]
    means, stds, info = {}, {}, {}
    for video in videos:
        assert video not in TEST
        source = ROOT / clips[video]["source"]
        assert digest(source) == data["sources"][video]["sha256"], video
        probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(source)])
        pts = np.array([float(f["best_effort_timestamp_time"]) for f in json.loads(probe)["frames"]])
        assert np.all(np.diff(pts) > 0)
        targets = np.arange(max(pts[0], 0.0), pts[-1] + 1e-7, 1 / FPS)
        indexes = np.unique(np.searchsorted(pts, targets + 1e-7, side="right") - 1)
        assert indexes.min() >= 0
        expression = "+".join(f"eq(n\\,{i})" for i in indexes)
        filters = (f"select='{expression}',scale={SIZE}:{SIZE}:force_original_aspect_ratio=decrease:flags=bicubic,"
                   f"pad={SIZE}:{SIZE}:(ow-iw)/2:(oh-ih)/2:color=0x7b7468,format=rgb24")
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(source), "-vf", filters,
                                       "-fps_mode", "vfr", "-f", "rawvideo", "pipe:1"])
        frames = np.frombuffer(raw, np.uint8).reshape(-1, SIZE, SIZE, 3)
        assert len(frames) == len(indexes), (video, len(frames), len(indexes))
        embs = []
        for b in range(0, len(frames), 64):
            pixels = torch.from_numpy(frames[b:b + 64].copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255
            pixels = (pixels - mean) / std
            with torch.inference_mode():
                image = model.visual_projection(model.vision_model(pixel_values=pixels).pooler_output)
                embs.append(torch.nn.functional.normalize(image, dim=-1).float().cpu().numpy())
        e = np.concatenate(embs)
        means[video], stds[video] = e.mean(0), e.std(0)
        info[video] = dict(frames=int(len(e)), duration=float(pts[-1]), split=data["sources"][video]["split"])
        print(video, info[video], flush=True)
    mean_arr = np.stack([means[w["video"]] for w in windows]).astype(np.float32)
    std_arr = np.stack([stds[w["video"]] for w in windows]).astype(np.float32)
    assert mean_arr.shape == (160, 768) and np.isfinite(mean_arr).all() and np.isfinite(std_arr).all()
    ref = np.load(SRC / "features.npz")["clip"]
    cos = (ref * mean_arr).sum(1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(mean_arr, axis=1)
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT / "clipscene.npz", mean=mean_arr, std=std_arr)
    np.savez_compressed(OUT / "per_video.npz", videos=np.array(videos),
                        mean=np.stack([means[v] for v in videos]), std=np.stack([stds[v] for v in videos]))
    meta = dict(fps=FPS, videos=info, window_vs_scene_cosine_mean=float(cos.mean()), window_vs_scene_cosine_min=float(cos.min()),
                prepared_sha256=digest(SRC / "prepared.json"), weights_sha256=expected["clip"], gpu=torch.cuda.get_device_name(),
                script_sha256=digest(__file__), npz_sha256=digest(OUT / "clipscene.npz"), test_evaluated=False)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "videos"}, indent=1))


if __name__ == "__main__":
    main()
