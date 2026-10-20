"""Per-frame L2 CLIP ViT-L/14 image projections for the 160 train/val windows (GPU host).

Reuses run_action_pilot.infer preprocessing exactly: the same prepared input frames
(review/generated/action-5090/v1-clip/inputs, 224 padded), /255, native CLIP mean/std,
visual_projection(pooler_output), L2 per frame. Output (160, 16, 768) float32 in
prepared.json window order. Test windows/videos are never loaded.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from transformers import CLIPModel

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "review/generated/action-5090"
OUT = ROOT / "review/generated/autoresearch/clip-frames"


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), "never overwrite an output dir"
    data = json.loads((SRC / "prepared.json").read_text())
    windows = data["windows"]
    assert len(windows) == 160 and {w["split"] for w in windows} == {"train", "validation"}
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
    inputs = SRC / "v1-clip"
    out, current, frames = [], None, None
    for w in windows:
        if w["video"] != current:
            item = data["inputs"][w["video"]]
            assert digest(inputs / item["file"]) == item["sha256"]
            frames = np.load(inputs / item["file"])["frames"]
            current = w["video"]
        pixels = torch.from_numpy(frames[w["frames"]].copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255
        pixels = (pixels - mean) / std
        with torch.inference_mode():
            image = model.visual_projection(model.vision_model(pixel_values=pixels).pooler_output)
            out.append(torch.nn.functional.normalize(image, dim=-1).float().cpu().numpy())
    frames_arr = np.stack(out).astype(np.float32)
    assert frames_arr.shape == (160, 16, 768) and np.isfinite(frames_arr).all()
    ref = np.load(SRC / "features.npz")["clip"]
    m = frames_arr.mean(1)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    cos = (m * ref).sum(1) / np.linalg.norm(ref, axis=1)
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT / "clipframes.npz", frames=frames_arr)
    meta = dict(shape=list(frames_arr.shape), min_cosine_vs_saved_clip=float(cos.min()), mean_cosine=float(cos.mean()),
                prepared_sha256=digest(SRC / "prepared.json"), weights_sha256=expected["clip"], gpu=torch.cuda.get_device_name(),
                script_sha256=digest(__file__), npz_sha256=digest(OUT / "clipframes.npz"), test_evaluated=False)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
