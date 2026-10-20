"""Per-frame DINOv2-base (facebook/dinov2-base) embeddings for the 160 train/val windows (GPU host).

Same prepared 224 padded input frames as extract_clip_frames.py
(review/generated/action-5090/v1-clip/inputs), /255, DINOv2's own ImageNet mean/std
(from its preprocessor_config.json), no resize/crop (224 = 16x16 patches of 14).
Saves the final-layernorm CLS token and the mean of the 256 patch tokens, both raw
(not normalized), (160, 16, 768) float32 in prepared.json window order.
Test windows/videos are never loaded.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from transformers import Dinov2Model

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "review/generated/action-5090"
OUT = ROOT / "review/generated/autoresearch/dinov2-frames"
MODEL_ID = "facebook/dinov2-base"
REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
WEIGHTS_SHA256 = "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841"


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), "never overwrite an output dir"
    data = json.loads((SRC / "prepared.json").read_text())
    windows = data["windows"]
    assert len(windows) == 160 and {w["split"] for w in windows} == {"train", "validation"}
    path = Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / REVISION
    assert digest(path / "model.safetensors") == WEIGHTS_SHA256
    torch.manual_seed(20261017)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = Dinov2Model.from_pretrained(path, local_files_only=True, use_safetensors=True).eval().requires_grad_(False).to("cuda")
    proc = json.loads((path / "preprocessor_config.json").read_text())
    mean = torch.tensor(proc["image_mean"], device="cuda")[None, :, None, None]
    std = torch.tensor(proc["image_std"], device="cuda")[None, :, None, None]
    inputs = SRC / "v1-clip"
    cls, patch, current, frames = [], [], None, None
    for w in windows:
        if w["video"] != current:
            item = data["inputs"][w["video"]]
            assert digest(inputs / item["file"]) == item["sha256"]
            frames = np.load(inputs / item["file"])["frames"]
            current = w["video"]
        pixels = torch.from_numpy(frames[w["frames"]].copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255
        assert pixels.shape[-2:] == (224, 224)
        pixels = (pixels - mean) / std
        with torch.inference_mode():
            h = model(pixel_values=pixels).last_hidden_state  # (16, 1+256, 768), final layernorm applied
        assert h.shape[1] == 257
        cls.append(h[:, 0].float().cpu().numpy())
        patch.append(h[:, 1:].mean(1).float().cpu().numpy())
    cls, patch = np.stack(cls).astype(np.float32), np.stack(patch).astype(np.float32)
    assert cls.shape == patch.shape == (160, 16, 768) and np.isfinite(cls).all() and np.isfinite(patch).all()
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT / "dinov2.npz", cls=cls, patch=patch)
    meta = dict(model=MODEL_ID, revision=REVISION, weights_sha256=WEIGHTS_SHA256, shape=list(cls.shape),
                keys=["cls", "patch"], normalization=dict(mean=proc["image_mean"], std=proc["image_std"]),
                prepared_sha256=digest(SRC / "prepared.json"), gpu=torch.cuda.get_device_name(),
                torch=torch.__version__, script_sha256=digest(__file__), npz_sha256=digest(OUT / "dinov2.npz"),
                test_evaluated=False)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
