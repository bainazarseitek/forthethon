"""Per-temporal-position VideoMAEv2 token means for the 160 train/val windows (GPU host).

Reuses run_action_videomaev2.infer preprocessing and model loading exactly (same vendor
module, strict official K710 distilled ViT-B weights, cached full-frame 224 inputs from
review/generated/action-pilot/inputs, /255, ImageNet mean/std, float32, no TF32).
The final block's 1568 tokens (patch_embed order T'=8, H'=14, W'=14) are captured with a
forward hook during the official forward_features call and averaged spatially:
  raw      (160, 8, 768)  spatial mean per temporal position, BEFORE fc_norm
  temporal (160, 8, 768)  fc_norm(raw) applied per temporal position (LayerNorm, no L2)
Exact identity: fc_norm(raw.mean(1)) == forward_features output (mean of all tokens then fc_norm).
Windows in review/generated/action-5090/prepared.json order. Test windows/videos are never loaded.

Usage (remote): PYTHONPATH=.videomaev2-deps:scripts python3 scripts/autoresearch/extract_v2_tokens.py PREPARED REF_NPZ OUT_DIR
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "review/generated/action-pilot"
CONFIG = ROOT / "data/action_videomaev2.json"
VENDOR = ROOT / "scripts/vendor/videomaev2/modeling_finetune.py"


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(prepared_path, ref_path, out):
    import torch
    import timm
    from huggingface_hub import hf_hub_download

    out = Path(out)
    assert not out.exists(), "never overwrite an output dir"
    data = json.loads(Path(prepared_path).read_text())
    windows = data["windows"]
    assert len(windows) == 160 and {w["split"] for w in windows} == {"train", "validation"}
    assert all(s["split"] != "test" for s in data["sources"].values())
    cfg = json.loads(CONFIG.read_text())
    assert cfg == data["v2"] and digest(VENDOR) == cfg["implementation_sha256"]
    assert torch.cuda.is_available() and timm.__version__ == cfg["timm"]
    torch.manual_seed(data["config"]["seed"]); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    weights = Path(hf_hub_download(cfg["repo"], cfg["filename"], revision=cfg["revision"], local_files_only=True))
    spec = importlib.util.spec_from_file_location("fortethon_official_videomaev2", VENDOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = checkpoint.get("module", checkpoint.get("model", checkpoint))
    assert state["head.weight"].shape == (710, 768)
    model = module.vit_base_patch16_224(num_classes=710, all_frames=16, tubelet_size=2, use_mean_pooling=True)
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    model = model.eval().requires_grad_(False).to("cuda")
    mean = torch.tensor(cfg["image_mean"], device="cuda")[None, :, None, None, None]
    std = torch.tensor(cfg["image_std"], device="cuda")[None, :, None, None, None]
    captured = {}
    model.blocks[-1].register_forward_hook(lambda m, i, o: captured.__setitem__("x", o))

    raw, feats, current, frames = [], [], None, None
    for w in windows:
        if w["video"] != current:
            item = data["inputs"][w["video"]]
            assert digest(SOURCE / item["file"]) == item["sha256"]
            frames = np.load(SOURCE / item["file"])["frames"]
            current = w["video"]
        pixels = torch.from_numpy(frames[w["frames"]].copy()).permute(3, 0, 1, 2)[None].to("cuda", torch.float32) / 255
        assert pixels.shape == (1, 3, 16, 224, 224)
        pixels = (pixels - mean) / std
        with torch.inference_mode():
            vector = model.forward_features(pixels)
            tokens = captured["x"]
            assert tokens.shape == (1, 1568, 768)
            per_t = tokens.reshape(1, 8, 196, 768).mean(2)
            assert torch.allclose(model.fc_norm(per_t.mean(1)), vector, atol=1e-5, rtol=1e-5)
        raw.append(per_t.cpu().numpy()[0]); feats.append(vector.cpu().numpy()[0])
    raw = np.stack(raw).astype(np.float32)
    with torch.inference_mode():
        temporal = model.fc_norm(torch.from_numpy(raw).cuda()).cpu().numpy().astype(np.float32)
        recon = model.fc_norm(torch.from_numpy(raw.mean(1)).cuda()).cpu().numpy()
    assert raw.shape == temporal.shape == (160, 8, 768) and np.isfinite(temporal).all()
    ref = np.load(ref_path)["videomaev2"]

    def cos(a):
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        return (a * ref).sum(1) / np.linalg.norm(ref, axis=1)

    c_exact, c_post, c_live = cos(recon), cos(temporal.mean(1)), cos(np.stack(feats))
    out.mkdir(parents=True)
    np.savez_compressed(out / "v2tokens.npz", temporal=temporal, raw=raw)
    meta = dict(shape=list(temporal.shape),
                min_cosine_fcnorm_of_mean_raw_vs_saved=float(c_exact.min()),
                min_cosine_live_forward_features_vs_saved=float(c_live.min()),
                min_cosine_mean_of_per_position_fcnorm_vs_saved=float(c_post.min()),
                mean_cosine_mean_of_per_position_fcnorm_vs_saved=float(c_post.mean()),
                prepared_sha256=digest(prepared_path), ref_sha256=digest(ref_path), weights_sha256=digest(weights),
                implementation_sha256=digest(VENDOR), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                script_sha256=digest(__file__), npz_sha256=digest(out / "v2tokens.npz"), test_evaluated=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
