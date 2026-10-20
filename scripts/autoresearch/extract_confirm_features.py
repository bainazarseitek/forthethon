"""CLIP and SigLIP features for the confirm-v1 windows (GPU host). Usage:
    python3 scripts/autoresearch/extract_confirm_features.py <confirm_set.json> <new output dir>

Same code paths as the frozen features:
  clip   - frames decoded with run_action_pilot.prepare's ffmpeg chain at 224 (select by source frame
           index, bicubic aspect-preserving downscale, centred pad 0x7b7468, rgb24); /255, native CLIP
           mean/std; visual_projection(vision_model(.).pooler_output), L2 per frame, mean over the 16
           frames, L2 window vector  (== features.npz 'clip', run_action_pilot.infer).
  siglip - the same frames decoded at 384 with the same chain; extract_siglip_frames.py exactly
           (preprocessor rescale/mean/std, get_image_features(.).pooler_output, eager attention,
           fp32, TF32 off), L2 per frame -> (N, 16, 1152) key "frames".
Sanity: 3 frozen windows (first frozen window of X-14, X-39, X-19) go through the same code; their
224 frames are compared with the frozen prepared inputs and their features with the saved
features.npz 'clip' and autoresearch siglip.npz on this host (cos >= 0.9999 required).
Only train/validation videos are decoded; test videos are never touched.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from transformers import CLIPModel, SiglipModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from extract_siglip_frames import MODEL_ID, PAD, REVISION, TEST, WEIGHTS_SHA256, decode  # noqa: E402

SRC = ROOT / "review/generated/action-5090"
SANITY_VIDEOS = ["X-14", "X-39", "X-19"]
assert PAD == "0x7b7468"


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def snapshot(model_id, revision):
    return Path.home() / ".cache/huggingface/hub" / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision


def main():
    started = time.monotonic()
    confirm, out = Path(sys.argv[1]), Path(sys.argv[2])
    assert not out.exists(), "never overwrite an output dir"
    cs = json.loads(confirm.read_text())
    frozen = json.loads((SRC / "prepared.json").read_text())
    fresh = cs["windows"]
    pos = [next(i for i, w in enumerate(frozen["windows"]) if w["video"] == v) for v in SANITY_VIDEOS]  # row in features
    sanity = [frozen["windows"][i] for i in pos]
    jobs = [("fresh", i, w) for i, w in enumerate(fresh)] + [("sanity", i, w) for i, w in enumerate(sanity)]
    videos = sorted({w["video"] for _, _, w in jobs}, key=lambda v: int(v[2:]))
    assert not set(videos) & TEST and all(w["split"] in ("train", "validation") for _, _, w in jobs)
    clips = {c["id"]: c for c in json.loads((ROOT / "review/generated/manifest.json").read_text())["clips"]}

    torch.manual_seed(20261017)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = frozen["config"]["models"]["clip"]
    cpath = snapshot(spec["id"], spec["revision"])
    expected = {m["name"]: m["sha256"] for m in json.loads((ROOT / "data/gpu_models.json").read_text())}
    assert digest(cpath / "model.safetensors") == expected["clip"]
    spath = snapshot(MODEL_ID, REVISION)
    assert digest(spath / "model.safetensors") == WEIGHTS_SHA256
    clip = CLIPModel.from_pretrained(cpath, local_files_only=True, use_safetensors=True).eval().requires_grad_(False).to("cuda")
    sig = SiglipModel.from_pretrained(spath, local_files_only=True, use_safetensors=True,
                                      attn_implementation="eager").eval().requires_grad_(False).to("cuda")
    cp = json.loads((cpath / "preprocessor_config.json").read_text())
    sp = json.loads((spath / "preprocessor_config.json").read_text())
    cmean, cstd = (torch.tensor(cp[k], device="cuda")[None, :, None, None] for k in ("image_mean", "image_std"))
    smean, sstd = (torch.tensor(sp[k], device="cuda")[None, :, None, None] for k in ("image_mean", "image_std"))
    ssize = sp["size"]["height"]
    assert ssize == 384

    def clip_vec(rgb):
        x = (torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255 - cmean) / cstd
        with torch.inference_mode():
            e = clip.visual_projection(clip.vision_model(pixel_values=x).pooler_output)
            v = torch.nn.functional.normalize(e, dim=-1).mean(dim=0).float().cpu().numpy()
        return v / max(np.linalg.norm(v), 1e-12)

    def sig_frames(rgb):
        x = (torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) * sp["rescale_factor"] - smean) / sstd
        with torch.inference_mode():
            f = sig.get_image_features(pixel_values=x).pooler_output.float().cpu().numpy()
        return f / np.linalg.norm(f, axis=-1, keepdims=True)

    res = {"fresh": (np.zeros((len(fresh), 768), np.float32), np.zeros((len(fresh), 16, 1152), np.float32)),
           "sanity": (np.zeros((3, 768), np.float32), np.zeros((3, 16, 1152), np.float32))}
    pixel_check = {}
    for video in videos:
        source = ROOT / clips[video]["source"]
        assert digest(source) == clips[video]["sha256"], video
        local = [(k, i, w) for k, i, w in jobs if w["video"] == video]
        idx = sorted({s for _, _, w in local for s in w["source_frames"]})
        lo, hi = decode(source, idx, 224), decode(source, idx, ssize)
        m = {s: j for j, s in enumerate(idx)}
        for k, i, w in local:
            sel = [m[s] for s in w["source_frames"]]
            res[k][0][i] = clip_vec(lo[sel])
            res[k][1][i] = sig_frames(hi[sel])
            if k == "sanity":
                item = frozen["inputs"][video]
                prep = np.load(SRC / "v1-clip" / item["file"])["frames"][w["frames"]]
                d = np.abs(prep.astype(np.int16) - lo[sel])
                pixel_check[video] = dict(max_abs=int(d.max()), exact=float((d == 0).mean()))
        print(video, len(local), "windows", flush=True)

    ref_clip = np.load(SRC / "features.npz")["clip"][pos]
    ref_sig = np.load(ROOT / "review/generated/autoresearch/siglip-frames/siglip.npz")["frames"][pos]
    c_clip = (res["sanity"][0] * ref_clip).sum(1) / np.linalg.norm(ref_clip, axis=1)
    c_sig_frames = (res["sanity"][1] * ref_sig).sum(-1)
    sm, rm = res["sanity"][1].mean(1), ref_sig.mean(1)
    c_sig_mean = (sm * rm).sum(1) / np.linalg.norm(sm, axis=1) / np.linalg.norm(rm, axis=1)
    sanity_out = dict(windows=[dict(row=p, index=w["index"], video=w["video"], start=w["start"]) for p, w in zip(pos, sanity)],
                      clip_cos=c_clip.tolist(), siglip_frame_cos_min=c_sig_frames.min(1).tolist(),
                      siglip_window_mean_cos=c_sig_mean.tolist(), pixels_224_vs_prepared=pixel_check)
    print(json.dumps(sanity_out, indent=1), flush=True)
    assert c_clip.min() >= 0.9999 and c_sig_frames.min() >= 0.9999, "sanity failed"
    out.mkdir(parents=True)
    np.savez_compressed(out / "confirm_features.npz", clip=res["fresh"][0], **{"siglip:frames": res["fresh"][1]})
    meta = dict(sanity=sanity_out, n=len(fresh), confirm_set_sha256=digest(confirm), prepared_sha256=digest(SRC / "prepared.json"),
                clip_weights_sha256=expected["clip"], siglip_weights_sha256=WEIGHTS_SHA256, gpu=torch.cuda.get_device_name(),
                torch=torch.__version__, script_sha256=digest(__file__), npz_sha256=digest(out / "confirm_features.npz"),
                seconds=round(time.monotonic() - started, 1), test_touched=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("done", meta["seconds"], "s")


if __name__ == "__main__":
    main()
