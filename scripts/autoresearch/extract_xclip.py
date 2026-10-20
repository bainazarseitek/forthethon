"""X-CLIP (video-text) window embeddings for the 160 train/val windows (GPU host). Pre-registered, one model.

Model, fixed before any result: microsoft/xclip-base-patch16-zero-shot @ 6746d6a5202bf24300cd9bf36f457f174483d41b
(model.safetensors sha256 0057077d...6adaf7, MIT licence; ViT-B/16, Kinetics-400 video-text contrastive training,
config num_frames=32). Chosen over xclip-base-patch32 (8 frames, ViT-B/32) for finer patches; patch32 not extracted.

Input: each window's 16 prepared full-frame 224x224 RGB frames (review/generated/action-pilot/inputs, sha256-checked;
windows in review/generated/action-5090/prepared.json order). The frame count is fixed by the config (cross-frame layers
reshape by num_frames, the MIT has num_frames position embeddings), so the 16 frames are mapped onto it:
32 -> every frame repeated twice in order, 8 -> every 2nd frame, 16 -> as is. Nothing is synthesised or borrowed from
neighbouring windows. Resize-shortest-edge-224 + centre-crop-224 are identities on 224x224 frames, so
pixels = (x/255 - image_mean)/image_std with the model's own preprocessor_config.json (ImageNet mean/std), float32,
TF32 off. Checked on sample windows against the official XCLIPProcessor and the full XCLIPModel forward.
Output: get_video_features(...).pooler_output (MIT-pooled projection, before L2) -> 'video' (160, 512) float32.
Test windows/videos are never loaded.

Usage (remote, project root):
  python3 scripts/autoresearch/extract_xclip.py --download         # online: pinned revision into the HF cache
  python3 scripts/autoresearch/extract_xclip.py PREPARED OUT_DIR   # offline; OUT_DIR must not exist
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "review/generated/action-pilot"
REPO = "microsoft/xclip-base-patch16-zero-shot"
REVISION = "6746d6a5202bf24300cd9bf36f457f174483d41b"
WEIGHTS_SHA256 = "0057077dd2bb6a5852fc9150f6e08f866448f4928f0acd7b267007f22a6adaf7"
FILES = ["config.json", "preprocessor_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "vocab.json", "merges.txt", "special_tokens_map.json"]
CHECK = (0, 80, 159)  # positions checked against the processor / full forward / a repeated forward


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download():
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(REPO, revision=REVISION, allow_patterns=FILES))
    assert digest(path / "model.safetensors") == WEIGHTS_SHA256
    print(json.dumps(dict(repo=REPO, revision=REVISION, snapshot=str(path), weights_sha256=WEIGHTS_SHA256)))


def frame_map(num_frames):
    idx = np.arange(16)
    return {8: idx[::2], 16: idx, 32: np.repeat(idx, 2)}[num_frames]


def main(prepared_path, out):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import huggingface_hub
    import torch
    import transformers
    from transformers import XCLIPModel, XCLIPProcessor

    out = Path(out)
    assert not out.exists(), "never overwrite an output dir"
    data = json.loads(Path(prepared_path).read_text())
    windows, splits = data["windows"], data["config"]["splits"]
    assert len(windows) == 160 and {w["split"] for w in windows} == {"train", "validation"}
    assert all(s["split"] != "test" for s in data["sources"].values())
    videos = {w["video"] for w in windows}
    assert not videos & set(splits["test"]) and videos <= set(splits["train"]) | set(splits["validation"])
    assert torch.cuda.is_available()
    torch.manual_seed(data["config"]["seed"]); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    path = Path(huggingface_hub.snapshot_download(REPO, revision=REVISION, allow_patterns=FILES, local_files_only=True))
    weights_sha256 = digest(path / "model.safetensors")
    assert weights_sha256 == WEIGHTS_SHA256
    pre = json.loads((path / "preprocessor_config.json").read_text())
    assert pre["size"] == 224 and pre["do_resize"] and pre["do_center_crop"] and pre["do_normalize"]
    tick = time.monotonic()
    model = XCLIPModel.from_pretrained(path, local_files_only=True, use_safetensors=True, dtype=torch.float32)
    model = model.eval().requires_grad_(False).to("cuda")
    load_seconds = time.monotonic() - tick
    assert all(p.dtype == torch.float32 for p in model.parameters())
    processor = XCLIPProcessor.from_pretrained(path, local_files_only=True)
    num_frames = model.config.vision_config.num_frames
    fmap = frame_map(num_frames)
    mean = torch.tensor(pre["image_mean"], device="cuda")[None, :, None, None]
    std = torch.tensor(pre["image_std"], device="cuda")[None, :, None, None]
    text = processor.tokenizer(["a video of a person"], return_tensors="pt").to("cuda")

    feats, timings, checks, loaded, current, frames = [], [], [], {}, None, None
    for i, w in enumerate(windows):
        if w["video"] != current:
            item = data["inputs"][w["video"]]
            assert digest(SOURCE / item["file"]) == item["sha256"]
            frames = np.load(SOURCE / item["file"])["frames"]
            current, loaded[w["video"]] = w["video"], item["sha256"]
        clip = frames[w["frames"]][fmap]
        assert clip.shape == (num_frames, 224, 224, 3) and clip.dtype == np.uint8
        pixels = torch.from_numpy(clip.copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255
        pixels = ((pixels - mean) / std)[None]
        torch.cuda.synchronize(); tick = time.monotonic()
        with torch.inference_mode():
            vector = model.get_video_features(pixel_values=pixels).pooler_output
        torch.cuda.synchronize(); timings.append(time.monotonic() - tick)
        assert vector.shape == (1, model.config.projection_dim)
        feats.append(vector.float().cpu().numpy()[0])
        if i in CHECK:
            official = processor(videos=list(clip), return_tensors="pt")["pixel_values"].to("cuda")
            assert official.shape == pixels.shape, (official.shape, pixels.shape)
            with torch.inference_mode():
                full = model(input_ids=text["input_ids"], attention_mask=text["attention_mask"],
                             pixel_values=pixels).video_embeds
                again = model.get_video_features(pixel_values=pixels).pooler_output
                from_official = model.get_video_features(pixel_values=official).pooler_output
            checks.append(dict(
                position=i, video=w["video"], split=w["split"],
                max_abs_pixels_vs_official_processor=float((official - pixels).abs().max()),
                cosine_vs_official_processor_features=float(torch.cosine_similarity(from_official, vector).item()),
                cosine_vs_full_forward_video_embeds=float(torch.cosine_similarity(full, vector).item()),
                max_abs_repeat_forward=float((again - vector).abs().max())))
    video = np.stack(feats).astype(np.float32)
    assert video.shape == (160, model.config.projection_dim) and np.isfinite(video).all()
    assert all(c["max_abs_pixels_vs_official_processor"] < 1e-5 and c["cosine_vs_full_forward_video_embeds"] > 0.99999
               and c["cosine_vs_official_processor_features"] > 0.99999 and c["max_abs_repeat_forward"] < 1e-5
               for c in checks), checks
    out.mkdir(parents=True)
    np.savez_compressed(out / "xclip.npz", video=video)
    meta = dict(
        repo=REPO, revision=REVISION, weights_file="model.safetensors", weights_sha256=weights_sha256, licence="mit",
        num_frames=num_frames, frame_mapping="16 window frames -> each repeated twice in order (np.repeat)"
        if num_frames == 32 else f"frame_map({num_frames})", frame_indices=fmap.tolist(),
        patch_size=model.config.vision_config.patch_size, projection_dim=model.config.projection_dim,
        parameters=sum(p.numel() for p in model.parameters()), image_mean=pre["image_mean"], image_std=pre["image_std"],
        preprocessing="x/255, (x-mean)/std; resize/centre-crop are identities at 224", dtype="float32", tf32=False,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        output="get_video_features(pixel_values).pooler_output (MIT mean-pooled, before L2), key 'video'",
        shape=list(video.shape), norm_min=float(np.linalg.norm(video, axis=1).min()),
        norm_max=float(np.linalg.norm(video, axis=1).max()), checks=checks,
        model_load_seconds=load_seconds, forward_seconds=float(sum(timings)),
        median_window_seconds=float(np.median(timings[1:])), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        transformers=transformers.__version__, huggingface_hub=huggingface_hub.__version__, numpy=np.__version__,
        prepared_sha256=digest(prepared_path), inputs_sha256=dict(sorted(loaded.items())),
        script_sha256=digest(__file__), npz_sha256=digest(out / "xclip.npz"), test_evaluated=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    download() if sys.argv[1:] == ["--download"] else main(*sys.argv[1:3])
