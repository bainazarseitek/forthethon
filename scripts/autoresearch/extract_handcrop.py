"""Hand-region CLIP ViT-L/14 embeddings for the 160 train/val windows (GPU host).

Hypothesis (pre-registered): CLIP embeddings of detected HAND regions, window-meaned and appended to the
full-frame CLIP mean, improve macro F1 (esp. hand_count). All constants below were fixed before any run.

Frames    exactly the 16 source frames of each window: prepared.json window["source_frames"] (decode-order
          indices; ffprobe re-run and checked against window["source_pts"]), decoded by ffmpeg
          select=eq(n\\,i) at SOURCE resolution 464x848, rgb24. Only the 28 train/val videos are opened.
Detector  GroundingDINO-base (IDEA-Research/grounding-dino-base @ 12bdfa31, sha256-checked), official
          processor resize/normalisation, float32, caption "hands. banknotes.".
          Box score = max sigmoid over all text tokens; keep score > 0.3 (box threshold) - exactly
          transformers' post_process_grounded_object_detection, cross-checked on every frame.
          A kept box is a hand detection if max sigmoid over the "hands" token(s) > 0.25 (text threshold)
          (banknote detections are recorded, not used for the crop).
Crop      hand detections ranked by "hands" score, NMS IoU 0.5, union of the top 2 boxes, each side padded
          by 20% of the union's width/height, clipped to the frame, integer px (floor/ceil).
          No hand detection in a frame -> that frame's crop is the full frame.
Encoder   CLIP ViT-L/14 (pinned, sha256-checked) with the action-pilot view: crop taken from the source-
          resolution RGB frame, aspect-preserving bicubic resize to longer side 224 (shorter side rounded
          as ffmpeg force_original_aspect_ratio=decrease), centred pad to 224x224 with 0x7b7468, /255,
          native CLIP mean/std, visual_projection(pooler_output), L2 per frame.
          Fidelity check: the same path on the full frame vs the saved per-frame CLIP (clip-frames/).
Outputs   review/generated/autoresearch/handcrop/ (new dir, never overwritten):
          handcrop.npz      frames (160,16,768) float32 crop embeddings, prepared.json order
          boxes.npz         crop_box (160,16,4) int32 xyxy source px, fallback (160,16) bool,
                            n_hand / n_banknote (160,16) int16, full (160,16,768) full-frame embeddings
                            through this same path (diagnostic)
          detections.json   every kept detection per unique source frame [x0,y0,x1,y1,score,hand,banknote]
          contact_sheet.jpg a few TRAIN windows: frame with crop box + the 224 crop
          meta.json         detection rates, fidelity checks, timings, hashes
Test windows/videos are never loaded.  --smoke: first video only, prints stats, writes nothing.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
import torch.nn.functional as F
import transformers
from torchvision.ops import nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, CLIPModel

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "review/generated/action-5090"
OUT = ROOT / "review/generated/autoresearch/handcrop"
FRAMES_REF = ROOT / "review/generated/autoresearch/clip-frames"
TEST = {"X-4", "X-5", "X-26", "X-27", "X-28", "X-29", "X-34", "X-35", "X-37", "X-47", "X-48"}
CAPTION = "hands. banknotes."
BOX_T, TEXT_T, NMS_IOU, TOP_K, PAD_FRAC = 0.3, 0.25, 0.5, 2, 0.2
W, H, SIZE, PAD_RGB = 464, 848, 224, (0x7B, 0x74, 0x68)
DET_BATCH, CLIP_BATCH, SEED = 4, 32, 20261017
SHEET_T = (0, 5, 10, 15)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def snapshot(repo, revision):
    return Path.home() / ".cache/huggingface/hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision


def ffprobe(source, *args):
    return json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", *args, "-of", "json", str(source)]))


def decode(source, indexes):
    expression = "+".join(f"eq(n\\,{i})" for i in indexes)
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(source), "-vf", f"select='{expression}',format=rgb24",
                                   "-fps_mode", "vfr", "-f", "rawvideo", "pipe:1"])
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    assert len(frames) == len(indexes)
    return frames


def square(img):
    """Action-pilot view: aspect-preserving bicubic resize to longer side 224, centred 0x7b7468 pad."""
    h, w = img.shape[:2]
    nw, nh = (SIZE, max(1, math.floor(SIZE * h / w + .5))) if w >= h else (max(1, math.floor(SIZE * w / h + .5)), SIZE)
    canvas = Image.new("RGB", (SIZE, SIZE), PAD_RGB)
    canvas.paste(Image.fromarray(np.ascontiguousarray(img)).resize((nw, nh), Image.BICUBIC), ((SIZE - nw) // 2, (SIZE - nh) // 2))
    return np.asarray(canvas)


def crop_box(boxes, hand):
    """Padded union of the top-2 hand boxes after NMS; full frame when there is none."""
    if not len(boxes):
        return [0, 0, W, H], True
    b = boxes[nms(boxes, hand, NMS_IOU)[:TOP_K]]  # nms indices are sorted by decreasing hand score
    x0, y0, x1, y1 = b[:, 0].min().item(), b[:, 1].min().item(), b[:, 2].max().item(), b[:, 3].max().item()
    pw, ph = PAD_FRAC * (x1 - x0), PAD_FRAC * (y1 - y0)
    x0, y0 = min(W - 1, max(0, math.floor(x0 - pw))), min(H - 1, max(0, math.floor(y0 - ph)))
    x1, y1 = min(W, max(x0 + 1, math.ceil(x1 + pw))), min(H, max(y0 + 1, math.ceil(y1 + ph)))
    return [x0, y0, x1, y1], False


def contact_sheet(rows, path):
    tw, gap, title = round(SIZE * W / H), 6, 18
    cell = tw + SIZE + gap
    sheet = Image.new("RGB", (len(SHEET_T) * cell, len(rows) * (SIZE + title + gap)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, (text, items) in enumerate(rows):
        y = r * (SIZE + title + gap)
        draw.text((4, y + 3), text, fill="black")
        for c, (frame, box, crop) in enumerate(items):
            full = Image.fromarray(frame.copy())
            ImageDraw.Draw(full).rectangle(box, outline=(255, 40, 40), width=4)
            sheet.paste(full.resize((tw, SIZE), Image.BICUBIC), (c * cell, y + title))
            sheet.paste(Image.fromarray(crop), (c * cell + tw, y + title))
    sheet.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    smoke = ap.parse_args().smoke
    assert smoke or not OUT.exists(), "never overwrite an output dir"
    started = time.monotonic()
    data = json.loads((SRC / "prepared.json").read_text())
    windows = data["windows"]
    assert len(windows) == 160 and {w["split"] for w in windows} == {"train", "validation"}
    videos = sorted({w["video"] for w in windows}, key=lambda v: int(v[2:]))
    assert len(videos) == 28 and not set(videos) & TEST
    clips = {c["id"]: c for c in json.loads((ROOT / "review/generated/manifest.json").read_text())["clips"]}
    models = {m["name"]: m for m in json.loads((ROOT / "data/gpu_models.json").read_text())}
    ref_meta = json.loads((FRAMES_REF / "meta.json").read_text())
    assert digest(FRAMES_REF / "clipframes.npz") == ref_meta["npz_sha256"]
    assert ref_meta["prepared_sha256"] == digest(SRC / "prepared.json")
    gpath = snapshot(models["grounding_dino"]["repo"], models["grounding_dino"]["revision"])
    cpath = snapshot(data["config"]["models"]["clip"]["id"], data["config"]["models"]["clip"]["revision"])
    assert digest(gpath / "model.safetensors") == models["grounding_dino"]["sha256"]
    assert digest(cpath / "model.safetensors") == models["clip"]["sha256"]
    torch.manual_seed(SEED)
    torch.set_num_threads(8)  # 255 default CPU threads oversubscribe the shared host (preprocessing 40x slower)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gproc = AutoProcessor.from_pretrained(gpath, local_files_only=True)
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(gpath, local_files_only=True, use_safetensors=True)
    gdino = gdino.eval().requires_grad_(False).to("cuda")
    clip = CLIPModel.from_pretrained(cpath, local_files_only=True, use_safetensors=True).eval().requires_grad_(False).to("cuda")
    cp = json.loads((cpath / "preprocessor_config.json").read_text())
    mean = torch.tensor(cp["image_mean"], device="cuda")[None, :, None, None]
    std = torch.tensor(cp["image_std"], device="cuda")[None, :, None, None]
    tok = gproc.tokenizer(CAPTION, return_offsets_mapping=True)

    def span(phrase):
        a, b = CAPTION.index(phrase), CAPTION.index(phrase) + len(phrase)
        idx = [i for i, (s, e) in enumerate(tok["offset_mapping"]) if e > s and s < b and e > a]
        assert idx
        return idx

    hand_pos, bank_pos = span("hands"), span("banknotes")
    tokens = gproc.tokenizer.convert_ids_to_tokens(tok["input_ids"])
    print("tokens", tokens, "hands", hand_pos, "banknotes", bank_pos, flush=True)

    def encode(images):
        out = []
        for b in range(0, len(images), CLIP_BATCH):
            px = torch.from_numpy(np.stack(images[b:b + CLIP_BATCH])).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255
            with torch.inference_mode():
                e = clip.visual_projection(clip.vision_model(pixel_values=(px - mean) / std).pooler_output)
            out.append(F.normalize(e, dim=-1).float().cpu().numpy())
        return np.concatenate(out)

    def detect(frames):
        rows = []
        for b in range(0, len(frames), DET_BATCH):
            batch = [Image.fromarray(f) for f in frames[b:b + DET_BATCH]]
            inputs = gproc(images=batch, text=[CAPTION] * len(batch), return_tensors="pt").to("cuda")
            assert all(ids[:len(tok["input_ids"])].tolist() == tok["input_ids"] for ids in inputs["input_ids"])
            with torch.inference_mode():
                out = gdino(**inputs)
            probs = out.logits.sigmoid().float()
            score = probs.max(-1).values
            hand, bank = probs[..., hand_pos].max(-1).values, probs[..., bank_pos].max(-1).values
            cx, cy, bw, bh = out.pred_boxes.float().unbind(-1)
            boxes = torch.stack([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H], -1)
            ref = gproc.post_process_grounded_object_detection(out, inputs["input_ids"], threshold=BOX_T,
                                                               text_threshold=TEXT_T, target_sizes=[(H, W)] * len(batch))
            for i in range(len(batch)):
                keep = score[i] > BOX_T
                is_hand, is_bank = keep & (hand[i] > TEXT_T), keep & (bank[i] > TEXT_T)
                assert int(keep.sum()) == len(ref[i]["scores"])  # same kept boxes as the standard API
                if keep.any():  # (with no kept box transformers returns text_labels [""])
                    assert ["hand" in t for t in ref[i]["text_labels"]] == is_hand[keep].tolist()
                assert torch.allclose(boxes[i][keep], ref[i]["boxes"].float(), atol=1e-3)
                box, fallback = crop_box(boxes[i][is_hand], hand[i][is_hand])
                kept = torch.cat([boxes[i][keep], score[i][keep, None], hand[i][keep, None], bank[i][keep, None]], 1)
                rows.append(dict(crop_box=box, fallback=fallback, n_hand=int(is_hand.sum()), n_banknote=int(is_bank.sum()),
                                 kept=np.round(kept.cpu().numpy().astype(float), 4).tolist()))
        return rows

    sheet_rows = [r for r, w in enumerate(windows) if w["split"] == "train" and w["label"] == "hand_count"]
    sheet_rows += [next(r for r, w in enumerate(windows) if w["split"] == "train" and w["label"] == c)
                   for c in ("machine", "laptop", "document", "safe", "leave")]
    sheet_keys = {(windows[r]["video"], windows[r]["source_frames"][t]) for r in sheet_rows for t in SHEET_T}
    det, emb_crop, emb_full, sheet_imgs, mae, det_seconds = {}, {}, {}, {}, [], 0.0
    for video in videos:
        assert video not in TEST
        source = ROOT / clips[video]["source"]
        assert digest(source) == data["sources"][video]["sha256"], video
        info = ffprobe(source, "-show_entries", "stream=width,height:stream_side_data=rotation")["streams"][0]
        assert (info["width"], info["height"]) == (W, H) and not any(d.get("rotation") for d in info.get("side_data_list", []))
        pts = np.array([float(f["best_effort_timestamp_time"]) for f in ffprobe(
            source, "-show_frames", "-show_entries", "frame=best_effort_timestamp_time")["frames"]])
        assert np.all(np.diff(pts) > 0)
        local = [w for w in windows if w["video"] == video]
        for w in local:
            assert np.allclose(pts[w["source_frames"]], w["source_pts"], rtol=0, atol=1e-6)
        indexes = sorted({i for w in local for i in w["source_frames"]})
        frames = decode(source, indexes)
        torch.cuda.synchronize()
        tick = time.monotonic()
        rows = detect(frames)
        torch.cuda.synchronize()
        det_seconds += time.monotonic() - tick
        full = [square(f) for f in frames]
        crops = [square(f[r["crop_box"][1]:r["crop_box"][3], r["crop_box"][0]:r["crop_box"][2]]) for f, r in zip(frames, rows)]
        ef, ec = encode(full), encode(crops)
        item = data["inputs"][video]
        assert digest(SRC / "v1-clip" / item["file"]) == item["sha256"]
        prepared = np.load(SRC / "v1-clip" / item["file"])["frames"]
        pos = {i: k for k, i in enumerate(indexes)}
        for w in local:
            for si, pi in zip(w["source_frames"], w["frames"]):
                mae.append(float(np.abs(full[pos[si]].astype(np.int16) - prepared[pi]).mean()))
        for k, i in enumerate(indexes):
            det[(video, i)] = dict(rows[k], pts=float(pts[i]))
            emb_full[(video, i)], emb_crop[(video, i)] = ef[k], ec[k]
            if (video, i) in sheet_keys:
                sheet_imgs[(video, i)] = (frames[k].copy(), rows[k]["crop_box"], crops[k])
        hit = np.mean([not r["fallback"] for r in rows])
        print(f"{video}: {len(local)} windows, {len(indexes)} frames, hand-detected {hit:.2f}, "
              f"pixel MAE vs prepared {np.mean(mae[-16 * len(local):]):.2f}, {time.monotonic() - started:.0f}s", flush=True)
        if smoke:
            ref = np.load(FRAMES_REF / "clipframes.npz")["frames"]
            rws = [r for r, w in enumerate(windows) if w["video"] == video]
            cos = [float(emb_full[(video, i)] @ ref[r, t]) for r in rws for t, i in enumerate(windows[r]["source_frames"])]
            print("smoke: full-frame cos vs saved", min(cos), np.mean(cos), "max GPU MiB",
                  torch.cuda.max_memory_allocated() / 2**20, "det s/frame", det_seconds / len(indexes), flush=True)
            for k in indexes[:16]:
                print(k, det[(video, k)]["crop_box"], det[(video, k)]["n_hand"], det[(video, k)]["n_banknote"],
                      [d[4:] for d in det[(video, k)]["kept"]][:6], flush=True)
            return

    order = [(w["video"], i) for w in windows for i in w["source_frames"]]
    crop_arr = np.stack([emb_crop[k] for k in order]).reshape(160, 16, 768).astype(np.float32)
    full_arr = np.stack([emb_full[k] for k in order]).reshape(160, 16, 768).astype(np.float32)
    box_arr = np.array([det[k]["crop_box"] for k in order], np.int32).reshape(160, 16, 4)
    fallback = np.array([det[k]["fallback"] for k in order]).reshape(160, 16)
    n_hand = np.array([det[k]["n_hand"] for k in order], np.int16).reshape(160, 16)
    n_bank = np.array([det[k]["n_banknote"] for k in order], np.int16).reshape(160, 16)
    assert np.isfinite(crop_arr).all() and np.isfinite(full_arr).all()
    ref = np.load(FRAMES_REF / "clipframes.npz")["frames"]
    cos = (full_arr * ref).sum(-1)
    labels = np.array([w["label"] for w in windows])
    split = np.array([w["split"] for w in windows])
    area = (box_arr[..., 2] - box_arr[..., 0]) * (box_arr[..., 3] - box_arr[..., 1]) / (W * H)
    hit = ~fallback
    rates = dict(
        window_frames=float(hit.mean()), unique_frames=float(np.mean([not d["fallback"] for d in det.values()])),
        windows_with_any_hand_frame=float(hit.any(1).mean()), windows_with_all_frames_hand=float(hit.all(1).mean()),
        by_split={s: float(hit[split == s].mean()) for s in ("train", "validation")},
        by_label={c: float(hit[labels == c].mean()) for c in dict.fromkeys(labels)},
        banknote_frames=float((n_bank > 0).mean()),
        crop_area_fraction_when_detected=dict(median=float(np.median(area[hit])), mean=float(area[hit].mean())) if hit.any() else None)
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT / "handcrop.npz", frames=crop_arr)
    np.savez_compressed(OUT / "boxes.npz", crop_box=box_arr, fallback=fallback, n_hand=n_hand, n_banknote=n_bank, full=full_arr)
    (OUT / "detections.json").write_text(json.dumps(
        [dict(video=v, source_frame=i, **det[(v, i)]) for v, i in sorted(det, key=lambda k: (int(k[0][2:]), k[1]))]))
    contact_sheet([(f"row {r} {windows[r]['video']} {windows[r]['start']:.3f}-{windows[r]['end']:.3f}s TRAIN {windows[r]['label']} | "
                    f"hand frames {int(hit[r].sum())}/16 | frames t={','.join(map(str, SHEET_T))}: source+box | 224 crop",
                    [sheet_imgs[(windows[r]["video"], windows[r]["source_frames"][t])] for t in SHEET_T]) for r in sheet_rows],
                  OUT / "contact_sheet.jpg")
    meta = dict(
        caption=CAPTION, box_threshold=BOX_T, text_threshold=TEXT_T, nms_iou=NMS_IOU, top_k=TOP_K, pad_fraction=PAD_FRAC,
        tokens=tokens, hands_token_positions=hand_pos, banknotes_token_positions=bank_pos, detection_rate=rates,
        fidelity=dict(full_frame_cos_vs_saved_clipframes=dict(min=float(cos.min()), mean=float(cos.mean())),
                      pixel_mae_vs_prepared_224=dict(mean=float(np.mean(mae)), max=float(np.max(mae)))),
        shape=list(crop_arr.shape), unique_frames=len(det), sheet_rows=sheet_rows,
        grounding_dino=dict(repo=models["grounding_dino"]["repo"], revision=models["grounding_dino"]["revision"],
                            sha256=models["grounding_dino"]["sha256"], processor=type(gproc.image_processor).__name__),
        clip=dict(**data["config"]["models"]["clip"], sha256=models["clip"]["sha256"]),
        detector_seconds=det_seconds, total_seconds=time.monotonic() - started, max_gpu_mib=torch.cuda.max_memory_allocated() / 2**20,
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, transformers=transformers.__version__,
        prepared_sha256=digest(SRC / "prepared.json"), clipframes_sha256=ref_meta["npz_sha256"], script_sha256=digest(__file__),
        npz_sha256=digest(OUT / "handcrop.npz"), boxes_sha256=digest(OUT / "boxes.npz"),
        detections_sha256=digest(OUT / "detections.json"), test_evaluated=False)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("detection_rate", "fidelity", "total_seconds", "max_gpu_mib")}, indent=1))


if __name__ == "__main__":
    main()
