"""Causal CLIP frame-pooling + a fixed table ROI + evidence memory.

prepare: decode on the laptop; infer: offline on laptop or own GPU server;
analyze: local plots and comparison. Inference inputs contain no annotation times.
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "review/generated/protocol-pilot"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def prepare(out):
    started = time.monotonic()
    config = json.loads((ROOT / "data/protocol_pilot.json").read_text())
    manifest = json.loads((ROOT / "review/generated/manifest.json").read_text())
    clips = {c["id"]: c for c in manifest["clips"]}
    frames, decisions, sources = [], [], {}
    (out / "frames").mkdir(exist_ok=True)
    for video in config["videos"]:
        c = clips[video]
        assert (c["width"], c["height"]) == (464, 848), "ROIs must be reviewed for a different camera"
        source = ROOT / c["source"]
        probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(source)])
        pts = np.array([float(f["best_effort_timestamp_time"]) for f in json.loads(probe)["frames"]])
        assert np.all(np.diff(pts) > 0)
        targets = np.unique(np.r_[np.arange(0, pts[-1], 1/config["sample_hz"]), pts[-1]])
        # Always floor to the latest available PTS; nearest could look into the future.
        indexes = np.searchsorted(pts, targets + 1e-7, side="right") - 1
        assert np.all(pts[indexes] <= targets + 1e-7)
        assert np.max(targets - pts[indexes]) < .07
        expression = "+".join(f"eq(n\\,{i})" for i in indexes)
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(source), "-vf",
            f"select='{expression}',format=rgb24", "-fps_mode", "vfr", "-f", "rawvideo", "pipe:1"])
        images = np.frombuffer(raw, np.uint8).reshape(-1, 848, 464, 3)
        assert len(images) == len(indexes)
        video_frames = []
        for i, source_index, target, arr in zip(range(len(indexes)), indexes, targets, images):
            item = dict(index=len(frames), video=video, target=float(target), pts=float(pts[source_index]),
                        source_frame=int(source_index), file=f"frames/{video}-{i:03d}.png")
            Image.fromarray(arr).save(out / item["file"])
            frames.append(item)
            video_frames.append(item)
        for t in np.r_[np.arange(1, pts[-1], config["decision_seconds"]), pts[-1]]:
            available = [f for f in video_frames if f["target"] <= t + 1e-7]
            window = [f for f in available if f["target"] > t - config["window_seconds"]]
            assert window and all(f["pts"] <= t + 1e-7 for f in window)
            decisions.append(dict(video=video, t=float(t), frame=available[-1]["index"],
                                  window=[f["index"] for f in window]))
        sources[video] = dict(sha256=digest(source), duration=c["duration"],
                              preview=c["video"], width=464, height=848)
        preview = Image.fromarray(images[len(images)//2]).copy()
        draw = ImageDraw.Draw(preview)
        draw.rectangle(config["scene_box"], outline="#2b79cb", width=3)
        draw.rectangle(config["table_box"], outline="#ee7941", width=3)
        preview.save(out / f"{video}-regions.jpg", quality=94)
    save_json(out / "prepared.json", dict(config=config, frames=frames, decisions=decisions, sources=sources,
        preparation_seconds=round(time.monotonic()-started, 3), config_sha256=digest(ROOT / "data/protocol_pilot.json")))
    print(f"Prepared {len(frames)} frames, {len(decisions)} causal decisions; no annotations in inference bundle", flush=True)


def infer(out):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import CLIPModel, CLIPProcessor
    started = time.monotonic()
    data = json.loads((out / "prepared.json").read_text())
    config = data["config"]
    checkpoint = Path.home() / ".cache/huggingface/hub" / ("models--" + config["model"].replace("/", "--")) / "snapshots" / config["revision"]
    assert (checkpoint / "model.safetensors").is_file(), f"Missing cached safe weights: {checkpoint}"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = CLIPModel.from_pretrained(checkpoint, local_files_only=True, use_safetensors=True).eval().to(device)
    processor = CLIPProcessor.from_pretrained(checkpoint, local_files_only=True)
    load_seconds = time.monotonic() - started
    texts = [p for group in config["prompts"].values() for prompts in group.values() for p in prompts]
    def sync():
        if device == "cuda": torch.cuda.synchronize()
        elif device == "mps": torch.mps.synchronize()
    sync()
    tick = time.monotonic()
    with torch.inference_mode():
        tokens = processor.tokenizer(texts, padding=True, return_tensors="pt").to(device)
        text_features = model.text_projection(model.text_model(**tokens).pooler_output)
        text_features = normalize(text_features.float().cpu().numpy())
    sync()
    text_seconds = time.monotonic() - tick
    vectors = []
    forward_seconds, preprocessing_seconds, latencies = 0., 0., []
    # One update at a time: timing includes both the scene and table view, no future batching.
    for item in data["frames"]:
        tick = time.monotonic()
        with Image.open(out / item["file"]) as image:
            crops = [image.crop(config[key]) for key in ("scene_box", "table_box")]
            # Pad non-square patches so CLIP's center crop cannot silently remove their edges.
            crops = [ImageOps.pad(c, (224, 224), method=Image.Resampling.BICUBIC, color=(123, 117, 104)) for c in crops]
            if item["index"] in {0, 20, len(data["frames"])-1}:
                for name, crop in zip(("scene", "table"), crops):
                    crop.save(out / f"input-{item['index']:03d}-{name}.png")
            pixels = processor.image_processor(crops, return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"].to(device)
        sync()
        preprocessing_seconds += time.monotonic() - tick
        tick = time.monotonic()
        with torch.inference_mode():
            emb = model.visual_projection(model.vision_model(pixel_values=pixels).pooler_output)
            emb = normalize(emb.float().cpu().numpy())
        sync()
        elapsed = time.monotonic() - tick
        forward_seconds += elapsed
        latencies.append(elapsed)
        vectors.append(emb)
        if (item["index"] + 1) % 10 == 0:
            print(f"{item['index']+1}/{len(data['frames'])} scene + table frames on {device}", flush=True)
    vectors = np.array(vectors)
    assert vectors.shape == (len(data["frames"]), 2, text_features.shape[1]) and np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=-1), 1, atol=1e-5)
    np.savez_compressed(out / "features.npz", images=vectors, text=text_features)
    save_json(out / "inference.json", dict(model=config["model"], revision=config["revision"], device=device,
        gpu=torch.cuda.get_device_name() if device == "cuda" else device, torch=torch.__version__,
        transformers=transformers.__version__, dtype="float32", dimension=text_features.shape[1],
        weights_sha256=digest(checkpoint / "model.safetensors"), script_sha256=digest(__file__),
        prepared_sha256=digest(out / "prepared.json"), texts=texts, model_load_seconds=load_seconds,
        text_seconds=text_seconds, image_forward_seconds=forward_seconds,
        image_preprocessing_seconds=preprocessing_seconds, per_frame_forward_seconds=latencies,
        total_seconds=time.monotonic()-started, learned_logit_scale=float(model.logit_scale.exp().item()),
        preprocessing="Original PTS -> fixed scene/table crop -> aspect-preserving bicubic pad to 224 -> CLIP normalization; no center crop. Mean L2 frame features over causal 2 s only for action prompts."))
    print(f"Saved {vectors.shape} image features and {text_features.shape} text features", flush=True)


def softmax(x, temperature):
    x = (np.asarray(x) - np.max(x)) / temperature
    return np.exp(x) / np.exp(x).sum()


def winner(scores, margin):
    keys = list(scores)
    order = sorted(keys, key=scores.get, reverse=True)
    return order[0] if scores[order[0]] - scores[order[1]] >= margin else "unknown"


def apply_memory(observations, config):
    """Pure causal function; receives scores only, never video IDs or annotation times."""
    held_cash, evidence_t = None, None
    previous_table, table_streak = "unknown", 0
    previous_action, action_streak = "unknown", 0
    seen = {}
    results = []
    for row in observations:
        table = winner(row["table"], config["evidence_margin"])
        table_streak = table_streak + 1 if table == previous_table else 1
        previous_table = table
        action = winner(row["actions"], config["evidence_margin"])
        action_streak = action_streak + 1 if action == previous_action else 1
        previous_action = action
        if action != "unknown" and action_streak >= config["confirm_steps"]:
            seen.setdefault(action, row["t"])
        if table_streak >= config["confirm_steps"]:
            if table == "cash":
                held_cash, evidence_t = row["cash_support"], row["t"]
            elif table == "clear":
                held_cash, evidence_t = None, None
        # shortcut: fixed six-second evidence lifetime, requires calibration on independent clips.
        fresh = held_cash is not None and row["t"] - evidence_t <= 6
        remembered = max(row["cash_support"], held_cash) if fresh else row["cash_support"]
        stage = ("cash_on_table" if fresh else "closing_seen" if "close" in seen else
                 "storage_seen" if "store" in seen else "counting_seen" if "count" in seen else "unconfirmed")
        expected = (["carry", "store", "leave"] if fresh else ["leave", "idle"] if "close" in seen else
                    ["close", "leave"] if "store" in seen else ["put_table", "carry", "store", "leave"] if "count" in seen else ["count", "carry", "leave"])
        results.append(dict(row, table_observation=table, action_observation=action, seen=dict(seen), stage=stage,
            expected=expected, memory_evidence_t=evidence_t if fresh else None, remembered_cash=remembered,
            memory_used=bool(fresh and remembered > row["cash_support"] + 1e-6),
            without_memory=min(row["cash_support"], row["away_support"]),
            with_memory=min(remembered, row["away_support"])))
    return results


def check(out):
    cfg = dict(evidence_margin=.02, confirm_steps=2)
    def obs(t, cash, clear, occluded, away=.9):
        return dict(t=t, table=dict(cash=cash, clear=clear, occluded=occluded, paper=0),
            actions=dict(count=.3, leave=.1), cash_support=.9 if cash > clear and cash > occluded else .1,
            away_support=away)
    rows = [obs(1,.3,.1,0), obs(2,.3,.1,0), obs(3,.1,0,.3), obs(4,.1,.3,0), obs(5,.1,.3,0), obs(12,.1,0,.3)]
    full = apply_memory(rows, cfg)
    assert full[2]["with_memory"] == .9 and full[2]["without_memory"] == .1
    assert full[4]["with_memory"] == .1, "Two clear observations must release remembered cash"
    assert full[5]["memory_evidence_t"] is None
    assert apply_memory(rows[:3], cfg) == full[:3], "No future leakage"
    expired = apply_memory(rows[:2] + [obs(10,.1,0,.3)], cfg)
    assert expired[-1]["memory_evidence_t"] is None, "Stale evidence must expire"
    seated = apply_memory([obs(1,.3,0,0,.1), obs(2,.3,0,0,.1)], cfg)
    assert seated[-1]["with_memory"] == .1, "Cash with an attending cashier is not unattended"
    assert apply_memory([obs(1,.1,0,.3)], cfg)[0]["stage"] == "unconfirmed"
    assert winner(dict(a=.2,b=.19), .02) == "unknown"
    print("Passed: persistence, clearing, evidence expiry, causality, attendance, unknown", flush=True)


def analyze(out):
    check(out)
    data = json.loads((out / "prepared.json").read_text())
    run = json.loads((out / "inference.json").read_text())
    assert run["prepared_sha256"] == digest(out / "prepared.json"), "Inference inputs changed"
    config = data["config"]
    features = np.load(out / "features.npz")
    image_features, text_features = features["images"], features["text"]
    groups, offset = {}, 0
    for name, prompts in config["prompts"].items():
        groups[name] = {}
        for key, texts in prompts.items():
            groups[name][key] = normalize(text_features[offset:offset+len(texts)].mean(0))
            offset += len(texts)
    assert offset == len(text_features)
    def score(vector, group):
        return {key: float(vector @ embedding) for key, embedding in groups[group].items()}
    rows = []
    for d in data["decisions"]:
        frame = data["frames"][d["frame"]]
        assert frame["pts"] <= d["t"] + 1e-7
        assert max(data["frames"][i]["pts"] for i in d["window"]) <= d["t"] + 1e-7
        action_vector = normalize(image_features[d["window"], 0].mean(0))
        row = dict(d, pts=frame["pts"], image=frame["file"],
            actions=score(action_vector, "actions"), table=score(image_features[d["frame"], 1], "table"),
            table_scene=score(image_features[d["frame"], 0], "table"),
            presence=score(image_features[d["frame"], 0], "presence"))
        for group, target, output in [("table", "cash", "cash_support"), ("presence", "away", "away_support")]:
            keys = list(row[group])
            row[output] = float(softmax(list(row[group].values()), config["temperature"])[keys.index(target)])
        rows.append(row)
    predictions = []
    for video in config["videos"]:
        source_rows = [r for r in rows if r["video"] == video]
        result = apply_memory(source_rows, config)
        # Recompute every prefix: future frames must not change an earlier decision.
        for n in range(1, len(source_rows)+1):
            assert apply_memory(source_rows[:n], config) == result[:n]
        predictions.extend(result)
    # Annotation access starts only after all predictions have been computed.
    manifest = json.loads((ROOT / "review/generated/manifest.json").read_text())
    clips = {c["id"]: c for c in manifest["clips"]}
    labels = {}
    for row in predictions:
        c = clips[row["video"]]
        relevant = [r for r in c["annotation"]["rows"] if r["start"] <= row["t"] <= r["end"]]
        label = "normal" if c["label"] == "clean" else "violation" if any(r["kind"] == "viol" for r in relevant) else "normal" if any(r["kind"] == "ok" for r in relevant) else "unknown"
        row["label"] = label
        # Object/attendance audit only in explicitly described, unambiguous intervals.
        row["table_truth"] = ("cash" if row["video"] == "X-26" and row["t"] >= 12.5 else
                              "clear" if row["video"] == "X-27" and row["t"] >= 21 else "unknown")
        row["presence_truth"] = ("attending" if row["t"] <= 7.5 else
                                 "away" if (row["video"] == "X-26" and row["t"] >= 9) or (row["video"] == "X-27" and row["t"] >= 18) else "unknown")
        row["manual_note"] = "; ".join(r["seen"] for r in relevant) or "Между размеченными интервалами"
        labels[row["video"]] = c["annotation"]["rows"]
    from sklearn.metrics import roc_auc_score
    valid = [r for r in predictions if r["label"] != "unknown"]
    truth = np.array([r["label"] == "violation" for r in valid])
    metrics = {}
    for method in ["without_memory", "with_memory"]:
        values = np.array([r[method] for r in valid])
        alarms = values >= config["alarm_threshold"]
        metrics[method] = dict(auc=float(roc_auc_score(truth, values)),
            true_positive=int((alarms & truth).sum()), false_negative=int((~alarms & truth).sum()),
            false_positive=int((alarms & ~truth).sum()), true_negative=int((~alarms & ~truth).sum()),
            first_alarm={v: next((r["t"] for r in predictions if r["video"] == v and r[method] >= config["alarm_threshold"]), None) for v in config["videos"]})
    table_rows = [r for r in predictions if r["table_truth"] != "unknown"]
    table_y = [r["table_truth"] == "cash" for r in table_rows]
    table_metrics = {}
    for group in ["table_scene", "table"]:
        margin = [r[group]["cash"] - max(v for k,v in r[group].items() if k != "cash") for r in table_rows]
        table_metrics[group] = dict(auc=float(roc_auc_score(table_y, margin)),
            correct_top1=sum(max(r[group], key=r[group].get) == r["table_truth"] for r in table_rows), total=len(table_rows))
    presence_rows = [r for r in predictions if r["presence_truth"] != "unknown"]
    object_metrics = dict(table=table_metrics, presence=dict(
        correct_top1=sum(max(r["presence"], key=r["presence"].get) == r["presence_truth"] for r in presence_rows), total=len(presence_rows)))
    result = dict(config=config, run=run, sources=data["sources"], rows=predictions, metrics=metrics,
        object_metrics=object_metrics, annotations=labels, analysis_script_sha256=digest(__file__),
        annotations_sha256=digest(ROOT / "docs/annotations.md"),
        memory_changed_decisions=sum(abs(r["with_memory"]-r["without_memory"]) > 1e-6 for r in predictions),
        memory_changed_alarms=sum((r["with_memory"] >= config["alarm_threshold"]) != (r["without_memory"] >= config["alarm_threshold"]) for r in predictions))
    save_json(out / "results.json", result)
    (out / "results.js").write_text("window.PROTOCOL_PILOT=" + json.dumps(result, ensure_ascii=False) + ";\n")
    import csv
    with (out / "scores.csv").open("w") as f:
        columns = ["video", "t", "pts", "label", "cash_support", "away_support", "without_memory", "with_memory", "stage", "table_observation", "action_observation"]
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predictions)
    plots(out, result)
    print(json.dumps(dict(metrics=metrics, objects=object_metrics,
        changed_scores=result["memory_changed_decisions"], changed_alarms=result["memory_changed_alarms"]), indent=2), flush=True)


def plots(out, result):
    os.environ.setdefault("MPLCONFIGDIR", str(out / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#2867a1", "#b54b45", "#998037", "#7e5ea5", "#288a75", "#bf6c30", "#788892"]
    fig, axes = plt.subplots(4, 2, figsize=(15, 13), sharex="col", layout="constrained")
    for col, video in enumerate(result["config"]["videos"]):
        rows = [r for r in result["rows"] if r["video"] == video]
        t = [r["t"] for r in rows]
        axes[0,col].set_title(video + (" / violation from 9 s (manual)" if video == "X-26" else " / clean (manual)"), fontsize=13)
        for group, ax in [("actions", axes[0,col]), ("table", axes[1,col])]:
            for color, key in zip(colors, rows[0][group]):
                ax.plot(t, [r[group][key] for r in rows], label=key, color=color, linewidth=1.5)
            ax.set_ylabel("Cosine / " + group)
            ax.legend(ncol=3 if group == "actions" else 2, fontsize=8, loc="upper left")
        axes[2,col].plot(t, [r["cash_support"] for r in rows], label="cash / current patch", color="#b54b45")
        axes[2,col].plot(t, [r["away_support"] for r in rows], label="away / current scene", color="#2867a1")
        axes[2,col].plot(t, [r["remembered_cash"] for r in rows], label="cash / memory", color="#288a75", linestyle="--")
        axes[2,col].set_ylabel("Relative support (not probability)")
        axes[2,col].legend(fontsize=8)
        axes[3,col].plot(t, [r["without_memory"] for r in rows], label="Without memory", color="#9b6978", linewidth=2)
        axes[3,col].plot(t, [r["with_memory"] for r in rows], label="With memory", color="#237b66", linestyle="--", linewidth=2)
        axes[3,col].axhline(result["config"]["alarm_threshold"], color="#555", linestyle=":", label="Fixed heuristic threshold")
        axes[3,col].set_ylabel("Unattended cash score")
        axes[3,col].legend(fontsize=8)
        axes[3,col].set_xlabel("Decision time, seconds (only past frames)")
        for ax in axes[:,col]:
            ax.grid(alpha=.18)
            if video == "X-26":
                ax.axvspan(9, result["sources"][video]["duration"], color="#b54b45", alpha=.065)
                ax.axvline(9, color="#b54b45", alpha=.4, linewidth=1)
        for ax in axes[2:,col]: ax.set_ylim(-.03,1.03)
    fig.suptitle("X-26 / X-27: frozen CLIP + table patch + causal memory\nSame visual observations for both decisions; annotation shown only for evaluation", fontsize=15)
    fig.savefig(out / "timelines.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "infer", "analyze", "check"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # A kernel lock is released even if a run fails; two runs cannot overwrite this experiment.
    with (args.out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
