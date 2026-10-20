"""Whole-video action split; frozen VideoMAE and CLIP; identical linear probes.

prepare locally, infer + fit on own GPU host, report locally. No automatic downloads.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from run_protocol_pilot import digest, normalize, save_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "review/generated/action-pilot"


def split_audit(config, manifest):
    clips = {c["id"]: c for c in manifest["clips"]}
    assignments = {}
    for split, videos in config["splits"].items():
        for video in videos:
            assert video in clips and video not in assignments, f"Repeated/unknown video: {video}"
            assignments[video] = split
    assert set(assignments) == set(clips), "Assign every source, including excluded/duplicate videos"
    for a, b, _ in manifest["pairs"]:
        assert assignments[a] == assignments[b], f"Paired-scenario leakage: {a}, {b}"
    hashes = {}
    for c in clips.values():
        if c["sha256"] in hashes:
            assert assignments[c["id"]] == assignments[hashes[c["sha256"]]], "Duplicate leakage"
        hashes[c["sha256"]] = c["id"]
        if c.get("duplicate_of"):
            assert c["id"] not in config["row_mapping"], "Do not sample duplicate videos"
    return clips, assignments


def make_windows(config, manifest):
    clips, assignments = split_audit(config, manifest)
    windows, exclusions, intervals = [], [], []
    class_ids = [c["id"] for c in config["classes"]]
    for video, c in clips.items():
        selected = {}
        for label, rows in config["row_mapping"].get(video, {}).items():
            assert label in class_ids
            for row in rows:
                assert row not in selected and 0 <= row < len(c["annotation"]["rows"])
                selected[row] = label
        for i, row in enumerate(c["annotation"]["rows"]):
            if i not in selected:
                exclusions.append(dict(video=video, row=i, reason="duplicate" if c.get("duplicate_of") else "not a selected unambiguous action"))
                continue
            low, high = row["start"] + config["boundary_margin"], row["end"] - config["boundary_margin"]
            starts = np.arange(low, high - config["window_seconds"] + 1e-7, config["stride_seconds"])
            if not len(starts):
                exclusions.append(dict(video=video, row=i, reason="too short for a 2 s window plus margins"))
                continue
            if len(starts) > config["max_windows_per_row"]:
                starts = starts[np.linspace(0, len(starts)-1, config["max_windows_per_row"]).round().astype(int)]
            interval = dict(video=video, row=i, label=selected[i], split=assignments[video], **row)
            intervals.append(interval)
            for start in starts:
                windows.append(dict(index=len(windows), video=video, row=i, split=assignments[video],
                    label=selected[i], start=float(start), end=float(start + config["window_seconds"]),
                    annotation_start=row["start"], annotation_end=row["end"], seen=row["seen"]))
    for split in config["splits"]:
        assert {w["label"] for w in windows if w["split"] == split} == set(class_ids), f"Missing class in {split}"
    for video in clips:
        spans = sorted((r["start"], r["end"]) for r in intervals if r["video"] == video)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "Conflicting action intervals"
    return windows, intervals, exclusions


def prepare(out):
    started = time.monotonic()
    config_path = ROOT / "data/action_pilot.json"
    config = json.loads(config_path.read_text())
    manifest_path = ROOT / "review/generated/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    clips, assignments = split_audit(config, manifest)
    windows, intervals, exclusions = make_windows(config, manifest)
    (out / "inputs").mkdir(exist_ok=True)
    (out / "thumbnails").mkdir(exist_ok=True)
    sources, inputs = {}, {}
    for video in sorted({w["video"] for w in windows}, key=lambda v: int(v[2:])):
        c = clips[video]
        source = ROOT / c["source"]
        assert digest(source) == c["sha256"], f"Source changed: {video}"
        probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(source)])
        pts = np.array([float(f["best_effort_timestamp_time"]) for f in json.loads(probe)["frames"]])
        assert np.all(np.diff(pts) > 0)
        local = [w for w in windows if w["video"] == video]
        for w in local:
            targets = w["start"] + np.arange(config["num_frames"]) * config["window_seconds"] / config["num_frames"]
            indexes = np.searchsorted(pts, targets + 1e-7, side="right") - 1
            assert indexes.min() >= 0 and len(set(indexes)) == config["num_frames"]
            assert np.all(pts[indexes] <= targets + 1e-7) and max(targets - pts[indexes]) < .07
            assert pts[indexes[0]] >= w["annotation_start"] and pts[indexes[-1]] < w["annotation_end"]
            w["source_frames"], w["source_pts"] = indexes.tolist(), pts[indexes].tolist()
        indexes = sorted({i for w in local for i in w["source_frames"]})
        expression = "+".join(f"eq(n\\,{i})" for i in indexes)
        size = config["image_size"]
        filters = (f"select='{expression}',scale={size}:{size}:force_original_aspect_ratio=decrease:flags=bicubic,"
                   f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=0x7b7468,format=rgb24")
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(source), "-vf", filters,
                                       "-fps_mode", "vfr", "-f", "rawvideo", "pipe:1"])
        frames = np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)
        assert len(frames) == len(indexes)
        mapping = {source_index: i for i, source_index in enumerate(indexes)}
        for w in local:
            w["frames"] = [mapping[i] for i in w["source_frames"]]
            w["thumbnail"] = f"thumbnails/{w['index']:04d}.jpg"
            strip = np.concatenate([frames[w["frames"][i]] for i in (0, 7, 15)], axis=1)
            Image.fromarray(strip).save(out / w["thumbnail"], quality=90)
        file = f"inputs/{video}.npz"
        np.savez_compressed(out / file, frames=frames)
        inputs[video] = dict(file=file, sha256=digest(out/file), frames=len(frames))
        sources[video] = dict(sha256=c["sha256"], preview=c["video"], split=assignments[video], duration=c["duration"])
        print(f"{video}: {len(local)} windows, {len(frames)} shared frames, {assignments[video]}", flush=True)
    data = dict(config=config, config_sha256=digest(config_path), manifest_sha256=digest(manifest_path),
        windows=windows, intervals=intervals, exclusions=exclusions, sources=sources, inputs=inputs,
        assignments=assignments, pairs=manifest["pairs"], duplicate_groups=manifest["summary"]["duplicates"],
        preparation_seconds=time.monotonic()-started)
    save_json(out / "prepared.json", data)
    print(json.dumps({s: dict(Counter(w["label"] for w in windows if w["split"] == s)) for s in config["splits"]}, indent=2))


def infer(out):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import CLIPModel, CLIPTokenizer, VideoMAEForPreTraining
    assert torch.cuda.is_available(), "This run is intended for the user's GPU host"
    torch.manual_seed(20261017)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = json.loads((out / "prepared.json").read_text())
    config = data["config"]
    features, metadata = {}, {}
    for name in ("videomae", "clip"):
        spec = config["models"][name]
        path = Path.home() / ".cache/huggingface/hub" / ("models--" + spec["id"].replace("/", "--")) / "snapshots" / spec["revision"]
        assert (path / "model.safetensors").is_file(), f"Missing safe weights: {path}"
        started = time.monotonic()
        if name == "videomae":
            pretrained = VideoMAEForPreTraining.from_pretrained(path, local_files_only=True, use_safetensors=True)
            model = pretrained.videomae
            del pretrained
        else:
            model = CLIPModel.from_pretrained(path, local_files_only=True, use_safetensors=True)
        model = model.eval().requires_grad_(False).to("cuda")
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        load_seconds = time.monotonic() - started
        processor = json.loads((path / "preprocessor_config.json").read_text())
        mean = torch.tensor(processor["image_mean"], device="cuda")[None, :, None, None]
        std = torch.tensor(processor["image_std"], device="cuda")[None, :, None, None]
        text_seconds = 0
        if name == "clip":
            texts = [text for cls in config["classes"] for text in cls["prompts"]]
            tokenizer = CLIPTokenizer.from_pretrained(path, local_files_only=True)
            tokens = tokenizer(texts, padding=True, return_tensors="pt").to("cuda")
            tick = time.monotonic()
            with torch.inference_mode():
                text = model.text_projection(model.text_model(**tokens).pooler_output).float().cpu().numpy()
            features["text"] = normalize(normalize(text).reshape(len(config["classes"]), 2, -1).mean(axis=1))
            text_seconds = time.monotonic()-tick
        vectors, timings = [], []
        current_video, frames = None, None
        for w in data["windows"]:
            if w["video"] != current_video:
                item = data["inputs"][w["video"]]
                assert digest(out / item["file"]) == item["sha256"]
                frames = np.load(out / item["file"])["frames"]
                current_video = w["video"]
            pixels = torch.from_numpy(frames[w["frames"]].copy()).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255
            pixels = (pixels - mean) / std
            torch.cuda.synchronize()
            tick = time.monotonic()
            with torch.inference_mode():
                if name == "videomae":
                    vector = model(pixel_values=pixels[None]).last_hidden_state.mean(dim=1)
                else:
                    image = model.visual_projection(model.vision_model(pixel_values=pixels).pooler_output)
                    vector = torch.nn.functional.normalize(image, dim=-1).mean(dim=0, keepdim=True)
                vector = vector.float().cpu().numpy()[0]
            torch.cuda.synchronize()
            timings.append(time.monotonic()-tick)
            vectors.append(vector)
            if len(vectors) % 40 == 0:
                print(f"{name}: {len(vectors)}/{len(data['windows'])} windows", flush=True)
        features[name] = normalize(np.array(vectors))
        assert features[name].shape == (len(data["windows"]), 768) and np.isfinite(features[name]).all()
        metadata[name] = dict(**spec, weights_sha256=digest(path/"model.safetensors"), parameters=sum(p.numel() for p in model.parameters()),
            trainable_encoder_parameters=0, frozen=True, dtype="float32", model_load_seconds=load_seconds,
            forward_seconds=sum(timings), median_window_seconds=float(np.median(timings[1:])),
            text_seconds=text_seconds, total_seconds=time.monotonic()-started,
            image_mean=processor["image_mean"], image_std=processor["image_std"])
        del model
        torch.cuda.empty_cache()
    np.savez_compressed(out/"features.npz", **features)
    save_json(out/"inference.json", dict(models=metadata, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        transformers=transformers.__version__, prepared_sha256=digest(out/"prepared.json"),
        features_sha256=digest(out/"features.npz"), script_sha256=digest(__file__),
        view=data.get("view", "Identical full-frame aspect-preserving 224 padding, 16 PTS-sampled frames per 2 s window; native model normalization"),
        pooling="VideoMAE: mean of all final spatial-temporal tokens, no mask/decoder. CLIP: mean of L2 frame embeddings. L2 window vectors for both."))


def metric(y, prediction, nclasses):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    precision, recall, f1, count = precision_recall_fscore_support(y, prediction, labels=list(range(nclasses)), zero_division=0)
    return dict(accuracy=float(accuracy_score(y, prediction)), macro_f1=float(f1_score(y, prediction, labels=list(range(nclasses)), average="macro", zero_division=0)),
        balanced_accuracy=float(recall.mean()), confusion=confusion_matrix(y, prediction, labels=list(range(nclasses))).tolist(),
        precision=precision.tolist(), recall=recall.tolist(), f1=f1.tolist(), support=count.tolist())


def fit(out):
    import sklearn
    from sklearn.linear_model import LogisticRegression
    data = json.loads((out/"prepared.json").read_text())
    run = json.loads((out/"inference.json").read_text())
    assert run["prepared_sha256"] == digest(out/"prepared.json")
    assert run["features_sha256"] == digest(out/"features.npz")
    vectors = np.load(out/"features.npz")
    cfg, windows = data["config"], data["windows"]
    ids = [c["id"] for c in cfg["classes"]]
    y = np.array([ids.index(w["label"]) for w in windows])
    split = np.array([w["split"] for w in windows])
    masks = {s: split == s for s in cfg["splits"]}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not ({w["video"] for w in windows if w["split"] == a} & {w["video"] for w in windows if w["split"] == b})
    result, saved_heads = {}, {}
    for name in ("videomae", "clip"):
        started = time.monotonic()
        x = vectors[name]
        candidates, best_score, best = [], -1, None
        for C in cfg["C_grid"]:
            head = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=3000, random_state=cfg["seed"])
            head.fit(x[masks["train"]], y[masks["train"]])
            assert max(head.n_iter_) < head.max_iter, "Classifier did not converge"
            validation = metric(y[masks["validation"]], head.predict(x[masks["validation"]]), len(ids))
            candidates.append(dict(C=C, validation_macro_f1=validation["macro_f1"]))
            if validation["macro_f1"] > best_score + 1e-12:
                best_score, best = validation["macro_f1"], head
        # Test is first evaluated after C is fixed; there is no validation refit.
        scores = best.predict_proba(x)
        prediction = scores.argmax(axis=1)
        result[name] = dict(C=best.C, selection=candidates, fit_seconds=time.monotonic()-started,
            scores=scores.tolist(), predictions=prediction.tolist(),
            metrics={s: metric(y[mask], prediction[mask], len(ids)) for s, mask in masks.items()})
        saved_heads[name+"_coef"] = best.coef_
        saved_heads[name+"_intercept"] = best.intercept_
        saved_heads[name+"_classes"] = best.classes_
    similarity = vectors["clip"] @ vectors["text"].T
    prediction = similarity.argmax(axis=1)
    result["clip_text"] = dict(scores=similarity.tolist(), predictions=prediction.tolist(),
        metrics={s: metric(y[mask], prediction[mask], len(ids)) for s, mask in masks.items()})
    for name, item in result.items():
        per_video = {}
        for video in sorted({w["video"] for w in windows if w["split"] == "test"}, key=lambda v: int(v[2:])):
            mask = np.array([w["video"] == video for w in windows])
            p = np.array(item["predictions"])[mask]
            per_video[video] = dict(windows=int(mask.sum()), accuracy=float((p == y[mask]).mean()))
        item["test_per_video"] = per_video
        item["test_video_mean_accuracy"] = float(np.mean([v["accuracy"] for v in per_video.values()]))
    np.savez(out/"heads.npz", **saved_heads)
    save_json(out/"fit.json", dict(models=result, sklearn=sklearn.__version__, prepared_sha256=digest(out/"prepared.json"),
        features_sha256=digest(out/"features.npz"), inference_sha256=digest(out/"inference.json"),
        heads_sha256=digest(out/"heads.npz"), script_sha256=digest(__file__),
        selection="C by validation macro F1; ties choose smaller C; training-only fit; no test tuning or refit"))
    print(json.dumps({n: {"C": v.get("C"), "test": v["metrics"]["test"], "video_mean_accuracy": v["test_video_mean_accuracy"]} for n, v in result.items()}, indent=2))


def report(out):
    os.environ.setdefault("MPLCONFIGDIR", str(out/"matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv
    data = json.loads((out/"prepared.json").read_text())
    fit_result = json.loads((out/"fit.json").read_text())
    assert fit_result["prepared_sha256"] == digest(out/"prepared.json")
    assert fit_result["features_sha256"] == digest(out/"features.npz")
    assert fit_result["heads_sha256"] == digest(out/"heads.npz")
    assert fit_result["inference_sha256"] == digest(out/"inference.json")
    run = json.loads((out/"inference.json").read_text())
    models, classes = fit_result["models"], data["config"]["classes"]
    names = {"videomae": "VideoMAE + linear", "clip": "CLIP + linear", "clip_text": "CLIP text (zero-shot)"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    labels = [c["short"] for c in classes]
    for ax, (name, model) in zip(axes, models.items()):
        m = model["metrics"]["test"]
        counts = np.array(m["confusion"])
        ax.imshow(counts / np.maximum(counts.sum(axis=1, keepdims=True), 1), vmin=0, vmax=1, cmap="Blues")
        for (r, c), count in np.ndenumerate(counts):
            ax.text(c, r, str(count), ha="center", va="center", color="white" if count / max(1, counts[r].sum()) > .55 else "#203040")
        ax.set(xticks=range(len(labels)), xticklabels=labels, yticks=range(len(labels)), yticklabels=labels,
               xlabel="Predicted action", ylabel="Manual action", title=f"{names[name]}\nmacro F1 {m['macro_f1']:.3f} · accuracy {m['accuracy']:.3f}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.suptitle("Held-out video groups · counts, row-normalized color · correlated windows")
    fig.savefig(out/"confusion.png", dpi=170)
    plt.close(fig)
    colors = ["#277c96", "#d28b39", "#6462a4", "#4a9979", "#ad618b", "#85764c"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 5.6), constrained_layout=True)
    for row, video in enumerate(("X-26", "X-27")):
        ws = [w for w in data["windows"] if w["video"] == video]
        for col, (name, title) in enumerate([("truth", "Manual action")] + [(n, names[n]) for n in models]):
            ax = axes[row, col]
            for w in ws:
                scores = np.eye(len(classes))[next(i for i, c in enumerate(classes) if c["id"] == w["label"])] if name == "truth" else models[name]["scores"][w["index"]]
                for i, score in enumerate(scores):
                    ax.scatter(w["end"], score, color=colors[i], s=22)
            if name == "truth":
                ax.set_ylim(-.08, 1.08)
            elif name != "clip_text":
                ax.set_ylim(0, 1)
            ax.set(title=title, xlabel="Window end (s)", ylabel=video, xlim=(0, data["sources"][video]["duration"]))
            ax.grid(alpha=.15)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0], [0], marker="o", color=c, linestyle="") for c in colors], labels, loc="outside lower center", ncol=6)
    fig.suptitle("X-26 / X-27 are held out · only annotated action windows are scored · gaps are unscored")
    fig.savefig(out/"timelines.png", dpi=160)
    plt.close(fig)
    summary = {}
    for split in data["config"]["splits"]:
        ws = [w for w in data["windows"] if w["split"] == split]
        summary[split] = dict(videos=sorted({w["video"] for w in ws}, key=lambda v: int(v[2:])), windows=len(ws),
            intervals=len({(w["video"], w["row"]) for w in ws}), classes=dict(Counter(w["label"] for w in ws)))
    payload = dict(classes=classes, summary=summary, models=models, windows=data["windows"], sources=data["sources"],
        assignments=data["assignments"], intervals=data["intervals"], exclusions=data["exclusions"],
        config=data["config"], inference=run, selection=fit_result["selection"], prepared_sha256=digest(out/"prepared.json"))
    save_json(out/"results.json", payload)
    (out/"results.js").write_text("window.ACTION_RESULTS = " + json.dumps(payload, ensure_ascii=False) + ";\n")
    with (out/"scores.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "video", "split", "start", "end", "truth", "model", "prediction"] + [c["id"] for c in classes])
        for w in data["windows"]:
            for name, model in models.items():
                writer.writerow([w["index"], w["video"], w["split"], w["start"], w["end"], w["label"], name,
                    classes[model["predictions"][w["index"]]]["id"]] + model["scores"][w["index"]])
    print(json.dumps(summary, indent=2))


def check():
    import copy
    config = json.loads((ROOT/"data/action_pilot.json").read_text())
    manifest = json.loads((ROOT/"review/generated/manifest.json").read_text())
    windows, _, _ = make_windows(config, manifest)
    for w in windows:
        assert w["annotation_start"] < w["start"] < w["end"] < w["annotation_end"]
    for video in ("X-27", "X-47"):
        bad = copy.deepcopy(config)
        bad["splits"]["test"].remove(video)
        bad["splits"]["train"].append(video)
        try:
            split_audit(bad, manifest)
        except AssertionError:
            pass
        else:
            raise AssertionError("Must reject paired/duplicate leakage")
    assert not any(w["video"] in ("X-47", "X-48") for w in windows)
    print(f"OK: {len(windows)} windows; all classes present; whole-video/pair/hash split; duplicate exclusion; boundary checks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "infer", "fit", "report", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == "check":
            check()
        else:
            globals()[args.mode](args.out)
