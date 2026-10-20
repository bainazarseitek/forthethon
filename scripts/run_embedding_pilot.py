"""Frozen, local Video Swin embeddings for three manually specified pairs.

Requires torch, torchvision, numpy, sklearn, matplotlib, Pillow, ffmpeg/ffprobe.
No downloads or cloud inference. Run --analyze-only to rebuild plots from saved vectors.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review/generated/embedding-pilot"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from PIL import Image

CONFIG = ROOT / "data/embedding_pilot.json"
CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/swin3d_b_1k-24f7c7c6.pth"
EXPECTED_HASH = "24f7c7c6fd97dd52439d72c2208e0785081811389185a4781405b482404e4b9b"
COLORS = {"normal": "#237b66", "violation": "#bd4057", "uncertain": "#92979d"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def run(*args):
    return subprocess.check_output(args)


def state(start, end, pair, video):
    if video == pair["reference"] or end <= pair["normal_until"] + 1e-6:
        return "normal"
    if start >= pair["violation_from"] - 1e-6:
        return "violation"
    return "uncertain"


def nearest_indexes(pts, targets):
    right = np.clip(np.searchsorted(pts, targets), 0, len(pts) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(abs(pts[left] - targets) <= abs(pts[right] - targets), left, right)


def build_segments(config, clips):
    segments = []
    duration = config["window_seconds"]

    def add(pair, video, start, purpose, title):
        end = start + duration
        assert 0 <= start < end <= clips[video]["duration"] + 1e-6
        item = dict(index=len(segments), id=f"s{len(segments):03d}", pair=pair["id"],
                    video=video, start=round(float(start), 6), end=round(float(end), 6),
                    label=state(start, end, pair, video), purpose=purpose, title=title)
        segments.append(item)
        return item["index"]

    for pair in config["pairs"]:
        for video in [pair["query"], pair["reference"]]:
            for start in np.arange(0, clips[video]["duration"] - duration + 1e-6, config["stride_seconds"]):
                add(pair, video, start, "grid", "Окно по временной сетке")
    matches = []
    for match in config["matched"]:
        pair = next(p for p in config["pairs"] if p["id"] == match["pair"])
        a = add(pair, pair["query"], match["query_start"], "matched", match["title"])
        b = add(pair, pair["reference"], match["reference_start"], "matched", match["title"])
        assert segments[a]["label"] == ("normal" if match["kind"] == "normal_control" else "violation")
        matches.append(dict(match, a=a, b=b))
    return segments, matches


def extract(config):
    import torch
    import torchvision
    from torchvision.models.video import swin3d_b

    started = time.monotonic()
    assert CHECKPOINT.is_file(), f"Put the official checkpoint at {CHECKPOINT}; this script does not download it."
    checkpoint_hash = sha(CHECKPOINT)
    assert checkpoint_hash == EXPECTED_HASH, "Unexpected checkpoint contents"
    manifest = json.loads((ROOT / "review/generated/manifest.json").read_text())
    clips = {c["id"]: c for c in manifest["clips"]}
    segments, matches = build_segments(config, clips)
    video_ids = list(dict.fromkeys(s["video"] for s in segments))
    source_hashes = {v: sha(ROOT / clips[v]["source"]) for v in video_ids}
    assert len(set(source_hashes.values())) == len(video_ids), "Duplicate videos must not cross the reference/query boundary"
    for folder in ["clips", "inputs"]:
        (OUT / folder).mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = swin3d_b(weights=None)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    model.head = torch.nn.Identity()
    model = model.eval().to(device)
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    vectors = np.zeros((len(segments), 1024), dtype=np.float32)
    inference_seconds = 0.0
    print(f"{len(segments)} segments; Video Swin B on {device}; weights verified", flush=True)

    for video in video_ids:
        path = ROOT / clips[video]["source"]
        frames = json.loads(run("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                                "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)))
        pts = np.array([float(f["best_effort_timestamp_time"]) for f in frames["frames"]])
        assert len(pts) > 0 and abs(pts[0]) < .001 and np.all(np.diff(pts) > 0)
        selected = [s for s in segments if s["video"] == video]
        for segment in selected:
            targets = segment["start"] + np.arange(config["frames_per_window"])/config["sampling_fps"]
            idx = nearest_indexes(pts, targets)
            assert np.all((pts[idx] >= segment["start"] - .04) & (pts[idx] < segment["end"]))
            assert np.max(abs(pts[idx] - targets)) < .07
            segment["frame_indexes"] = idx.tolist()
            segment["sampled_pts"] = pts[idx].round(6).tolist()
        indexes = sorted({i for s in selected for i in s["frame_indexes"]})
        expression = "+".join(f"eq(n\\,{i})" for i in indexes)
        # Keep the pretrained model's short-side resize / center crop, exposing the crop in the viewer.
        raw = run("ffmpeg", "-v", "error", "-i", str(path), "-vf",
                  f"select='{expression}',scale=256:-2:flags=bilinear,format=rgb24,crop=224:224",
                  "-fps_mode", "vfr", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1")
        images = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 224, 224, 3)
        assert len(images) == len(indexes)
        positions = {frame: i for i, frame in enumerate(indexes)}
        for segment in selected:
            sample = images[[positions[i] for i in segment["frame_indexes"]]].copy()
            Image.fromarray(sample[len(sample)//2]).save(OUT / "inputs" / f"{segment['id']}.jpg", quality=92)
            x = torch.from_numpy(sample).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)/255
            x = ((x - mean)/std).permute(1, 0, 2, 3)[None]
            tick = time.monotonic()
            with torch.inference_mode():
                vector = model(x).float().cpu().numpy()[0]
            inference_seconds += time.monotonic() - tick
            assert vector.shape == (1024,) and np.isfinite(vector).all() and np.linalg.norm(vector) > 0
            vectors[segment["index"]] = vector
            segment["clip"] = f"clips/{segment['id']}.mp4"
            segment["input_image"] = f"inputs/{segment['id']}.jpg"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(segment["start"]), "-i", str(path),
                            "-t", str(config["window_seconds"]), "-an", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            str(OUT / segment["clip"])], check=True)
            if (segment["index"] + 1) % 10 == 0:
                print(f"embedded {segment['index']+1}/{len(segments)}; {video} {segment['start']:.1f}s", flush=True)
        print(f"{video}: {len(selected)} windows, {time.monotonic()-started:.1f}s elapsed", flush=True)
    np.save(OUT / "embeddings_raw.npy", vectors)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(OUT / "embeddings.npy", vectors)
    result = dict(config=config, segments=segments, matches=matches, run=dict(
        model="torchvision Video Swin B / KINETICS400_V1; head replaced by Identity",
        dimension=1024, device=device, torch=torch.__version__, torchvision=torchvision.__version__,
        checkpoint=str(CHECKPOINT), checkpoint_sha256=checkpoint_hash, source_sha256=source_hashes,
        config_sha256=sha(CONFIG), annotations_sha256=sha(ROOT / config["annotation_source"]),
        script_sha256=sha(Path(__file__)),
        preprocessing="32 RGB frames at 15 fps, nearest source PTS; ffmpeg bilinear short side 256 (even height), center crop 224; mean .485/.456/.406, std .229/.224/.225; L2-normalized pooled features",
        model_url="https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.swin3d_b.html",
        inference_seconds=round(inference_seconds, 2), total_seconds=round(time.monotonic()-started, 2)))
    (OUT / "run.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result, vectors


def anomaly(cosine, query, bank):
    assert query and bank and not set(query).intersection(bank)
    similarities = cosine[np.ix_(query, bank)]
    best = similarities.argmax(axis=1)
    return 1 - similarities[np.arange(len(query)), best], [bank[i] for i in best]


def analyze(result, vectors):
    config, segments = result["config"], result["segments"]
    assert vectors.shape == (len(segments), 1024) and np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    cosine = np.clip(vectors @ vectors.T, -1, 1)
    assert np.allclose(cosine, cosine.T) and np.allclose(np.diag(cosine), 1, atol=1e-5)
    np.save(OUT / "cosine.npy", cosine)
    grid = [s["index"] for s in segments if s["purpose"] == "grid"]
    metrics = []
    for pair in config["pairs"]:
        query = [i for i in grid if segments[i]["video"] == pair["query"]]
        gold = [i for i in grid if segments[i]["video"] == pair["reference"]]
        other_gold = [i for i in grid if any(segments[i]["video"] == p["reference"] for p in config["pairs"] if p["id"] != pair["id"])]
        scores, nearest = anomaly(cosine, query, gold)
        other_scores, other_nearest = anomaly(cosine, query, other_gold)
        valid = np.array([segments[i]["label"] != "uncertain" for i in query])
        labels = np.array([segments[i]["label"] == "violation" for i in query])
        assert set(labels[valid]) == {False, True}
        for n, i in enumerate(query):
            segments[i].update(score=float(scores[n]), nearest=int(nearest[n]),
                               other_score=float(other_scores[n]), other_nearest=int(other_nearest[n]))
        metrics.append(dict(pair=pair["id"], title=pair["title"], query=pair["query"], reference=pair["reference"],
                            normal=int((~labels & valid).sum()), violation=int((labels & valid).sum()),
                            uncertain=int((~valid).sum()), reference_windows=len(gold),
                            auc=float(roc_auc_score(labels[valid], scores[valid])),
                            other_pair_auc=float(roc_auc_score(labels[valid], other_scores[valid])),
                            normal_score_median=float(np.median(scores[valid & ~labels])),
                            violation_score_median=float(np.median(scores[valid & labels]))))
    for match in result["matches"]:
        match["cosine"] = float(cosine[match["a"], match["b"]])
        match["distance"] = 1 - match["cosine"]
    pca = PCA(n_components=2, svd_solver="full").fit(vectors[grid])
    coordinates = pca.transform(vectors)
    for s, xy in zip(segments, coordinates):
        s["pca"] = [float(v) for v in xy]
    result.update(metrics=metrics, pca_variance=pca.explained_variance_ratio_.tolist(),
                  macro_auc=float(np.mean([m["auc"] for m in metrics])),
                  limitations=["Six source videos; windows overlap and are not independent samples.",
                    "Reference bank and chosen intervals use supplied manual annotations. No trained classifier or calibrated threshold.",
                    "AUC uses normal and violation windows from each query video only; reference windows are never scored against themselves.",
                    "Other-pair bank removes both videos of the target pair, but does not establish OOD generalization.",
                    "Center crop and short context can hide small objects and earlier actions; distance is not violation probability."])
    make_plots(result, cosine, grid)
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (OUT / "results.js").write_text("window.EMBEDDING_PILOT = " + json.dumps(result, ensure_ascii=False) + ";\n")
    print(json.dumps(dict(metrics=metrics, matched=result["matches"], run=result["run"]), ensure_ascii=False, indent=2), flush=True)


def make_plots(result, cosine, grid):
    segments, pairs = result["segments"], result["config"]["pairs"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f7f8fa", "axes.facecolor": "#ffffff", "savefig.facecolor": "#f7f8fa"})
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
    for ax, pair, metric in zip(axes, pairs, result["metrics"]):
        selected = [segments[i] for i in grid if segments[i]["video"] == pair["query"]]
        x = [(s["start"] + s["end"])/2 for s in selected]
        ax.plot(x, [s["score"] for s in selected], color="#354c60", lw=1, zorder=1)
        ax.plot(x, [s["other_score"] for s in selected], color="#9a7024", ls="--", lw=1, label="Банк из других пар")
        ax.scatter(x, [s["score"] for s in selected], c=[COLORS[s["label"]] for s in selected], s=26, zorder=3)
        ax.axvspan(pair["normal_until"], pair["violation_from"], color="#999999", alpha=.12)
        ax.axvline(pair["violation_from"], color=COLORS["violation"], lw=.8, ls=":")
        ax.set_title(f"{pair['query']} → эталон {pair['reference']}   |   AUC {metric['auc']:.2f}   |   {metric['normal']} норм. / {metric['violation']} наруш. окон", loc="left", fontsize=11)
        ax.set_ylabel("1 − max cosine")
        ax.set_xlabel("Середина окна в исходном видео, с")
        ax.grid(alpha=.15)
    handles = [Line2D([], [], marker="o", ls="", color=COLORS[k], label=v) for k, v in
               [("normal", "Норма по разметке"), ("violation", "Нарушение по разметке"), ("uncertain", "Исключённое окно")]]
    handles.append(Line2D([], [], color="#9a7024", ls="--", label="Банк из других пар"))
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
    fig.savefig(OUT / "timelines.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    xy = np.array([segments[i]["pca"] for i in grid])
    for pair, marker in zip(pairs, ["o", "s", "^"]):
        for label in COLORS:
            ids = [j for j, i in enumerate(grid) if segments[i]["pair"] == pair["id"] and segments[i]["label"] == label]
            axes[0].scatter(xy[ids, 0], xy[ids, 1], c=COLORS[label], marker=marker, s=27, alpha=.65)
    variance = result["pca_variance"]
    axes[0].set(xlabel=f"PC1 · {variance[0]:.1%}", ylabel=f"PC2 · {variance[1]:.1%}", title="PCA: цвет — состояние, форма — пара")
    axes[0].legend(handles=[Line2D([], [], color="#4a5e6c", marker=m, ls="", label=f"{p['query']} / {p['reference']}") for p, m in zip(pairs, ["o", "s", "^"])], loc="best", frameon=False)
    axes[0].grid(alpha=.15)
    heat = cosine[np.ix_(grid, grid)]
    im = axes[1].imshow(heat, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
    ticks, labels, start = [], [], 0
    for video in dict.fromkeys(segments[i]["video"] for i in grid):
        count = sum(segments[i]["video"] == video for i in grid)
        ticks.append(start + (count-1)/2); labels.append(video)
        if start:
            axes[1].axhline(start-.5, color="white", lw=.5); axes[1].axvline(start-.5, color="white", lw=.5)
        start += count
    axes[1].set(xticks=ticks, xticklabels=labels, yticks=ticks, yticklabels=labels, title="Cosine similarity всех окон")
    axes[1].tick_params(axis="x", rotation=45)
    fig.colorbar(im, ax=axes[1], shrink=.75, label="Cosine similarity · не вероятность")
    fig.savefig(OUT / "embedding-map.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    assert state(1, 3, {"reference": "b", "normal_until": 3.3, "violation_from": 5}, "a") == "normal"
    assert state(3, 5.1, {"reference": "b", "normal_until": 3.3, "violation_from": 5}, "a") == "uncertain"
    assert state(6, 8, {"reference": "b", "normal_until": 3.3, "violation_from": 5}, "a") == "violation"
    assert nearest_indexes(np.array([0., .0333, .1]), np.array([.03, .08])).tolist() == [1, 2]
    scores, neighbors = anomaly(np.array([[1., .9, .4], [.9, 1., .2], [.4, .2, 1.]]), [0, 2], [1])
    assert np.allclose(scores, [.1, .8]) and neighbors == [1, 1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        result = json.loads((OUT / "run.json").read_text())
        assert result["run"]["config_sha256"] == sha(CONFIG), "Configuration changed; rerun extraction"
        vectors = np.load(OUT / "embeddings.npy")
    else:
        result, vectors = extract(json.loads(CONFIG.read_text()))
    analyze(result, vectors)
