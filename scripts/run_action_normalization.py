"""Train-only StandardScaler ablation on cached action features; validation only.

Original action artifacts remain read-only. No encoders or test prediction calls.
"""
import argparse
from collections import Counter
import csv
import fcntl
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from run_action_pilot import metric, split_audit
from run_protocol_pilot import digest, save_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "review/generated/action-pilot"
OUT = ROOT / "review/generated/action-normalization"


def transform(train, validation, standardize):
    if not standardize:
        return train, validation, None
    scaler = StandardScaler().fit(train.astype(np.float64))
    assert scaler.n_samples_seen_ == len(train)
    assert np.allclose(scaler.mean_, train.astype(np.float64).mean(axis=0), atol=1e-12)
    assert np.allclose(scaler.var_, train.astype(np.float64).var(axis=0), atol=1e-12)
    return scaler.transform(train), scaler.transform(validation), scaler


def check():
    train = np.array([[1., 3., 7.], [3., 5., 7.], [5., 7., 7.]])
    validation = np.array([[100., -200., 7.]])
    a, b, first = transform(train, validation, True)
    c, _, shifted = transform(train, validation * 1000, True)
    assert np.array_equal(a, c) and np.array_equal(first.mean_, shifted.mean_)
    assert np.array_equal(first.scale_, shifted.scale_)
    assert first.scale_[-1] == 1 and np.isfinite(a).all() and np.isfinite(b).all()
    assert np.allclose(a.mean(axis=0), 0) and np.allclose(a[:, :2].std(axis=0), 1)
    assert np.allclose(b, (validation - first.mean_) / first.scale_)
    raw, raw_validation, scaler = transform(train, validation, False)
    assert np.array_equal(raw, train) and np.array_equal(raw_validation, validation) and scaler is None
    print("OK: validation perturbations cannot change train scaler; constant dimensions remain finite")


def run(out):
    started = time.monotonic()
    before_hashes = {f: digest(SOURCE / f) for f in ("prepared.json", "features.npz", "inference.json", "fit.json", "heads.npz", "results.json")}
    data = json.loads((SOURCE/"prepared.json").read_text())
    original = json.loads((SOURCE/"fit.json").read_text())
    inference = json.loads((SOURCE/"inference.json").read_text())
    assert original["prepared_sha256"] == inference["prepared_sha256"] == before_hashes["prepared.json"]
    assert original["features_sha256"] == inference["features_sha256"] == before_hashes["features.npz"]
    assert original["heads_sha256"] == before_hashes["heads.npz"]
    cfg = data["config"]
    # Audit the original snapshot, not a possibly edited live configuration.
    manifest_path = ROOT / "review/generated/manifest.json"
    assert digest(manifest_path) == data["manifest_sha256"]
    _, assignments = split_audit(cfg, json.loads(manifest_path.read_text()))
    assert assignments == data["assignments"]
    rows = {s: [w for w in data["windows"] if w["split"] == s] for s in ("train", "validation")}
    indexes = {s: np.array([w["index"] for w in ws]) for s, ws in rows.items()}
    assert not set(indexes["train"]) & set(indexes["validation"])
    ids = [c["id"] for c in cfg["classes"]]
    y = {s: np.array([ids.index(w["label"]) for w in ws]) for s, ws in rows.items()}
    assert all(set(labels) == set(range(len(ids))) for labels in y.values())
    results, parameters = {}, {}
    with np.load(SOURCE/"features.npz") as features:
        for name in ("videomae", "clip"):
            x = {s: features[name][idx] for s, idx in indexes.items()}
            assert all(np.isfinite(a).all() and a.shape[1] == 768 for a in x.values())
            variants = {}
            for variant, standardize in (("before", False), ("after", True)):
                train, validation, scaler = transform(x["train"], x["validation"], standardize)
                candidates, best, best_score = [], None, -1
                fixed_c = None
                for C in cfg["C_grid"]:
                    head = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=3000, random_state=cfg["seed"])
                    head.fit(train, y["train"])
                    assert np.array_equal(head.classes_, np.arange(len(ids)))
                    assert max(head.n_iter_) < head.max_iter, "Classifier failed to converge"
                    scores = head.predict_proba(validation)
                    metrics = metric(y["validation"], scores.argmax(axis=1), len(ids))
                    train_metrics = metric(y["train"], head.predict(train), len(ids))
                    candidate = dict(C=C, validation=metrics, train=train_metrics, iterations=int(max(head.n_iter_)))
                    candidates.append(candidate)
                    if C == original["models"][name]["C"]:
                        fixed_c = candidate
                    if metrics["macro_f1"] > best_score + 1e-12:
                        best, best_score = (head, candidate, scores), metrics["macro_f1"]
                head, candidate, scores = best
                if variant == "before":
                    old = original["models"][name]
                    assert head.C == old["C"]
                    assert np.array_equal(scores.argmax(axis=1), np.array(old["predictions"])[indexes["validation"]])
                    assert abs(best_score - old["metrics"]["validation"]["macro_f1"]) < 1e-12
                    for a, b in zip(candidates, old["selection"]):
                        assert a["C"] == b["C"] and abs(a["validation"]["macro_f1"]-b["validation_macro_f1"]) < 1e-12
                variants[variant] = dict(**candidate, grid=candidates, at_original_C=fixed_c,
                    scores=scores.tolist(), predictions=scores.argmax(axis=1).tolist(),
                    mean_max_probability=float(scores.max(axis=1).mean()))
                prefix = name + "_" + variant
                parameters[prefix+"_coef"] = head.coef_
                parameters[prefix+"_intercept"] = head.intercept_
                parameters[prefix+"_mean"] = scaler.mean_ if scaler else np.zeros(train.shape[1])
                parameters[prefix+"_scale"] = scaler.scale_ if scaler else np.ones(train.shape[1])
                if scaler:
                    parameters[prefix+"_var"] = scaler.var_
                    parameters[prefix+"_n_samples_seen"] = np.array(scaler.n_samples_seen_)
                    assert np.allclose(train.mean(axis=0), 0, atol=1e-5)
                z = (x["validation"] - parameters[prefix+"_mean"]) / parameters[prefix+"_scale"]
                logits = z @ head.coef_.T + head.intercept_
                reproduced = np.exp(logits-logits.max(axis=1, keepdims=True))
                reproduced /= reproduced.sum(axis=1, keepdims=True)
                assert np.allclose(reproduced, scores, atol=1e-6), "Saved pipeline cannot reproduce scores"
            results[name] = dict(variants=variants, train_feature_std_median=float(np.median(x["train"].std(axis=0))),
                delta_macro_f1=variants["after"]["validation"]["macro_f1"]-variants["before"]["validation"]["macro_f1"])
    np.savez(out/"pipelines.npz", **parameters)
    result = dict(models=results, classes=cfg["classes"], C_grid=cfg["C_grid"], seed=cfg["seed"],
        splits={s: dict(windows=len(ws), videos=sorted({w["video"] for w in ws}, key=lambda v: int(v[2:])),
            intervals=len({(w["video"], w["row"]) for w in ws}), classes=dict(Counter(w["label"] for w in ws))) for s, ws in rows.items()},
        validation_windows=rows["validation"], source_hashes=before_hashes, pipelines_sha256=digest(out/"pipelines.npz"),
        script_sha256=digest(__file__), sklearn=sklearn.__version__, numpy=np.__version__, python=platform.python_version(),
        device="local CPU", seconds=time.monotonic()-started, test_evaluated=False,
        method="Before: cached L2 embeddings. After: StandardScaler mean and population variance fitted only on 123 train windows, no new L2 normalization. Same balanced logistic regression and C grid; choose validation macro F1, ties smaller C; no validation refit.",
        checks=["Original prepared snapshot and feature hashes verified", "Whole-video, pair and duplicate split unchanged", "Scaler statistics match train only", "All original validation grid scores and selected predictions reproduced", "All four saved pipelines reproduce validation probabilities", "No test features transformed or scored", "Original action artifacts unchanged"])
    assert before_hashes == {f: digest(SOURCE/f) for f in before_hashes}
    save_json(out/"results.json", result)
    (out/"results.js").write_text("window.ACTION_NORMALIZATION = " + json.dumps(result, ensure_ascii=False) + ";\n")
    with (out/"scores.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "video", "start", "end", "truth", "model", "variant", "C", "prediction"] + ids)
        for name, model in results.items():
            for variant, v in model["variants"].items():
                for i, w in enumerate(rows["validation"]):
                    writer.writerow([w["index"], w["video"], w["start"], w["end"], w["label"], name, variant, v["C"], ids[v["predictions"][i]]] + v["scores"][i])
    for name, model in results.items():
        print(name, json.dumps({k: dict(C=v["C"], macro_f1=v["validation"]["macro_f1"], accuracy=v["validation"]["accuracy"], train_f1=v["train"]["macro_f1"], at_original_C=v["at_original_C"]["validation"]["macro_f1"], grid=[(r["C"], r["validation"]["macro_f1"]) for r in v["grid"]]) for k,v in model["variants"].items()}), flush=True)


def report(out):
    os.environ.setdefault("MPLCONFIGDIR", str(SOURCE/"matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = json.loads((out/"results.json").read_text())
    assert digest(out/"pipelines.npz") == result["pipelines_sha256"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, (name, model) in zip(axes, result["models"].items()):
        for key, label, color in (("before", "L2 only (before)", "#8794a4"), ("after", "Train-only z-score (after)", "#267b86")):
            variant = model["variants"][key]
            ax.plot(result["C_grid"], [c["validation"]["macro_f1"] for c in variant["grid"]], marker="o", label=label, color=color)
            ax.scatter(variant["C"], variant["validation"]["macro_f1"], s=100, facecolor="none", edgecolor=color, linewidth=2)
        ax.set(xscale="log", ylim=(0, 1), xlabel="C (same fixed grid)", ylabel="Validation macro F1", title="VideoMAE" if name == "videomae" else "CLIP")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("123 train windows -> scaler + head | 37 validation windows / 6 videos | test not evaluated")
    fig.savefig(out/"validation.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "report", "check"), default="run", nargs="?")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == "check":
            check()
        elif args.mode == "report":
            report(OUT)
        else:
            check()
            run(OUT)
            report(OUT)
