"""NAP (K=8) on CLIP mean + standardized PCA16 motion block, balanced LR C=10.

Pre-registered (iteration 5, ONE evaluation; constants fixed before running, none tuned on evaluator output).
Hypothesis: NAP removes scene identity and PCA16 frame-diff motion adds the hand-vs-machine motion cue;
they act on different failure causes, so the combination beats the champion and is more stable on val
than NAP alone (sceneinv_nap: CV 0.698, val -0.034; motion_pca16: CV +0.030, val +0.026).

Spec (both source modules imported by file path, their code reused verbatim):
- Scene block = sceneinv_nap.py: `clip` -> train-only StandardScaler -> nap_basis (K=8 within-class
  between-video directions, TRAIN only) projected out of train and test.
- Motion block = motion_pca16.py: parts() per-dim mean |frame diff| of `clipframes:frames` ->
  train-only StandardScaler + PCA(16) -> train-only StandardScaler (the column standardisation that
  motion_pca16's final scaler applies to these columns). Not NAP-projected.
- Head: hstack([NAP(clip), motion16]) -> LogisticRegression(C=10, balanced, seed 20261017).
"""
import importlib.util
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

K_NAP, K_MOTION, C, SEED = 8, 16, 10, 20261017
HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"nap_motion_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nap, mot = _load("sceneinv_nap"), _load("motion_pca16")
assert (nap.K, nap.C, mot.K, mot.SEED) == (K_NAP, C, K_MOTION, SEED), "source modules changed"


def fit_predict(train, test):
    y = train["y"]
    videos = np.array([m["video"] for m in train["meta"]])
    sc = StandardScaler().fit(train["X"]["clip"])
    Z, Zt = sc.transform(train["X"]["clip"]), sc.transform(test["X"]["clip"])
    V = nap.nap_basis(Z, y, videos, K_NAP)
    P = lambda A: A - (A @ V.T) @ V
    _, d_tr = mot.parts(train["X"]["clipframes:frames"])
    _, d_te = mot.parts(test["X"]["clipframes:frames"])
    red = make_pipeline(
        StandardScaler(), PCA(n_components=K_MOTION, random_state=SEED), StandardScaler()
    ).fit(d_tr)
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED)
    lr.fit(np.hstack([P(Z), red.transform(d_tr)]), y)
    return lr.predict(np.hstack([P(Zt), red.transform(d_te)])).tolist()
