"""Horizontal-flip TRAIN augmentation (pre-registered before any flipped feature existed; ONE evaluation).

Hypothesis: a horizontally flipped frame is label-preserving for all 6 actions (none is left/right
specific); adding each training window's mirrored copy shows the head every scene in a mirrored
layout and improves cross-scene macro F1 (esp. hand_count, 3 scenario groups).
Counter-hypothesis: cameras share a layout (safe/machine always on the same side), so position is a
useful cue and flipping hurts safe/machine; or CLIP scene identity is appearance, which a flip does
not change, so the effect is ~0.

Features: CLIP window mean = L2(mean over the 16 frames of L2 per-frame projections).
  original window: X["clip"] (champion feature; equals L2(mean(clipframes:frames)), cos >= 0.9999999)
  flipped window:  L2(mean(X["clipflip:frames"])), the same frames with the 123 content columns
                   mirrored in place (scripts/autoresearch/extract_clip_flip.py)
Train rows: original + flipped windows (2N rows, labels duplicated); StandardScaler fit on those 2N rows.
Balanced LR C=10, max_iter=3000, random_state=20261017. Predict the ORIGINAL test windows only.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C, SEED = 10, 20261017


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def views(d):
    return d["X"]["clip"], l2(d["X"]["clipflip:frames"].mean(1))


def fit_predict(train, test):
    orig, flip = views(train)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED))
    model.fit(np.vstack([orig, flip]), np.concatenate([train["y"], train["y"]]))
    return model.predict(views(test)[0])
