"""Horizontal-flip train augmentation + TEST-TIME augmentation (pre-registered before any flipped
feature existed; ONE evaluation).

Hypothesis: flipped frames are label-preserving for all 6 actions; training on original + mirrored
windows AND averaging the head's probabilities over a test window and its mirror removes left/right
layout as a cue and improves cross-scene macro F1.
Counter-hypothesis: a fixed camera layout (safe/machine side) is a useful cue that flipping destroys,
or CLIP features are already nearly flip-invariant so the effect is ~0.

Features: CLIP window mean = L2(mean over the 16 frames of L2 per-frame projections).
  original window: X["clip"] (champion feature)
  flipped window:  L2(mean(X["clipflip:frames"])) (content columns mirrored in place;
                   scripts/autoresearch/extract_clip_flip.py)
Training identical to flip_trainaug: original + flipped train windows (2N rows, labels duplicated),
StandardScaler fit on those 2N rows, balanced LR C=10, max_iter=3000, random_state=20261017.
Prediction: argmax of (P(original test window) + P(flipped test window)) / 2.
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
    t_orig, t_flip = views(test)
    p = (model.predict_proba(t_orig) + model.predict_proba(t_flip)) / 2
    return model.classes_[p.argmax(1)]
