"""Flip-SYMMETRIZED CLIP feature, no augmentation (pre-registered before any flipped feature existed;
ONE evaluation).

Hypothesis: averaging a window's embedding with its mirror's makes the feature exactly invariant to
left/right layout, so the head cannot key on which side of the frame things are, which improves
cross-scene macro F1.
Counter-hypothesis: a fixed camera layout (safe/machine side) is a useful cue that symmetrizing
destroys, or CLIP features are already nearly flip-invariant so the effect is ~0.

Feature: m_o = X["clip"] (champion: L2(mean over 16 frames of L2 per-frame projections)),
         m_f = L2(mean(X["clipflip:frames"])) (content columns mirrored in place;
               scripts/autoresearch/extract_clip_flip.py),
         x   = L2((m_o + m_f) / 2), for train and test alike. N train rows, no augmentation.
StandardScaler fit on the train rows, balanced LR C=10, max_iter=3000, random_state=20261017.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C, SEED = 10, 20261017


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def feats(d):
    return l2((d["X"]["clip"] + l2(d["X"]["clipflip:frames"].mean(1))) / 2)


def fit_predict(train, test):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED))
    model.fit(feats(train), train["y"])
    return model.predict(feats(test))
