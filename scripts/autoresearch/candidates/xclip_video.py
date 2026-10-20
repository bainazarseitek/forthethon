"""X-CLIP video embedding alone. Pre-registered, no tuning.

Feature: xclip:video = microsoft/xclip-base-patch16-zero-shot get_video_features (cross-frame ViT-B/16 + MIT
temporal pooling, 512-d, Kinetics-400 video-text trained; scripts/autoresearch/extract_xclip.py), L2-normalised as
in X-CLIP's own similarity. Train-only StandardScaler + balanced LR C=10, seed 20261017 (same head as the champion).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261017


def feat(d):
    v = d["X"]["xclip:video"]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def fit_predict(train, test):
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=SEED))
    model.fit(feat(train), train["y"])
    return model.predict(feat(test))
