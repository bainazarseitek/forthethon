"""Late fusion of CLIP (per-frame image-text semantics) and X-CLIP (video-text, temporal). Pre-registered, no tuning.

Two heads, each train-only StandardScaler + balanced LR C=10 (seed 20261017):
  CLIP   - champion feature (CLIP ViT-L/14 L2 window mean, 768-d)
  X-CLIP - xclip:video (microsoft/xclip-base-patch16-zero-shot get_video_features, 512-d), L2-normalised
Prediction = argmax of the equal-weight mean of the two heads' class probabilities.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261017


def feats(d):
    v = d["X"]["xclip:video"]
    return [d["X"]["clip"], v / np.linalg.norm(v, axis=1, keepdims=True)]


def fit_predict(train, test):
    probs, classes = [], None
    for a, b in zip(feats(train), feats(test)):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=SEED))
        m.fit(a, train["y"])
        assert classes is None or (m.classes_ == classes).all()
        classes = m.classes_
        probs.append(m.predict_proba(b))
    return classes[np.mean(probs, axis=0).argmax(1)]
