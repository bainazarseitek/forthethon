"""Late fusion: equal-weight mean of class probabilities from the CLIP champion head and the SigLIP head.

Pre-registered before any SigLIP evaluation. Heads: CLIP window mean (as baseline_clip.py) and
SigLIP so400m window mean (as siglip_mean.py), each scaler + balanced LR C=10; argmax of the
probability mean. Weights 0.5/0.5, C=10, seed fixed, no tuning.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261017


def feats(d):
    m = d["X"]["siglip:frames"].mean(1)
    return [d["X"]["clip"], m / np.linalg.norm(m, axis=1, keepdims=True)]


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
