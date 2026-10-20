"""SigLIP so400m-patch14-384 window mean (L2) + scaler + balanced LR C=10.

Pre-registered before any SigLIP evaluation: the champion recipe (baseline_clip.py) with only the
encoder swapped. Feature = mean of the 16 per-frame L2 SigLIP image embeddings (siglip:frames,
see extract_siglip_frames.py), then L2. C=10, seed fixed, no tuning.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261017


def feat(d):
    m = d["X"]["siglip:frames"].mean(1)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def fit_predict(train, test):
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=SEED))
    model.fit(feat(train), train["y"])
    return model.predict(feat(test))
