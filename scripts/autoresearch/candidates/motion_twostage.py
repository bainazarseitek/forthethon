"""Two-stage: champion LR (CLIP mean, scaler, balanced LR C=10). Windows predicted hand_count or machine are re-decided by a
binary hand-vs-machine LR (C=1, balanced, fixed a priori) on the 6 scalar motion stats, trained on train hand_count+machine only."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from motion_scalars import clip_mean, motion_stats  # noqa: E402

SEED = 20261017
PAIR = ("hand_count", "machine")


def fit_predict(train, test):
    ftr, fte, y = train["X"]["clipframes:frames"], test["X"]["clipframes:frames"], train["y"]
    main = make_pipeline(StandardScaler(), LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=SEED))
    main.fit(clip_mean(ftr), y)
    pred = main.predict(clip_mean(fte)).astype(object)
    mask_tr = np.isin(y, PAIR)
    if len(set(y[mask_tr])) < 2:
        return pred
    head = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=3000, random_state=SEED))
    head.fit(motion_stats(ftr[mask_tr]), y[mask_tr])
    m = np.isin(pred, PAIR)
    if m.any():
        pred[m] = head.predict(motion_stats(fte[m]))
    return pred
