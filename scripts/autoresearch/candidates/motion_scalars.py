"""Champion CLIP mean + 6 scalar motion stats (consecutive-frame cosine distance mean/max/std, mean |diff| L2,
first-last cosine distance, mean frame-to-centroid distance), standardized and scaled by W=3 (fixed a priori); balanced LR C=10."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

W = 3.0
SEED = 20261017


def motion_stats(f):
    fn = f / np.linalg.norm(f, axis=2, keepdims=True)
    cd = 1 - (fn[:, 1:] * fn[:, :-1]).sum(2)  # (n, 15)
    mean = fn.mean(1)
    mean_n = mean / np.linalg.norm(mean, axis=1, keepdims=True)
    to_c = 1 - (fn * mean_n[:, None]).sum(2)
    return np.stack([
        cd.mean(1), cd.max(1), cd.std(1),
        np.linalg.norm(np.diff(f, axis=1), axis=2).mean(1),
        1 - (fn[:, 0] * fn[:, -1]).sum(1),
        to_c.mean(1),
    ], 1)


def clip_mean(f):
    m = f.mean(1)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def fit_predict(train, test):
    ftr, fte = train["X"]["clipframes:frames"], test["X"]["clipframes:frames"]
    sm, ss = StandardScaler().fit(clip_mean(ftr)), StandardScaler().fit(motion_stats(ftr))
    feat = lambda f: np.hstack([sm.transform(clip_mean(f)), W * ss.transform(motion_stats(f))])
    model = LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=SEED)
    model.fit(feat(ftr), train["y"])
    return model.predict(feat(fte))
