"""Scene-invariant weighting (pre-registered, constants fixed before running).

CLIP mean -> train-only StandardScaler -> LR C=10 (no class_weight), with sample weights
w_i = 1 / (n_classes * n_videos_with_class(c_i) * n_windows(video_i, c_i)), rescaled to mean 1.
Every class has equal total weight and, within a class, every (video, class) pair contributes equally,
so windows-heavy videos cannot dominate a class (reduces learning scene identity).
"""
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

C = 10


def pair_weights(y, videos):
    pairs = Counter(zip(videos, y))
    vids_per_class = Counter(c for _, c in pairs)
    k = len(vids_per_class)
    w = np.array([1.0 / (k * vids_per_class[c] * pairs[(v, c)]) for v, c in zip(videos, y)])
    return w * len(w) / w.sum()


def fit_predict(train, test):
    X, y = train["X"]["clip"], train["y"]
    videos = [m["video"] for m in train["meta"]]
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=3000, random_state=20261017)
    lr.fit(sc.transform(X), y, sample_weight=pair_weights(y, videos))
    return lr.predict(sc.transform(test["X"]["clip"])).tolist()
