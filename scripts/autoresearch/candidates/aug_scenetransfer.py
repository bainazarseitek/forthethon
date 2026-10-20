"""Scene-offset transfer augmentation (pre-registered, constants fixed before running; ONE evaluation).

CLIP mean -> train-only StandardScaler (Z). On TRAIN only: class means m_c; per-video scene offset
mu_v = mean over video v's train windows of (z_i - m_{y_i}). For each train window i (class c, video v)
add S=3 synthetic samples z_i + (mu_{v'} - mu_v), v' = 3 distinct other train videos drawn uniformly
(rng seed 20261017, fresh per fit), label stays c. Balanced LR C=10 on real+synthetic; test is only scaled.
Teaches the head that moving a window into another scene's offset does not change the action.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

S, C, SEED = 3, 10, 20261017


def scene_transfer(Z, y, videos, rng):
    R = Z.copy()
    for c in np.unique(y):
        R[y == c] -= Z[y == c].mean(0)
    vids = np.unique(videos)
    mu = {v: R[videos == v].mean(0) for v in vids}
    Xs, ys = [], []
    for z, c, v in zip(Z, y, videos):
        for w in rng.choice(vids[vids != v], S, replace=False):
            Xs.append(z + mu[w] - mu[v])
            ys.append(c)
    return np.vstack([Z, np.array(Xs)]), np.concatenate([y, np.array(ys)])


def fit_predict(train, test):
    X, y = train["X"]["clip"], train["y"]
    videos = np.array([m["video"] for m in train["meta"]])
    sc = StandardScaler().fit(X)
    Za, ya = scene_transfer(sc.transform(X), y, videos, np.random.default_rng(SEED))
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED)
    lr.fit(Za, ya)
    return lr.predict(sc.transform(test["X"]["clip"])).tolist()
