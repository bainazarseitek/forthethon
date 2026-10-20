"""Nonlinear head, candidate 2 of 2 (pre-registered; constants fixed before running).

Problem: only linear heads (and LDA/NCM) were tried; classes like machine/document may be multi-modal across scenes.
Hypothesis: a nonlinear head on train-standardized CLIP window means improves macro F1.
Spec: StandardScaler fit on the original (non-duplicated) train windows. Class balancing by oversampling train
windows to equal class counts: each class reaches the largest class count N via floor(N/n_c) full copies plus a
seeded random subset (without replacement) of the remainder. MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2,
early_stopping=False, max_iter=2000), all other arguments sklearn defaults (relu, adam, lr 1e-3, tol 1e-4).
5 members with seeds 20261017..20261021; each seed drives that member's oversampling remainder and the MLP's
init/shuffling. Prediction = argmax of the mean predict_proba over the 5 members. No hyperparameter search.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SEED, MEMBERS = 20261017, 5


def oversample(y, rng):
    classes, counts = np.unique(y, return_counts=True)
    n = counts.max()
    idx = []
    for c, k in zip(classes, counts):
        rows = np.where(y == c)[0]
        idx += [np.tile(rows, n // k), rng.choice(rows, n % k, replace=False)]
    return np.concatenate(idx)


def fit_predict(train, test):
    y = train["y"]
    sc = StandardScaler().fit(train["X"]["clip"])
    Z, Zt = sc.transform(train["X"]["clip"]), sc.transform(test["X"]["clip"])
    proba, classes = 0.0, None
    for s in range(SEED, SEED + MEMBERS):
        idx = oversample(y, np.random.default_rng(s))
        mlp = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, early_stopping=False,
                            max_iter=2000, random_state=s).fit(Z[idx], y[idx])
        proba = proba + mlp.predict_proba(Zt)
        classes = mlp.classes_
    return classes[np.argmax(proba, axis=1)].tolist()
