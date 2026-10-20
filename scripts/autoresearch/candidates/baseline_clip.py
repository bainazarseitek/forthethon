"""Baseline: CLIP mean embedding, train-only StandardScaler, balanced LR, C=10 (5090 report champion)."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_predict(train, test):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=20261017),
    )
    model.fit(train["X"]["clip"], train["y"])
    return model.predict(test["X"]["clip"])
