"""Pre-registered: CLIP text-prompt similarity features only (24 prompts, cliptext:sim_frames),
train-only StandardScaler, balanced LR C=1. Hypothesis: scene-independent text priors generalise
across scenes better than the 768-d image embedding (esp. hand_count)."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KEY = "cliptext:sim_frames"


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=3000, random_state=20261017))


def fit_predict(train, test):
    m = model().fit(train["X"][KEY], train["y"])
    return m.predict(test["X"][KEY])
