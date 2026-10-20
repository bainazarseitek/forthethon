"""Pre-registered: late fusion, equal-weight average of class probabilities from the champion
(CLIP mean, StandardScaler, balanced LR C=10) and the text-similarity head (text_sim_lr: 24-d
cliptext:sim_frames, StandardScaler, balanced LR C=1)."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KEY = "cliptext:sim_frames"


def fit_predict(train, test):
    img = make_pipeline(StandardScaler(), LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=20261017))
    txt = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=3000, random_state=20261017))
    img.fit(train["X"]["clip"], train["y"])
    txt.fit(train["X"][KEY], train["y"])
    assert list(img.classes_) == list(txt.classes_)
    p = 0.5 * img.predict_proba(test["X"]["clip"]) + 0.5 * txt.predict_proba(test["X"][KEY])
    return img.classes_[p.argmax(1)]
