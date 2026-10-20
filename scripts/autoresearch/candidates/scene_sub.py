"""Scene subtraction (pre-registered, one look): x = L2(clip - scene_mean), where scene_mean is the
label-free mean CLIP frame embedding over the whole video at 2 fps (extract_clip_scene.py).
Then train-only StandardScaler, balanced LR C=10 (same head as champion)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def feats(d):
    x = d["X"]["clip"] - d["X"]["clipscene:mean"]
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def fit_predict(train, test):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=10, class_weight="balanced", max_iter=3000, random_state=20261017))
    model.fit(feats(train), train["y"])
    return model.predict(feats(test))
