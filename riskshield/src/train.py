"""LightGBM + isotonic calibration. Calibration is what makes the rupee math valid."""
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=63,
              min_child_samples=40, subsample=0.85, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=-1)


def fit(Xtr, ytr, Xva, yva, params=None):
    m = lgb.LGBMClassifier(**{**PARAMS, **(params or {})})
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def calibrate(model, Xva, yva):
    raw = model.predict_proba(Xva)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw, yva)
    return iso


def predict(model, iso, X):
    return iso.predict(model.predict_proba(X)[:, 1])


def ece(p, y, bins=10):
    """Expected calibration error -- proof the probabilities mean something."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)
