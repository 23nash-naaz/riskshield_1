"""Adversarial validation.

Train a classifier to tell TRAIN rows from TEST rows. If it succeeds, the
distributions differ, and the features it leans on are the ones that will
decay in production. Drop them even if they look strong offline.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


def audit(train, test, cols, y_train, auc_thresh=0.80, top=8):
    """Drop a feature only if it drifts hard AND contributes little signal.
    A feature can drift and still be worth keeping if it carries the label;
    dropping on drift alone throws away real predictive power."""
    X = pd.concat([train[cols], test[cols]], ignore_index=True)
    z = np.r_[np.zeros(len(train)), np.ones(len(test))]
    dm = lgb.LGBMClassifier(n_estimators=120, num_leaves=31, verbose=-1).fit(X, z)
    auc = roc_auc_score(z, dm.predict_proba(X)[:, 1])

    sm = lgb.LGBMClassifier(n_estimators=120, num_leaves=31, verbose=-1)\
        .fit(train[cols], y_train)
    d = pd.Series(dm.feature_importances_, index=cols)
    s = pd.Series(sm.feature_importances_, index=cols)
    d, s = d / max(d.max(), 1), s / max(s.max(), 1)
    ratio = (d + 1e-6) / (s + 1e-6)          # drift per unit of signal

    unstable = []
    if auc >= auc_thresh:
        cand = ratio[(d > 0.25) & (s < 0.10)].sort_values(ascending=False)
        unstable = cand.head(top).index.tolist()
    return {"auc": float(auc), "unstable": unstable,
            "drift_imp": d, "signal_imp": s, "ratio": ratio.sort_values(ascending=False)}
