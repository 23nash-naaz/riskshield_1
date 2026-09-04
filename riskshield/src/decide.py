"""Score -> action + reason codes.

DEFENCE-ONLY: reason codes are coarse buckets, never exact thresholds or
feature values. A leaked API response must not be reverse-engineerable into
an evasion recipe.
"""
import numpy as np
from economics import decide, COST

# feature -> coarse public bucket. Anything not listed is never disclosed.
CODES = {
    "f_amt_z": "AMOUNT_OFF_BASELINE",
    "f_amt_ratio": "AMOUNT_OFF_BASELINE",
    "f_amt_vs_max": "AMOUNT_OFF_BASELINE",
    "f_txn_last_1h": "VELOCITY_ANOMALY",
    "f_txn_last_24h": "VELOCITY_ANOMALY",
    "f_dt_prev": "VELOCITY_ANOMALY",
    "g_deg_DeviceInfo": "SHARED_DEVICE_CLUSTER",
    "g_dev_per_uid": "SHARED_DEVICE_CLUSTER",
    "g_uid_deg": "SHARED_DEVICE_CLUSTER",
    "is_new_acct": "THIN_ACCOUNT_HISTORY",
    "acct_age_days": "THIN_ACCOUNT_HISTORY",
    "h_n_prev": "THIN_ACCOUNT_HISTORY",
    "f_hour": "UNUSUAL_TIME",
}


def reasons(shap_row, cols, k=3):
    order = np.argsort(-np.abs(shap_row))
    out = []
    for i in order:
        c = CODES.get(cols[i])
        if c and shap_row[i] > 0 and c not in out:
            out.append(c)
        if len(out) == k:
            break
    return out or ["MODEL_COMPOSITE"]


def score_one(model, iso, cols, row, amount, merchant_margin=0.20):
    import pandas as pd
    X = pd.DataFrame([row])[cols]
    p = float(iso.predict(model.predict_proba(X)[:, 1])[0])
    c = {**COST, "margin": merchant_margin}
    action, costs = decide(p, amount, c)
    sv = model.predict(X, pred_contrib=True)[0][:-1]
    return {
        "risk_score": round(p, 4),
        "action": action,
        "reason_codes": reasons(sv, cols),
        "expected_cost_inr": {k: round(v, 2) for k, v in costs.items()},
        "saved_vs_allow_inr": round(costs["allow"] - costs[action], 2),
    }
