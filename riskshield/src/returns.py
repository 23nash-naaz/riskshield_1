"""Return-risk scorer — the quiet margin killer.

Chargebacks get attention; returns silently eat 15-30% of margin in Indian
e-commerce. This module generates synthetic return data (same discipline as
synth.py) and scores return risk per transaction.

Abuse patterns injected:
  1. Wardrobing — buy, use, return within window
  2. Return velocity — account with abnormally high return rate
  3. Price arbitrage — return low-value, keep high-value from same order
  4. Serial returner — new account, many returns, no keeps
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

RNG = np.random.default_rng(42)
DAY = 86400


def generate_return_data(txn_df, return_rate=0.12, abuse_rate=0.03, seed=42):
    """Generate synthetic return events from a transaction DataFrame.

    Args:
        txn_df: DataFrame with TransactionID, uid, TransactionAmt, TransactionDT
        return_rate: fraction of transactions that get returned (legitimate)
        abuse_rate: fraction of accounts that are return abusers

    Returns:
        DataFrame with return features and is_abusive label
    """
    rng = np.random.default_rng(seed)
    n = len(txn_df)
    df = txn_df.copy()

    # Base return probability — some products more likely to be returned
    df["return_prob"] = rng.beta(2, 15, n)  # right-skewed, most low

    # --- Inject abuse patterns ---
    uids = df.uid.unique()
    abuser_uids = set(rng.choice(uids, int(len(uids) * abuse_rate), replace=False))

    # Flag abuser transactions
    is_abuser = df.uid.isin(abuser_uids)

    # Abusers have much higher return probability
    df.loc[is_abuser, "return_prob"] = rng.beta(8, 3, is_abuser.sum())

    # Generate return events
    df["is_returned"] = rng.random(n) < df.return_prob
    df["is_abusive"] = (df.is_returned & is_abuser).astype(int)

    # Return timing: legitimate returns take 5-25 days, abusive often faster
    df["return_delay_days"] = np.where(
        df.is_returned,
        np.where(df.is_abusive,
                 rng.uniform(1, 5, n),          # abusive: fast returns
                 rng.uniform(5, 25, n)),         # legit: normal returns
        0
    )

    # Only keep returned transactions for the scorer
    returned = df[df.is_returned].copy()

    # Build features
    g = df.groupby("uid")
    uid_stats = g.agg(
        total_txns=("TransactionID", "count"),
        total_returns=("is_returned", "sum"),
        avg_amount=("TransactionAmt", "mean"),
        std_amount=("TransactionAmt", "std"),
    ).fillna(0)
    uid_stats["return_rate"] = uid_stats.total_returns / uid_stats.total_txns.clip(1)
    uid_stats["high_return_rate"] = (uid_stats.return_rate > 0.3).astype(int)

    returned = returned.merge(uid_stats, on="uid", how="left")

    # Amount deviation from account baseline
    returned["amt_vs_avg"] = returned.TransactionAmt / (returned.avg_amount + 1)
    returned["amt_vs_std"] = (
        (returned.TransactionAmt - returned.avg_amount) / (returned.std_amount + 1)
    )

    # Velocity: returns in last 7 days for this account
    returned = returned.sort_values("TransactionDT")
    returned["return_velocity_7d"] = _return_velocity(returned, "uid", 7 * DAY)

    # Account age at return
    returned["acct_age_at_return"] = returned.get("acct_age_days", 0)
    returned["is_new_returner"] = (returned.acct_age_at_return < 14).astype(int)

    # Return delay feature
    returned["fast_return"] = (returned.return_delay_days < 3).astype(int)

    return returned


def _return_velocity(df, key, window):
    """Count of prior returns for the same entity within a time window."""
    out = np.zeros(len(df))
    for _, idx in df.groupby(key, sort=False).indices.items():
        t = df.TransactionDT.values[idx]
        left = np.searchsorted(t, t - window, side="left")
        out[idx] = np.arange(len(t)) - left
    return out


RETURN_FEATURES = [
    "TransactionAmt", "return_rate", "high_return_rate", "amt_vs_avg",
    "amt_vs_std", "return_velocity_7d", "is_new_returner", "fast_return",
    "total_txns", "total_returns",
]


def train_return_scorer(df):
    """Train a LightGBM return-abuse scorer with temporal split."""
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    if n < 100:
        return None, None, {"pr_auc": 0, "n": n, "note": "too few returns"}

    split = int(n * 0.7)
    tr, te = df.iloc[:split], df.iloc[split:]

    X_tr = tr[RETURN_FEATURES].fillna(0)
    y_tr = tr.is_abusive
    X_te = te[RETURN_FEATURES].fillna(0)
    y_te = te.is_abusive

    if y_tr.sum() == 0 or y_te.sum() == 0:
        return None, None, {"pr_auc": 0, "n": n, "note": "no abusive returns in split"}

    m = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, subsample=0.85, verbose=-1, n_jobs=-1
    )
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(30, verbose=False)])

    p_raw = m.predict_proba(X_te)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw, y_te)
    p_cal = iso.predict(p_raw)

    pr_auc = average_precision_score(y_te, p_cal)

    # Cost analysis: each abusive return costs ~30% of item value + processing
    abuse_cost_per_return = float(te.TransactionAmt.mean() * 0.30 + 150)

    metrics = {
        "pr_auc": round(pr_auc, 4),
        "n_test": len(te),
        "n_abusive_test": int(y_te.sum()),
        "abuse_rate": round(float(y_te.mean()), 4),
        "avg_abuse_cost_inr": round(abuse_cost_per_return, 0),
    }

    return m, iso, metrics


def score_return_risk(model, iso, row):
    """Score a single return transaction."""
    if model is None:
        return {"return_risk_score": 0.0, "action": "accept", "note": "model not available"}

    X = pd.DataFrame([row])[RETURN_FEATURES].fillna(0)
    p = float(iso.predict(model.predict_proba(X)[:, 1])[0])

    if p > 0.7:
        action = "reject_return"
    elif p > 0.3:
        action = "manual_review"
    else:
        action = "accept"

    return {
        "return_risk_score": round(p, 4),
        "action": action,
    }
