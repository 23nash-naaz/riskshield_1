"""Honest metrics. PR-AUC not ROC-AUC (3% base rate makes ROC flattering),
bootstrap CIs, slice breakdown, and rupees vs three baselines."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from economics import decide_vec, expected_cost, COST


def boot_ap(y, p, n=300, seed=0):
    """Percentile bootstrap. Jitter breaks the score ties created by resampling
    duplicates, which otherwise deflates AP and yields a CI that misses the
    point estimate."""
    rng = np.random.default_rng(seed)
    eps = 1e-9 * (p.max() - p.min() + 1e-12)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        vals.append(average_precision_score(y[i], p[i] + rng.normal(0, eps, len(i))))
    return (float(average_precision_score(y, p)),
            float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def op_point(y, p, amount, c=COST):
    acts, _ = decide_vec(p, amount, c)
    flagged = acts != "allow"
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    good = int((y == 0).sum())
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "fp_per_1k_good": 1000 * fp / max(good, 1),
        "review_rate": float(flagged.mean()),
        "pct_stepup": float((acts == "stepup").mean()),
        "pct_block": float((acts == "block").mean()),
    }


def rupees(y, p, amount, c=COST):
    """Realised rupee cost per 1000 txns, model vs baselines."""
    n = len(y)
    def cost_of(acts):
        return sum(expected_cost(y[i].astype(float), amount[i], acts[i], c)
                   for i in range(n)) * 1000 / n
    model_acts, _ = decide_vec(p, amount, c)
    hi = np.quantile(amount, 0.97)
    return {
        "model": cost_of(model_acts),
        "allow_all": cost_of(np.array(["allow"] * n)),
        "block_all": cost_of(np.array(["block"] * n)),
        "rule_amount": cost_of(np.where(amount > hi, "block", "allow")),
        "stepup_all": cost_of(np.array(["stepup"] * n)),
    }


def slices(df, y, p, amount, c=COST):
    acts, _ = decide_vec(p, amount, c)
    d = pd.DataFrame({"y": y, "p": p, "amt": amount, "act": acts})
    d["amt_band"] = pd.qcut(amount, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    d["acct"] = np.where(df.is_new_acct.values == 1, "new", "returning")
    d["night"] = np.where(df.f_hour.values.astype(int) < 6, "00-06", "06-24")
    rows = []
    for col in ["amt_band", "acct", "night", "card4"]:
        s = df[col].values if col in df else d[col].values
        for v in pd.unique(s):
            m = s == v
            if m.sum() < 50 or d.y[m].sum() == 0:
                continue
            sub = d[m]
            fl = sub.act != "allow"
            rows.append({
                "slice": f"{col}={v}", "n": int(m.sum()),
                "fraud_rate": float(sub.y.mean()),
                "PR_AUC": average_precision_score(sub.y, sub.p),
                "recall": float((fl & (sub.y == 1)).sum() / max(sub.y.sum(), 1)),
                "fp_per_1k": 1000 * float((fl & (sub.y == 0)).sum() / max((sub.y == 0).sum(), 1)),
            })
    return pd.DataFrame(rows).sort_values("PR_AUC")
