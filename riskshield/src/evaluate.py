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
    """Realised rupee cost per 1000 txns, model vs baselines, with CI and decomposition."""
    n = len(y)
    
    # Vectorized computation of expected costs for each action
    from economics import ACTIONS
    M = np.stack([expected_cost(y.astype(float), amount, a, c) for a in ACTIONS]) # (n_actions, n)
    
    def cost_of(acts):
        # acts can be an array of action strings
        act_idx = np.array([ACTIONS.index(a) for a in acts])
        return M[act_idx, np.arange(n)]
        
    model_acts, _ = decide_vec(p, amount, c)
    hi = np.quantile(amount, 0.97)
    
    # Cost Decomposition for the model
    # Fees & Goods (Fraud losses)
    is_fraud = y == 1
    is_good = y == 0
    fees = sum((model_acts == "allow") & is_fraud) * c["cb_fee"] + sum((model_acts == "stepup") & is_fraud) * (1 - c["stepup_stops"]) * c["cb_fee"]
    goods = sum(((model_acts == "allow") & is_fraud) * amount) + sum(((model_acts == "stepup") & is_fraud) * amount) * (1 - c["stepup_stops"])
    friction_opex = sum(model_acts == "stepup") * c["stepup_opex"] + sum(model_acts == "review") * c["review_opex"]
    margin_abandon = sum(((model_acts == "stepup") & is_good) * amount) * c["margin"] * c["stepup_abandon"]
    false_blocks = sum(((model_acts == "block") & is_good) * amount) * c["margin"]

    decomp = {
        "fees": float(fees * 1000 / n),
        "goods": float(goods * 1000 / n),
        "friction": float((friction_opex + margin_abandon) * 1000 / n),
        "false_blocks": float(false_blocks * 1000 / n),
    }

    # Best fixed threshold (p > thresh -> block, else allow)
    # Simple grid search for best threshold
    grid = np.linspace(0, 1, 101)
    best_fixed_cost = float("inf")
    for thresh in grid:
        acts_fixed = np.where(p > thresh, "block", "allow")
        cost = np.sum(cost_of(acts_fixed))
        if cost < best_fixed_cost:
            best_fixed_cost = cost
    best_fixed = best_fixed_cost * 1000 / n

    # Oracle (perfect knowledge of y)
    # For y=1 (fraud), best action is block (0 cost if amount * margin > 0, actually stepup could be cheaper if negative cost, but it's not)
    oracle_acts = np.where(y == 1, "block", "allow")
    oracle = float(np.sum(cost_of(oracle_acts)) * 1000 / n)
    
    allow_all_cost = float(np.sum(cost_of(np.array(["allow"] * n))) * 1000 / n)
    model_cost = float(np.sum(cost_of(model_acts)) * 1000 / n)
    
    # % of achievable savings
    max_savings = allow_all_cost - oracle
    model_savings = allow_all_cost - model_cost
    pct_savings_captured = (model_savings / max_savings * 100) if max_savings > 0 else 0.0

    # Bootstrap CI for model cost
    rng = np.random.default_rng(0)
    model_costs_arr = cost_of(model_acts)
    boot_vals = [np.sum(rng.choice(model_costs_arr, size=n, replace=True)) * 1000 / n for _ in range(300)]
    
    return {
        "model": model_cost,
        "model_ci_2.5": float(np.percentile(boot_vals, 2.5)),
        "model_ci_97.5": float(np.percentile(boot_vals, 97.5)),
        "allow_all": allow_all_cost,
        "block_all": float(np.sum(cost_of(np.array(["block"] * n))) * 1000 / n),
        "rule_amount": float(np.sum(cost_of(np.where(amount > hi, "block", "allow"))) * 1000 / n),
        "stepup_all": float(np.sum(cost_of(np.array(["stepup"] * n))) * 1000 / n),
        "best_fixed_threshold": best_fixed,
        "oracle": oracle,
        "pct_savings_captured": pct_savings_captured,
        "decomposition": decomp
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
