"""Rupee-optimal decisions. This is the layer that turns a score into money.

Four actions, not two. Step-up (3DS/OTP) converts a false positive from
"lost sale" into "3 seconds of friction". REVIEW routes high-ticket borderline
cases to human analysts to protect LTV.

Cost Constants Split:
- Learnable (Deterministic): cb_fee, stepup_opex, review_opex
- Unlearnable (Stochastic): margin, stepup_abandon, stepup_stops
"""
import numpy as np

COST = {
    # Learnable (Deterministic)
    "cb_fee": 1500.0,        # dispute handling fee per chargeback, INR
    "stepup_opex": 2.0,      # cost of an OTP/3DS call, INR
    "review_opex": 50.0,     # cost of human analyst review, INR
    
    # Unlearnable (Stochastic)
    "margin": 0.20,          # merchant contribution margin on a good sale
    "stepup_abandon": 0.08,  # fraction of good users who drop at OTP
    "stepup_stops": 0.90,    # fraction of fraud 3DS blocks
    "rolling_cb_ratio": 0.008, # rolling 30-day chargeback ratio (0.8%)
}


def expected_cost(p, amount, action, c=COST):
    # Cascading dispute penalties: Visa VAMP / Mastercard ECP
    # If ratio approaches 1%, fee multiplies by 10x
    cb_fee = c["cb_fee"] * (10.0 if c.get("rolling_cb_ratio", 0) >= 0.01 else 1.0)
    
    if action == "allow":
        return p * (amount + cb_fee)
    if action == "block":
        return (1 - p) * amount * c["margin"]
    if action == "stepup":
        return (p * (1 - c["stepup_stops"]) * (amount + cb_fee)
                + (1 - p) * amount * c["margin"] * c["stepup_abandon"]
                + c["stepup_opex"])
    if action == "review":
        return c["review_opex"]
    raise ValueError(action)


ACTIONS = ("allow", "stepup", "review", "block")


def decide(p, amount, c=COST):
    costs = {a: expected_cost(p, amount, a, c) for a in ACTIONS}
    best = min(costs, key=costs.get)
    return best, costs


def decide_vec(p, amount, c=COST):
    M = np.stack([expected_cost(p, amount, a, c) for a in ACTIONS])
    i = M.argmin(0)
    return np.array(ACTIONS)[i], M.min(0)


def realised_cost(p_true_label, amount, action, c=COST):
    """Actual rupees given the realised outcome (for eval, not decisions)."""
    y = p_true_label
    return expected_cost(y.astype(float), amount, action, c)


def thresholds_for_margin(margin, amount, c=COST):
    """Score cut-offs where each action becomes optimal, for one merchant."""
    cc = {**c, "margin": margin}
    grid = np.linspace(0, 1, 2001)
    acts, _ = decide_vec(grid, np.full_like(grid, amount), cc)
    out = {}
    for a in ACTIONS:
        hit = np.where(acts == a)[0]
        if len(hit):
            out[a] = float(grid[hit[0]])
    return out
