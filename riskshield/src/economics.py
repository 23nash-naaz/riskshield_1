"""Rupee-optimal decisions. This is the layer that turns a score into money.

Three actions, not two. Step-up (3DS/OTP) converts a false positive from
"lost sale" into "3 seconds of friction", which is what makes aggressive
thresholds affordable in production.
"""
import numpy as np

COST = {
    "cb_fee": 1500.0,        # dispute handling fee per chargeback, INR
    "margin": 0.20,          # merchant contribution margin on a good sale
    "stepup_abandon": 0.08,  # fraction of good users who drop at OTP
    "stepup_stops": 0.90,    # fraction of fraud 3DS blocks
    "stepup_opex": 2.0,      # cost of an OTP/3DS call, INR
}


def expected_cost(p, amount, action, c=COST):
    if action == "allow":
        return p * (amount + c["cb_fee"])
    if action == "block":
        return (1 - p) * amount * c["margin"]
    if action == "stepup":
        return (p * (1 - c["stepup_stops"]) * (amount + c["cb_fee"])
                + (1 - p) * amount * c["margin"] * c["stepup_abandon"]
                + c["stepup_opex"])
    raise ValueError(action)


ACTIONS = ("allow", "stepup", "block")


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
