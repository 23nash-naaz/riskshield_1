"""Online feature store.

Maintains per-entity running state and computes, at serve time, the SAME
features the model was trained on (history.py semantics: strictly past-only,
Welford for variance). This closes the training/serving parity gap: the API
accepts a RAW transaction, not precomputed features.

In-memory dict here; the production swap is Redis with the same interface
(see ARCHITECTURE.md). Union-find approximates the identifier graph online.
"""
import math
from collections import deque

DAY = 86400
K = 16  # sequence buffer length, matches seqenc.K


class _UF:
    """Union-find over identifier nodes -> online connected-component size."""
    def __init__(self):
        self.p, self.sz = {}, {}

    def find(self, x):
        self.p.setdefault(x, x)
        self.sz.setdefault(x, 1)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]

    def csize(self, x):
        return self.sz[self.find(x)]


class FeatureStore:
    def __init__(self):
        self.acct = {}        # uid -> running aggregates
        self.neigh = {}       # identifier node -> set of uids (degree)
        self.uf = _UF()

    # ---------- feature computation (READ state, do not mutate) ----------
    def features(self, uid, amount, ts, device, email, bin_, acct_age_days):
        a = self.acct.get(uid)
        n = a["n"] if a else 0
        mean = a["mean"] if a else float("nan")
        var = (a["m2"] / (n - 1)) if a and n > 1 else float("nan")
        amax = a["max"] if a else float("nan")
        std = math.sqrt(var) if var == var else float("nan")

        dt_prev = ts - a["last_ts"] if a else float("nan")
        dt_mean = (a["dt_sum"] / (n - 1)) if a and n > 1 else float("nan")

        recent = a["recent"] if a else deque()
        last_1h = sum(1 for t in recent if ts - t <= 3600)
        last_24h = sum(1 for t in recent if ts - t <= DAY)

        f = {
            "TransactionAmt": amount,
            "f_amt_log": math.log1p(amount),
            "f_hour": int(ts / 3600 % 24),
            "acct_age_days": acct_age_days,
            "is_new_acct": int(acct_age_days < 7),
            "h_n_prev": n,
            "f_amt_z": (amount - mean) / (std + 1.0) if n > 1 else 0.0,
            "f_amt_ratio": amount / (mean + 1.0) if n else 0.0,
            "f_amt_vs_max": amount / (amax + 1.0) if n else 0.0,
            "f_dt_prev": dt_prev if dt_prev == dt_prev else 0.0,
            "f_dt_ratio": (dt_prev / (dt_mean + 1.0)) if dt_prev == dt_prev and dt_mean == dt_mean else 0.0,
            "f_txn_last_1h": last_1h,
            "f_txn_last_24h": last_24h,
        }
        # graph
        u = f"u:{uid}"
        for name, val in (("DeviceInfo", device), ("P_emaildomain", email),
                          ("card_bin", bin_)):
            node = f"{name}:{val}"
            f[f"g_deg_{name}"] = len(self.neigh.get(node, ()))
        f["g_uid_deg"] = len(self.neigh.get(u, ()))
        f["g_comp_size"] = self.uf.csize(u) if u in self.uf.p else 1
        f["g_dev_per_uid"] = f["g_deg_DeviceInfo"] / (f["g_uid_deg"] + 1)
        return f

    def sequence(self, uid):
        a = self.acct.get(uid)
        return list(a["seq"]) if a else []

    # ---------- state update (call AFTER scoring the txn) ----------
    def update(self, uid, amount, ts, device, email, bin_, seq_feats=None):
        a = self.acct.setdefault(uid, {
            "n": 0, "mean": 0.0, "m2": 0.0, "max": 0.0,
            "last_ts": None, "dt_sum": 0.0,
            "recent": deque(maxlen=512), "seq": deque(maxlen=K)})
        a["n"] += 1
        d = amount - a["mean"]
        a["mean"] += d / a["n"]
        a["m2"] += d * (amount - a["mean"])
        a["max"] = max(a["max"], amount)
        if a["last_ts"] is not None:
            a["dt_sum"] += ts - a["last_ts"]
        a["last_ts"] = ts
        a["recent"].append(ts)
        if seq_feats is not None:
            a["seq"].append(seq_feats)

        u = f"u:{uid}"
        for name, val in (("DeviceInfo", device), ("P_emaildomain", email),
                          ("card_bin", bin_)):
            node = f"{name}:{val}"
            self.neigh.setdefault(node, set()).add(uid)
            self.neigh.setdefault(u, set()).add(node)
            self.uf.union(u, node)
