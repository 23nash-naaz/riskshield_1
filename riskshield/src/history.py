"""Per-entity history features. STRICTLY past-only.

Every aggregate is shift(1) before use, so row i never sees its own value or
any future row. A plain groupby().transform('mean') would leak the future and
is the most common way fraud models produce fake offline scores.
"""
import numpy as np
import pandas as pd

DAY = 86400


def _past(g, fn):
    return g.shift(1).expanding().agg(fn)


def add_history(df, key="uid"):
    df = df.sort_values("TransactionDT").copy()
    g = df.groupby(key, sort=False)

    df["h_n_prev"] = g.cumcount()
    amt = g.TransactionAmt
    df["h_amt_mean"] = amt.transform(lambda s: s.shift(1).expanding().mean())
    df["h_amt_std"] = amt.transform(lambda s: s.shift(1).expanding().std())
    df["h_amt_max"] = amt.transform(lambda s: s.shift(1).expanding().max())

    # amount relative to this account's own baseline -- not the global one
    df["f_amt_z"] = (df.TransactionAmt - df.h_amt_mean) / (df.h_amt_std + 1.0)
    df["f_amt_ratio"] = df.TransactionAmt / (df.h_amt_mean + 1.0)
    df["f_amt_vs_max"] = df.TransactionAmt / (df.h_amt_max + 1.0)

    # inter-arrival velocity
    dt = g.TransactionDT
    df["f_dt_prev"] = df.TransactionDT - dt.shift(1)
    df["h_dt_mean"] = dt.transform(lambda s: s.diff().shift(1).expanding().mean())
    df["f_dt_ratio"] = df.f_dt_prev / (df.h_dt_mean + 1.0)

    df["f_txn_last_1h"] = _rolling_count(df, key, 3600)
    df["f_txn_last_24h"] = _rolling_count(df, key, DAY)
    df["f_hour"] = (df.TransactionDT / 3600 % 24).astype(int)
    df["f_amt_log"] = np.log1p(df.TransactionAmt)
    return df


def _rolling_count(df, key, window_s):
    """Count of prior txns for the same entity inside a time window."""
    out = np.zeros(len(df))
    for _, idx in df.groupby(key, sort=False).indices.items():
        t = df.TransactionDT.values[idx]
        left = np.searchsorted(t, t - window_s, side="left")
        out[idx] = np.arange(len(t)) - left
    return out
