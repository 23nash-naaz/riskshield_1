"""UID reconstruction.

IEEE-CIS has no customer ID. But D1 = days since the card/account was opened,
so (TransactionDay - D1) is a constant per account. Combined with card1+addr1
it gives a stable pseudo-account key. This is the single highest-value feature
step: it turns independent rows into accounts with histories.
"""
import numpy as np

DAY = 86400


def add_uid(df):
    day = (df.TransactionDT / DAY).astype(int)
    open_day = day - df.D1.fillna(-999)
    df = df.copy()
    df["uid"] = (df.card1.astype(str) + "|" + df.addr1.astype(str)
                 + "|" + open_day.astype(int).astype(str))
    df["acct_age_days"] = df.D1.fillna(-1)
    df["is_new_acct"] = (df.acct_age_days < 7).astype(int)
    return df


def uid_quality(df):
    """Sanity metric: how well did we recover true accounts? (synthetic only)"""
    if "uid_true" not in df:
        return None
    g = df.groupby("uid").uid_true.nunique()
    return {"n_uid": len(g), "pct_pure": float((g == 1).mean())}
