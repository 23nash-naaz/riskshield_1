"""Delayed chargeback labels.

A chargeback lands 30-90 days after the transaction. So recent transactions
are UNLABELLED, not negative. Training on them as negatives is the standard
production bug: it teaches the model that recent fraud is normal.

We train only on the matured window, and use Elkan-Noto PU correction to
rescale scores when the fresh window must be included.
"""
DAY = 86400
MATURITY_DAYS = 60


def maturity_split(df, maturity_days=MATURITY_DAYS):
    cutoff = df.TransactionDT.max() - maturity_days * DAY
    mature = df[df.TransactionDT < cutoff]
    fresh = df[df.TransactionDT >= cutoff]
    return mature, fresh, cutoff


def estimate_c(model_scores_on_known_positives):
    """Elkan-Noto c = P(labelled | positive). Mean score on held-out positives."""
    return float(max(model_scores_on_known_positives.mean(), 1e-3))


def pu_correct(p, c):
    """Observed P(labelled|x) -> true P(positive|x)."""
    return (p / c).clip(0, 1)
