"""Generate an IEEE-CIS-schema-compatible dataset.

Real run: drop train_transaction.csv into data/ and loader.py uses it instead.
Synthetic mode exists so the pipeline is runnable + testable end to end.
Injects the three fraud mechanisms the model is supposed to catch:
  1. account takeover  -> burst of txns on an old account, amounts off its baseline
  2. card testing      -> many small txns, one device, many cards
  3. bust-out          -> new account, ramps up, then one huge txn
"""
import numpy as np
import pandas as pd

DAY = 86400
RNG = np.random.default_rng(7)


def _base_accounts(n_acct, days):
    open_day = RNG.integers(-400, days - 5, n_acct)
    return pd.DataFrame({
        "uid_true": np.arange(n_acct),
        "card1": RNG.integers(1000, 18000, n_acct),
        "addr1": RNG.integers(100, 540, n_acct),
        "open_day": open_day,
        "base_amt": np.exp(RNG.normal(6.2, 0.9, n_acct)),
        "device": RNG.integers(0, n_acct // 3 + 1, n_acct),
        "email": RNG.integers(0, 60, n_acct),
        "bin": RNG.integers(0, 400, n_acct),
    })


def generate(n_acct=9000, days=180, seed=7):
    global RNG
    RNG = np.random.default_rng(seed)
    acc = _base_accounts(n_acct, days)
    rows = []

    for a in acc.itertuples():
        n = max(1, RNG.poisson(9))
        t = np.sort(RNG.uniform(max(a.open_day, 0), days, n)) * DAY
        amt = a.base_amt * np.exp(RNG.normal(0, 0.45, n))
        for ti, ai in zip(t, amt):
            rows.append((a.uid_true, a.card1, a.addr1, a.open_day, a.device,
                         a.email, a.bin, ti, ai, 0))

    # --- account takeover: 1.5% of accounts, burst late in life
    for a in acc.sample(int(n_acct * 0.015), random_state=1).itertuples():
        t0 = RNG.uniform(max(a.open_day, 10), days - 2) * DAY
        for k in range(RNG.integers(3, 9)):
            rows.append((a.uid_true, a.card1, a.addr1, a.open_day,
                         a.device + 9999, a.email, a.bin,
                         t0 + k * RNG.uniform(60, 900),
                         a.base_amt * RNG.uniform(3, 14), 1))

    # --- card testing rings: one device, many fresh cards, tiny amounts
    for ring in range(45):
        dev = 50000 + ring
        t0 = RNG.uniform(5, days - 1) * DAY
        for c in range(RNG.integers(12, 40)):
            rows.append((900000 + ring * 100 + c, RNG.integers(1000, 18000),
                         RNG.integers(100, 540), days - 1, dev,
                         RNG.integers(0, 60), RNG.integers(0, 400),
                         t0 + c * RNG.uniform(5, 90),
                         RNG.uniform(20, 120), 1))

    # --- bust-out: new account, ramp, then one huge hit
    for a in acc[acc.open_day > days - 60].sample(120, random_state=2).itertuples():
        t0 = a.open_day * DAY
        for k in range(6):
            rows.append((a.uid_true, a.card1, a.addr1, a.open_day, a.device,
                         a.email, a.bin, t0 + k * 2 * DAY,
                         200 * (2.4 ** k), int(k >= 4)))

    df = pd.DataFrame(rows, columns=[
        "uid_true", "card1", "addr1", "open_day", "DeviceInfo", "P_emaildomain",
        "card_bin", "TransactionDT", "TransactionAmt", "isFraud"])
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    df["TransactionID"] = np.arange(len(df))
    # D1 = days since account opened, as in IEEE-CIS (with realistic missingness)
    df["D1"] = (df.TransactionDT / DAY).astype(int) - df.open_day
    df.loc[RNG.random(len(df)) < 0.03, "D1"] = np.nan
    df["ProductCD"] = RNG.choice(list("WCHRS"), len(df))
    df["card4"] = RNG.choice(["visa", "mastercard", "rupay", "amex"], len(df),
                             p=[.45, .35, .17, .03])
    return df.drop(columns=["open_day"])


if __name__ == "__main__":
    d = generate()
    d.to_parquet("data/txns.parquet")
    print(len(d), "rows | fraud rate", round(d.isFraud.mean(), 4))
