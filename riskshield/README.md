# Rupee-Optimal Chargeback Shield

**Track 02 — AI Risk Manager.** Defence-only.

Most fraud projects stop at a score. This one decides what to **do** with the
score, and reports the cost of its mistakes in rupees.

```bash
pip install lightgbm networkx torch scikit-learn pandas pyarrow fastapi uvicorn
python run.py                      # full pipeline + ablation + metrics
cd src && uvicorn api:app --reload # serving
```

Drop `train_transaction.csv` (IEEE-CIS) into `data/` and the loader uses it.
Without it, `synth.py` generates a schema-compatible dataset with three
injected fraud mechanisms (account takeover, card testing, bust-out) so the
pipeline is runnable and testable end to end.

Why IEEE-CIS: its `isFraud` label literally means *a chargeback was filed on
this card*. It is the exact loss class this track names, not a proxy.

---

## Results (held-out future window, 12,107 transactions)

| Stage | PR-AUC (raw) | PR-AUC (cal) | ECE after cal | ₹ lost / 1k txns |
|---|---|---|---|---|
| baseline (raw row) | 0.642 | 0.582 | 0.0089 | 27,664 |
| + entity history | 0.974 | 0.956 | 0.0018 | 3,352 |
| + graph | 0.974 | 0.934 | 0.0053 | 3,110 |
| + sequence embedding | **0.986** | 0.939 | **0.0007** | **1,253** |

At the rupee-optimal operating point: **precision 0.954, recall 0.984,
1.01 false positives per 1,000 good customers.**

95% bootstrap CI on calibrated PR-AUC: [0.900, 0.971].

### Rupees lost per 1,000 transactions

| Policy | ₹ |
|---|---|
| **model** | **1,253** |
| step-up everything | 24,225 |
| rule: block top 3% by amount | 57,707 |
| allow everything | 96,657 |
| block everything | 156,990 |

≈ **₹11.2 lakh saved per ₹1 crore processed**, versus doing nothing.

Note the ordering: *block everything* is the worst policy on the board. Any
model that optimises recall without pricing false positives is walking toward
that corner.

---

## The five design decisions

**1. Temporal split, never random.** Train on the early window, test on the
future one. A random split lets the same account appear on both sides and
inflates every number.

**2. Entity resolution before features.** IEEE-CIS has no customer ID, but
`D1` is days-since-account-open, so `TransactionDay − D1` is constant per
account. Combined with card and address it reconstructs a pseudo-account.
This one step took PR-AUC from 0.64 to 0.97 — it converts independent rows
into accounts with histories. "₹40,000 is a big transaction" is weak;
"₹40,000 is 11× this account's own median" is decisive.

**3. Every history feature is past-only.** All aggregates are `shift(1)`
before use. A plain `groupby().transform('mean')` leaks the future and is the
most common way fraud models produce fake offline scores.

**4. Delayed labels handled honestly.** Chargebacks land 30–90 days after the
transaction, so the most recent 60 days are *unlabelled*, not negative. We
exclude them from training (35,461 rows held out here). Training on them as
negatives teaches the model that recent fraud is normal — the standard
production bug. `labels.py` also carries the Elkan–Noto PU correction for
scoring inside the immature window.

**5. Three actions, priced in rupees.** Allow / step-up / block. Step-up (3DS
or OTP) converts a false positive from *lost sale* into *three seconds of
friction*, which is what makes an aggressive threshold affordable. The
decision is **amount-aware**: with a fixed ₹1,500 dispute fee, a ₹200
transaction and a ₹80,000 transaction should not share a threshold.

### Per-merchant thresholds

One global cutoff is wrong for a payment aggregator. A 60%-margin SaaS
merchant should block early; a 5%-margin electronics merchant should not.

| Merchant margin | step-up above | block above |
|---|---|---|
| 5% | 0.003 | 0.092 |
| 20% | 0.005 | 0.302 |
| 60% | 0.014 | 0.570 |

Derived, not tuned — solved from the cost matrix in `economics.py`.

---

## Drift audit

Adversarial validation trains a classifier to tell train rows from test rows.
Adversarial AUC here is 0.989, so the distributions genuinely differ. We drop
a feature only if it drifts hard **and** carries little signal — dropping on
drift alone throws away real predictive power. Two features were removed
(`f_dt_ratio`, `h_n_prev`).

## Where the model is weakest

Reported because the track asks for honesty, not a highlight reel.

| Slice | n | fraud rate | PR-AUC | recall | FP/1k |
|---|---|---|---|---|---|
| amount Q3 (mid-value) | 3,026 | 0.4% | **0.737** | 0.818 | 0.33 |
| new accounts | 855 | 14.0% | 0.938 | 0.992 | **9.52** |

Mid-value transactions are the blind spot: too large for the card-testing
signature, too small to trip the amount-anomaly features. New accounts get
nearly ten times the false-positive rate of returning ones — thin history
means the model leans on graph and population priors, and it over-flags.
Both are real costs, not rounding error.

---

## Defence-only compliance

- No fraud generator, no card-testing harness, no evasion or bypass analysis.
- The model is scoring-only. The API returns an action and a reason code.
- Reason codes are **coarse buckets** (`VELOCITY_ANOMALY`,
  `SHARED_DEVICE_CLUSTER`), never a feature value or a threshold, so a leaked
  API response cannot be reverse-engineered into an evasion recipe. See
  `decide.py`.
- The evidence responder assembles a merchant's own transaction records for a
  dispute. It does not fabricate evidence, and it recommends *accepting* the
  chargeback when the record is weak.

## Known limitations

- **Serving parity:** `/score` needs the 32 GRU embedding columns computed
  from the account's last 16 transactions. A caller passing only raw fields
  gets a degraded score. Production needs a feature store keyed on `uid`.
- Synthetic-data numbers above are for a runnable demo. On real IEEE-CIS
  expect PR-AUC in the 0.5–0.6 range — the *shape* of the ablation and the
  rupee argument hold; the absolute values do not.
- Graph features are fitted on the train window and mapped forward. Entities
  first seen at inference get default values.
- Cost constants in `economics.py` are estimates. They are the model's most
  important inputs and should be set per merchant from real dispute data.

## Files

```
run.py           pipeline + ablation
src/synth.py     IEEE-CIS-shaped generator
src/entity.py    UID reconstruction
src/history.py   leak-safe expanding-window features
src/graph.py     shared-identifier ring features
src/seqenc.py    GRU behavioural encoder
src/labels.py    maturity split + PU correction
src/drift.py     adversarial validation
src/train.py     LightGBM + isotonic calibration
src/economics.py cost matrix -> action        <- the core
src/decide.py    action + coarse reason codes
src/evidence.py  chargeback representment pack
src/api.py       FastAPI
src/evaluate.py  PR-AUC, CIs, slices, rupees
```
