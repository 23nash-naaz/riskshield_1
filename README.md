# 🛡️ RiskShield: Rupee-optimal chargeback defence
**Razorpay Buildathon 2026 · Track 02: AI Risk Manager · Defence-only.**

Most fraud projects stop at a score. This one decides what to do with the score, and prices its own mistakes in rupees.

```bash
pip install -r requirements.txt
python run.py # pipeline + ablation + metrics
uvicorn src.api:app --reload # serving
```
*Drop `train_transaction.csv` (IEEE-CIS) into `data/`. Without it, `synth.py` generates a schema-compatible dataset so the pipeline runs end to end.*

---

## The problem, in money

A merchant sells ₹40,000 of goods on a stolen card. The cardholder disputes it. The bank claws back ₹40,000 and adds a ₹1,500 dispute fee. The merchant loses the goods, the money, and the fee.

The obvious fix — block anything suspicious — is worse. Block a real customer on a ₹200 order and you lose roughly ₹580 in margin, friction, and lifetime value. The fraud would have cost ₹17. You lost 34× more by being safe.

There is a dial between losing money to thieves and losing money by insulting customers. RiskShield sets it per transaction, in ₹.

*Why IEEE-CIS: its `isFraud` label literally means a chargeback was filed on this card. That is the exact loss class this track names, not a proxy for it.*

---

## Results

Held-out future window, 12,107 transactions. Numbers below are from `synth.py` — a generator that injects account takeover, card testing, and bust-out patterns. They verify the pipeline; they are not a claim about real-world accuracy. On real IEEE-CIS, expect PR-AUC in the 0.5–0.6 range. The shape of the ablation and the rupee argument hold. The absolute values do not. Saying this is cheaper than being caught not saying it.

| Stage | PR-AUC raw | PR-AUC cal | ECE after cal | ₹ lost / 1k txns |
| :--- | :--- | :--- | :--- | :--- |
| baseline (raw row features) | 0.642 | 0.582 | 0.0089 | 27,664 |
| + entity history | 0.974 | 0.956 | 0.0018 | 3,352 |
| + graph | 0.974 | 0.934 | 0.0053 | 3,110 |
| + sequence embedding | 0.986 | 0.939 | 0.0007 | 1,253 |

At the rupee-optimal operating point: precision 0.954, recall 0.984, 1.01 false positives per 1,000 good customers. 95% bootstrap CI on calibrated PR-AUC: [0.900, 0.971].

### Policies, ranked by what they cost

| Policy | ₹ lost / 1k txns |
| :--- | :--- |
| **RiskShield** | **1,253** |
| step-up everything | 24,225 |
| rule: block top 3% by amount | 57,707 |
| allow everything | 96,657 |
| block everything | 156,990 |

**≈ ₹11.2 lakh saved per ₹1 crore processed, versus doing nothing.**

Note the ordering: *block everything* is the worst policy on the board — worse than allowing all fraud through. Any model that optimises recall without pricing false positives is walking toward that corner.

---

## Five design decisions

**1. Temporal split, never random.** Train on the early window, test on the future one. A random split lets the same account appear on both sides and inflates every number above.

**2. Entity resolution before features.** IEEE-CIS has no customer ID. But D1 is days-since-account-open, so `TransactionDay − D1` is constant per account. Combined with card and address, it reconstructs a pseudo-account.
This one step took PR-AUC from 0.64 to 0.97 — the largest jump in the table, larger than any model change. It converts independent rows into accounts with histories. "₹40,000 is a big transaction" is weak. "₹40,000 is 11× this account's own median" is decisive.

**3. Every history feature is past-only.** All aggregates are `shift(1)` before use. A plain `groupby().transform('mean')` leaks the future and is the most common way fraud models produce fake offline scores.

**4. Delayed labels handled honestly.** Chargebacks land 30–90 days late, so the most recent 60 days are unlabelled, not negative. We exclude them from training — 35,461 rows held out here. Training on them as negatives teaches the model that recent fraud is normal, which is the standard production bug. `labels.py` also carries the Elkan–Noto PU correction for scoring inside the immature window.

**5. Three actions, priced in rupees.** Allow / step-up / block. Step-up (3DS or OTP) converts a false positive from a lost sale into three seconds of friction — that is what makes an aggressive threshold affordable. Most legitimate customers complete the OTP; most fraudsters drop off, because they do not have the phone. Under 3DS liability shift, fraud that authenticates lands on the issuer.

---

## There is no single threshold

The decision is amount-aware. With a fixed ₹1,500 dispute fee, a ₹200 transaction and a ₹80,000 transaction have no business sharing a cutoff — for the small one, blocking costs more than the fraud.

It is also merchant-aware. One global cutoff is wrong for a payment aggregator: a 60%-margin SaaS merchant should block early, a 5%-margin electronics merchant almost never should.

| Merchant margin | step-up above | block above |
| :--- | :--- | :--- |
| 5% | 0.003 | 0.092 |
| 20% | 0.005 | 0.302 |
| 60% | 0.014 | 0.570 |

Derived, not tuned — solved from the cost matrix in `economics.py`. Change one constant and every threshold moves.

---

## Drift audit

Adversarial validation trains a classifier to tell train rows from test rows. Adversarial AUC is 0.989 — the distributions genuinely differ, which is expected on a temporal split of payments data.

We drop a feature only if it drifts hard and carries little signal. Dropping on drift alone throws away real predictive power. Two features removed: `f_dt_ratio`, `h_n_prev`.

---

## Where the model is weakest

Reported because the track asks for honesty, not a highlight reel.

| Slice | n | fraud rate | PR-AUC | recall | FP / 1k |
| :--- | :--- | :--- | :--- | :--- | :--- |
| amount Q3 (mid-value) | 3,026 | 0.4% | 0.737 | 0.818 | 0.33 |
| new accounts | 855 | 14.0% | 0.938 | 0.992 | 9.52 |

Mid-value transactions are the blind spot — too large for the card-testing signature, too small to trip the amount-anomaly features.

New accounts get nearly ten times the false-positive rate of returning ones. Thin history means the model leans on graph and population priors, and it over-flags. For a merchant that is a first-purchase experience problem, not a rounding error.

---

## Training/serving parity

The failure mode that kills fraud systems in production, so we measured it instead of assuming.

Offline features use Pandas expanding windows over full history. Online features come from a store that has only seen what streamed through it. Offline recall is 0.98; online replay recall is ~0.75 early in the stream, because graph state and sequence buffers are still filling.

We report the gap rather than close it quietly. A system whose offline and online numbers agree without anyone checking is a system where nobody checked.

Production fix: log every online feature vector at score time, diff it nightly against the batch recompute, alert per feature on divergence. That is what makes "feature store" a monitored invariant rather than a word in a diagram.

---

## Resilience

The API never hard-fails a payment. Every response carries a degraded flag.

| Failure | Behaviour |
| :--- | :--- |
| Feature store down | Score on row-only features (the 0.64 PR-AUC baseline), flag degraded |
| Model artifact corrupt | Fall back to previous registry version; decision logic unchanged |
| Score timeout (> 50 ms) | Default action = step-up, re-score asynchronously |
| Drift gate trips | Block promotion, keep serving old model, page on-call |

Friction over failure. Step-up as the timeout default falls straight out of the cost matrix: the two expensive mistakes are a silent allow and a hard block. Friction costs seconds. When the system does not know, it makes the customer prove themselves rather than guessing.

---

## Where I chose not to use an LLM

The scorer is gradient-boosted trees, deliberately.
- **Fast** — single-digit milliseconds against a 50 ms budget
- **Free** — no per-call cost at any volume
- **Calibrated** — isotonic regression gives a probability you can multiply by ₹1,500 and get a real number
- **Auditable** — `pred_contrib` returns exact per-feature contributions
- **Deterministic** — same input, same output, which matters when the output becomes a dispute record

An LLM cannot produce a calibrated probability. If p is not a real probability, every rupee downstream is fiction and the premise of this project collapses. Reason codes are deterministic templates for the same reason: VELOCITY_ANOMALY is auditable, generated text is not.

The judgment on display is knowing where AI does not belong on the hot path.

---

## Defence-only compliance

No fraud generator, no card-testing harness, no evasion or bypass analysis.
The model is scoring-only. The API returns an action and a reason code.
Reason codes are coarse buckets (`VELOCITY_ANOMALY`, `SHARED_DEVICE_CLUSTER`) — never a feature value, never a threshold. A leaked API response cannot be reverse-engineered into an evasion recipe. See `decide.py`.

The evidence responder assembles the merchant's own records for a dispute. It fabricates nothing, and recommends accepting the chargeback when the record is weak.

Robustness is evaluated through distribution shift and slice analysis, not adversarial probing.

---

## Honest limitations

- **Synthetic numbers are demo scaffolding**, labelled as such above. Real IEEE-CIS will be lower.
- **Serving parity**. `/score` needs the 32 GRU embedding columns computed from the account's last 16 transactions. A caller passing only raw fields gets a degraded score. Production needs a feature store keyed on uid.
- **Cost constants** in `economics.py` are estimates. They are the model's most important inputs and should be set per merchant from real dispute data.
- **Graph features** are fitted on the train window and mapped forward. Entities first seen at inference get default values.

---

## Files

- `run.py` pipeline + ablation
- `src/synth.py` IEEE-CIS-shaped generator
- `src/entity.py` UID reconstruction
- `src/history.py` leak-safe expanding-window features
- `src/graph.py` shared-identifier ring features
- `src/featurestore.py` online state — the training/serving seam
- `src/labels.py` maturity split + PU correction
- `src/drift.py` adversarial validation
- `src/train.py` LightGBM + isotonic calibration
- `src/economics.py` cost matrix → action ← the core
- `src/decide.py` action + coarse reason codes
- `src/evidence.py` chargeback representment pack
- `src/api.py` FastAPI
- `src/evaluate.py` PR-AUC, CIs, slices, rupees

See **[ARCHITECTURE.md](riskshield/ARCHITECTURE.md)** for the prototype-to-production mapping.

---

## Next

Ordered by expected rupee impact, not by how impressive they sound.

1. **Real IEEE-CIS numbers.** Retrain on the actual dataset and replace the results table. The synthetic run proves the pipeline; only the real one proves the model.
2. **Feature store keyed on uid.** Closes the serving-parity gap and makes `/score` usable with raw fields alone.
3. **A fourth action: review.** Manual review wins at high amounts, where a fixed analyst cost buys certainty on a decision worth far more than the analyst's time. Pure change to `economics.py`, no retraining.
4. **Sensitivity analysis on the cost constants.** Vary each ±50%, rank by rupee swing. Turns the weakest admission in this README into a measured statement about which estimates need to be right — and which of those are observable in production versus which need a deliberate A/B test.
5. **Reject inference.** Blocked transactions never produce labels, so every retrain is biased toward what the previous model already approved. Log 1% of would-be blocks as allowed to keep an unbiased label stream.
6. **Cascading dispute penalties.** Visa VAMP and Mastercard ECP are step functions — the chargeback that pushes the 30-day ratio past 1% costs 10–50× a normal one. A fee that is convex in the rolling ratio would make the system tighten automatically near the cliff.
