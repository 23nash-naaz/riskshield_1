# IMPLEMENTATION PLAN — RiskShield
Step-by-step build order, what each step uses, and why that choice over the alternatives.

The build order IS the priority order: each step multiplies the value of everything after it.
Stop-loss rules included, because the discipline of skipping a stage is worth more than the stage.

---

## STEP 0 — Frame the problem as decisions, not detection
**Build:** nothing yet. Write the cost model on paper: what does each mistake cost in rupees?
- Missed fraud = transaction amount + Rs 1,500 dispute fee
- False block = lost sale x merchant margin
- Step-up (OTP/3DS) = small friction cost, stops ~90% of fraud

**Why first:** every later choice (metric, threshold, even model) derives from this.
Teams that skip it end up optimising F1, which prices a Rs 200 and a Rs 80,000
mistake identically.

---

## STEP 1 — Data: IEEE-CIS Fraud Detection (Kaggle)
**Use:** `train_transaction.csv`, 590k card transactions.
**Why this dataset:** its `isFraud` label literally means "a chargeback was filed"
— the exact loss class the track names, not a proxy. It also carries device /
email / address identifiers, which makes entity and ring features possible.
**Why not PaySim:** synthetic mobile-money data, no chargeback semantics, no identifiers.
**Fallback built:** a schema-compatible synthetic generator (`synth.py`) with three
injected fraud mechanisms (account takeover, card testing, bust-out) so the whole
pipeline runs and is testable without Kaggle access.

## STEP 2 — Temporal split (never random)
**Use:** sort by `TransactionDT`; train on the early window, validate next, test on the future.
**Why:** a random split puts the same card on both sides — the model memorises
identities and the offline score is a lie. Fraud models are always deployed on
the future; the test set must be the future.
**Tool:** plain pandas. No library needed; the discipline is the tool.

## STEP 3 — Entity resolution (the single biggest lift)
**Use:** reconstruct a pseudo-account UID = card1 + addr1 + (TransactionDay − D1).
D1 is days-since-account-open, so TransactionDay − D1 is constant per account.
**Why:** fraud is a property of an account over time, not of a row. This converts
590k independent rows into ~200k accounts with histories. Measured impact:
PR-AUC 0.64 → 0.97, more than every other feature combined.
**Why not embeddings/clustering for identity:** the D1 invariant is exact, free,
and explainable; learned identity is approximate and a hackathon time sink.

## STEP 4 — Leak-safe history features
**Use:** per-UID expanding windows, always `shift(1)` (strictly past-only):
amount z-score vs the account's own history, ratio to its own max, inter-arrival
velocity, txns in last 1h/24h.
**Why:** "Rs 40,000" is weak evidence; "11x this account's own median" is decisive.
**Why shift(1) is non-negotiable:** a plain `groupby().transform('mean')` includes
the future — the most common way fraud models fake their offline scores.
**Tool:** pandas expanding windows + one numpy `searchsorted` for time-window counts
(vectorised; a Python loop over 590k rows would take minutes).

## STEP 5 — Graph features (abuse-ring signal)
**Use:** bipartite graph account ↔ {device, email domain, card BIN}; per txn take
node degree, connected-component size, devices-per-account ratio. Fitted on the
TRAIN window only, mapped forward.
**Why:** one device touching 30 cards is the card-testing signature; no per-row
feature can see it. Covers the track's "abuse-ring sentinel" direction as a
feature, where its value is measurable, instead of as a whole project, where
ring precision/recall is unmeasurable (rings aren't labelled).
**Why networkx, not a GNN:** three cheap numbers capture most of the signal;
a GNN costs ~6 hours of sampling/leakage debugging for maybe +1% PR-AUC.

## STEP 6 — Delayed-label handling (the production step)
**Use:** exclude the most recent 60 days from training (labels not yet mature);
Elkan–Noto PU correction for scoring inside the immature window.
**Why:** chargebacks arrive 30–90 days late. Training on recent txns as
"not fraud" teaches the model that new fraud is normal — the standard
production bug. This step barely moves offline metrics; it makes them TRUE.

## STEP 7 — Model: LightGBM + isotonic calibration
**Use:** LightGBM (gradient-boosted trees) with early stopping on the validation
window; isotonic regression fitted on validation to calibrate scores.
**Why LightGBM:** still the strongest learner on heterogeneous tabular data
(tested against deep tabular models like TabNet/FT-Transformer, which
consistently lose here), trains in seconds, gives SHAP contributions for
reason codes.
**Why calibration is mandatory, not optional:** the decision layer computes
expected rupees = p x cost. That math is only valid if p is a real probability.
Isotonic over Platt because the score distribution is heavily skewed.
**Honest cost discovered:** isotonic's step function trades ~0.05 ranking AP
for valid probabilities — we report both numbers.

## STEP 8 — Sequence encoder (optional, timeboxed)
**Use:** small GRU over each account's last 16 transactions → 32-d embedding,
fed INTO LightGBM as extra columns.
**Why hybrid, not end-to-end deep:** GBDT stays better on the tabular part;
the GRU contributes the one thing trees can't see — temporal shape (ramping,
bursting, drift from baseline). This is the architecture Stripe/Feedzai
converged on. Measured: +0.01 PR-AUC and −60% rupee cost at the operating point.
**Stop-loss:** if it doesn't beat the previous stage in 5 hours, ship without it.
The code auto-skips this stage if torch isn't installed.

## STEP 9 — Drift audit (adversarial validation)
**Use:** train a classifier to distinguish train rows from test rows; features
it relies on are the ones that will decay in production.
**Refinement that mattered:** drop only features with HIGH drift AND LOW
predictive signal — dropping on drift alone destroyed good features. Went from
dropping 8 features to 2, no performance loss.

## STEP 10 — Economics layer (the differentiator)
**Use:** ~80 lines (`economics.py`): expected-rupee cost per action, argmin over
{allow, step-up, block}, amount-aware and margin-aware.
**Why three actions:** step-up converts a false positive from "lost sale" into
"3 seconds of friction" — the realistic production answer that makes aggressive
thresholds affordable.
**Why per-merchant thresholds:** a 5%-margin electronics merchant and a
60%-margin SaaS merchant need different cut-offs; for a payment aggregator one
global threshold is wrong by construction. Thresholds are SOLVED from the cost
matrix, not tuned.
**Hard rule enforced:** model tuning stopped when this layer started. The last
3% of AUC is worth nothing next to the decision layer.

## STEP 11 — Honest evaluation
**Use:** PR-AUC (not ROC — at a 3% base rate ROC flatters), bootstrap 95% CIs,
expected-calibration-error, FP per 1,000 good customers, rupee comparison vs
four baseline policies, per-segment slice table INCLUDING the worst slices,
ablation table (each stage's contribution), model card with failure modes.
**Why publish weaknesses:** the track's bar is honesty; the mid-value blind spot
(PR-AUC 0.74) and new-account over-flagging (9.5 FP/1k) are real costs, and
reporting them is the differentiator, not a confession.

## STEP 12 — Online feature store (script → product)
**Use:** `featurestore.py` — per-account running state with Welford mean/variance,
time-window velocity counters, union-find for graph components. The API accepts
a RAW transaction and computes training-identical features live, ~4 ms/txn.
**Why:** this closes training/serving skew, the gap most teams never measure.
We measured it anyway: offline recall 0.98 vs 0.75 through the online path —
shipped as a finding with the monitoring design to close it.
**Why in-memory dict now, Redis later:** identical interface (`features()/update()`);
the architecture doc maps the swap. Cold-start solved the production way:
startup replays the pre-test event stream (9,267 accounts), like a store
rebuilding from its event log.

## STEP 13 — Platform: FastAPI + live console
**Use:** FastAPI serving a single-file vanilla-JS dashboard: live decision tape
over held-out transactions, margin slider that moves the allow/step-up/block
threshold bands in real time, running rupee-saved counter, true chargebacks
revealed only AFTER each decision.
**Why FastAPI:** async, typed via pydantic, auto OpenAPI docs, one file.
**Why vanilla JS, no React/build step:** one HTML file, zero npm, nothing to
break during a demo; the value is the decision engine, not the frontend stack.
**Why the slider is the demo centrepiece:** it makes the per-merchant economics
argument physical — the judge watches thresholds move as margin changes.

## STEP 14 — Evidence responder (closes the loop)
**Use:** `evidence.py` — maps Visa/Mastercard dispute reason codes to the
signals that actually rebut them (unauthorised-use ⇒ device continuity + prior
good history; not-received ⇒ delivery proof), scores evidence strength, and
recommends ACCEPTING the chargeback when the record is weak.
**Why:** prevention + representment = the full chargeback loss loop; a
responder that knows when NOT to fight is the honest version of this feature.

## STEP 15 — Defense-only guardrails
**Use:** no fraud generation or evasion tooling anywhere; API reason codes are
coarse buckets (VELOCITY_ANOMALY, SHARED_DEVICE_CLUSTER), never feature values
or thresholds — a leaked response can't be reverse-engineered into an evasion
recipe.
**Why designed-in, not stated:** the track disqualifies offense capability;
showing adversarial-ML awareness in the API contract is stronger than a
compliance sentence.

---

## Tool summary

| Tool | Used for | Why over alternatives |
|---|---|---|
| pandas + numpy | features, splits | vectorised windows; the leak-safety lives in HOW they're used |
| LightGBM | classifier | beats deep tabular models here; fast; SHAP for reason codes |
| scikit-learn | isotonic calibration, metrics | standard, auditable |
| PyTorch (optional) | GRU encoder | temporal shape trees can't see; auto-skipped if absent |
| networkx | ring features | 3 numbers ≈ most of a GNN's value at 1% of the cost |
| FastAPI + uvicorn | platform API | typed, async, self-documenting |
| Vanilla HTML/JS | console | zero build step, demo can't break on tooling |
| in-mem store → Redis (design) | online features | same interface now and at scale |

## Timeboxes and stop-losses (what made it finish)
- Model tuning HARD-STOPPED when the economics layer began (step 10).
- GRU had a 5-hour stop-loss and an auto-skip path.
- GNN, tabular transformers, autoencoders: evaluated and rejected up front
  (cost/benefit noted in README) instead of half-built.
