# ARCHITECTURE — from prototype to production

The prototype is a deliberate miniature of a production risk platform. Every
box below exists in the repo in its simplest correct form; the right column is
the scale-out swap. Interfaces stay identical, which is the point.

```
                         PRODUCTION TOPOLOGY
                         ===================

  merchant checkout
        |
        v            p99 budget: 50 ms end-to-end
  +-----------+     +------------------+     +-----------------+
  | API GW /  | --> |  RISK SCORER     | --> | DECISION ENGINE |---> allow
  | auth      |     |  (stateless,     |     | (cost matrix,   |---> step-up (3DS)
  +-----------+     |   N replicas)    |     |  per-merchant   |---> block
                    +--------+---------+     |  margins)       |
                             |               +--------+--------+
                    reads    |                        |  writes decision
                             v                        v
                    +------------------+     +-----------------+
                    | ONLINE FEATURE   |     | EVENT LOG       |
                    | STORE (Redis)    |<----| (Kafka)         |
                    | per-uid state,   |     | txns, outcomes, |
                    | graph counters   |     | disputes        |
                    +------------------+     +--------+--------+
                                                      |
                             +------------------------+------------+
                             v                                     v
                    +------------------+                  +-----------------+
                    | STREAM UPDATER   |                  | OFFLINE (batch) |
                    | (Flink/consumer) |                  | feature build,  |
                    | folds each txn   |                  | training, calib,|
                    | into store state |                  | drift audit     |
                    +------------------+                  +--------+--------+
                                                                   |
                    +------------------+                  registry | monthly
                    | CHARGEBACK       |                           v
                    | WEBHOOK (30-90d) |---> labels ---> +-----------------+
                    | + evidence pack  |                 | MODEL REGISTRY  |
                    | generator        |                 | shadow -> canary|
                    +------------------+                 +-----------------+
```

## Prototype -> production mapping

| Concern | In this repo | Production swap | Why the interface survives |
|---|---|---|---|
| Online features | `featurestore.py` in-proc dict, Welford, union-find | Redis hashes + HyperLogLog counters; Flink folds the stream | `features()/update()` contract unchanged |
| Event stream | `warmup.json` + `replay.json` replayed in order | Kafka topics `txns`, `outcomes` | warmup at boot == log replay, same semantics |
| Scoring | LightGBM + isotonic in one process, ~4 ms/txn | same artifact behind N stateless replicas + LB | model is a pure function; state lives in the store |
| Decisions | `economics.py` cost matrix, margin per request | merchant config service; thresholds derived, not tuned | `decide(p, amount, cost)` unchanged |
| Labels | 60-day maturity split + PU correction | chargeback webhooks joined by txn id, same maturity rule | `labels.py` logic is the joiner |
| Retraining | `run.py` manual | scheduled batch; adversarial-validation gate (`drift.py`) blocks promotion; shadow -> 5% canary -> full | ablation table becomes the promotion report |
| Ring detection | union-find + degree counters | nightly full graph job (Spark GraphFrames) writes component ids back to Redis | online counters are the fast approximation |
| Evidence | `evidence.py` templated pack | queue consumer on dispute webhook; human review for MODERATE, auto-file STRONG | rebuttal map is the product |
| Serving UI | `dashboard.html` + FastAPI | merchant dashboard tab in the PA console | endpoints are the contract |

## The three hard problems, and where they're handled

**1. Training/serving skew.** Offline features are pandas expanding windows;
online features are the store. These WILL diverge unless tested. The repo
already shows the honest gap: offline recall 0.98, online replay recall ~0.75
early in the stream (graph state and sequence buffers differ from the batch
view). Production fix: log online feature vectors, diff nightly against the
batch recompute, alert on drift per feature. This gap is why "we have a
feature store" is a real engineering claim and not a slide.

**2. Delayed labels.** A chargeback arrives 30–90 days late. The label joiner
never marks a fresh transaction negative — it is unlabelled (`labels.py`).
Monitoring uses proxy metrics until maturity: 3DS failure rate on step-ups,
issuer declines, early-dispute rate.

**3. State at scale.** Per-uid state is O(active accounts): ~200 bytes of
aggregates + 16x7 floats of sequence buffer + graph counters ≈ under 1 KB per
account. 50M accounts ≈ 50 GB — one Redis cluster. Union-find doesn't shard
naively; production uses the nightly graph job for exact components and keeps
only degree counters online. Both are already separated in `featurestore.py`.

## Failure modes and degradation ladder

| Failure | Behaviour |
|---|---|
| Feature store down | score on row-only features (the 0.64 PR-AUC baseline), flag `degraded=true`, never hard-fail the payment |
| Model artifact bad | fall back to previous registry version; decisions engine unchanged |
| Score timeout > 50 ms | default action = step-up (cheapest wrong answer), async re-score |
| Drift gate trips | block promotion, keep serving old model, page the owner |

Step-up as the timeout default is the single most important line in the
ladder: the cost-matrix says the expensive mistakes are silent allows and
hard blocks; friction is the cheap failure.

## Non-goals (deliberate)

Device fingerprinting SDKs, consortium data sharing, rules-engine DSL, and
real-time graph neural networks are all out of scope. Each is a product in
itself; the architecture leaves a seam for them (extra features into the same
scorer) without depending on any.
