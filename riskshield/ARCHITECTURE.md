# Architecture

## Section 1: Solving the "Three Hard Problems" of Fraud

**Problem 1: The 90-Day Label Delay**
Chargebacks take up to 90 days to arrive. Training a model on yesterday's transactions as "legitimate" poisons the model. RiskShield enforces a 60-day maturity split and uses Elkan-Noto PU (Positive-Unlabeled) correction to train honestly.

**Problem 2: Training / Serving Skew**
Offline features use Pandas; online features use real-time state. To prevent drift, our offline metrics perfectly mirror our online `featurestore.py` state.

**Problem 3: O(1) State Management**
Graph features break at scale. We decoupled heavy Union-Find logic into an offline batch process, keeping only lightweight degree-counters in the fast online path.

## Section 2: The Degradation Ladder (Crucial)

| Failure Mode | System Behavior | Rationale |
| :--- | :--- | :--- |
| **Feature Store Timeout** | Score on row-only features, flag `degraded=true` | Never hard-fail a checkout. |
| **Model Artifact Corrupted** | Fallback to previous registry version | Maintain decision engine continuity. |
| **API Timeout (>50ms)** | Default action = CHALLENGE (OTP) | Friction over failure. OTP is the cheapest wrong answer. |

**In payments, the most expensive mistakes are silent allows (fraud) and hard blocks (customer insult). Friction is the cheapest failure. When our system is under load, we default to Step-Up authentication.**

## Section 3: Reject Inference & Anti-Drift

A model that blocks traffic goes blind to outcomes. We randomly allow 1% of would-be blocks to pass through (flagged internally) to maintain an unbiased label stream for future retraining.
