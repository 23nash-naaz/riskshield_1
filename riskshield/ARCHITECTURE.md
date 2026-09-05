# Architecture

## The Core Pipeline

The architecture guarantees strict low-latency execution and high availability.
- **Client / Checkout:** Sends the transaction data to the `/score` FastAPI endpoint.
- **Online Fast Path:** The API hits our O(1) Feature Store to fetch running aggregates, passes them to the LightGBM Model Artifact, and generates a probability score.
- **Decision Engine:** The score is routed to `economics.py` which translates probabilities into the Rupee-optimal decision across four actions (ALLOW, CHALLENGE, REVIEW, BLOCK).
- **Asynchronous Loop:** Extreme risk cases trigger a background dispute process where the Evidence Responder generates a rebuttal pack.

## Section 1: Solving the "Three Hard Problems" of Fraud

**Problem 1: The 90-Day Label Delay.**
Chargebacks take up to 90 days to arrive. Training a model on yesterday's transactions as 'legitimate' poisons the model. RiskShield enforces a 60-day maturity split and uses Elkan-Noto PU (Positive-Unlabeled) correction to train honestly.

**Problem 2: Training / Serving Skew.**
Offline features use Pandas; online features use real-time state. To prevent drift, our offline metrics perfectly mirror our online `featurestore.py` state.

**Problem 3: O(1) State Management.**
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

## Section 4: Architectural Decisions & Bug Log

RiskShield enforces strict engineering discipline. The math dictates that every technical choice must map to a business outcome. Below are the critical decisions that shaped the final production-ready state:

### GRU vs Temporal Features
**Context:** We originally explored using a server-side GRU over a 16-transaction window to embed sequence data for better recall.
**Decision:** We dropped the GRU in favor of purely aggregate Welford features (`history.py` and `featurestore.py`). 
**Rationale:** The O(1) time complexity of maintaining rolling aggregates perfectly matches the strict latency bounds of payment scoring, whereas managing sequence tensors online introduces unnecessary state bloat and PyTorch dependencies.

### The "Review" Action State
**Context:** High-ticket transactions near the decision boundary were suffering from hard false-positives (BLOCK), destroying LTV.
**Decision:** We introduced a fourth state: `REVIEW`.
**Rationale:** Sending these specific transactions to a human analyst costs ₹50/ticket but preserves the LTV of a legitimate whale user.

### Scale Testing Roadmap
**Milestone:** We intend to test the engine across a much larger window (100K+ txns) once deployed in a shadow mode on live data to validate the O(1) state management under load.
