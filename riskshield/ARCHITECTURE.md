# Architecture

```mermaid
graph TD
    A[Client / Checkout] -->|Transaction| B(FastAPI Endpoint: /score)
    
    subgraph Online Fast Path
        B -->|Fetch/Update O(1)| C[(Feature Store: Redis/dict)]
        B -->|Row + Aggregates| D[LightGBM Model Artifact]
        D -->|Probability| E{Decision Engine: economics.py}
    end

    subgraph 4-Action Output
        E -->|Low Risk| F(ALLOW)
        E -->|Mid Risk| G(CHALLENGE: OTP)
        E -->|High Risk| H(REVIEW: Analyst)
        E -->|Extreme Risk| I(BLOCK)
    end
    
    subgraph Asynchronous Loop
        I -->|Dispute Logged| J[Evidence Responder]
        J -->|LLM + Rules| K(Rebuttal Pack JSON)
    end
```
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

## Section 4: Where I Chose Not to Use an LLM

While LLMs are excellent for unstructured tasks, placing them in the critical path of a transaction scoring engine introduces unacceptable latency, hallucination risk, and unit cost. We strictly reserved generative AI for the asynchronous **Evidence Responder** (dispute management), keeping the core `/score` endpoint driven by deterministic math, LightGBM, and O(1) state lookups.

## Section 5: Closing the Recall Gap (Offline vs. Online)

Our baseline model showed a pristine 0.98 recall offline, which plummeted to 0.75 when subjected to online temporal splits. This is the **Training/Serving Skew**. To close this gap, our `featurestore.py` perfectly mirrors the exact Welford variance equations and sliding-window logic used during offline training (`history.py`), guaranteeing that a feature computed on Tuesday looks mathematically identical to the same feature computed during Friday's batch job.

## Section 6: Learnable vs. Unlearnable Cost Constants

We decompose our margin economics into two strict categories:
- **Learnable Constants**: Base transaction rates, known processing fees, and chargeback penalties. These are deterministic and hardcoded into `economics.py`.
- **Unlearnable / Stochastic Constants**: LTV destruction, friction bounce rates, and analyst time (₹50 per REVIEW). These are treated as hyper-parameters and exposed to the merchant via the UI, allowing them to drag a slider and see the profit margin update dynamically.
