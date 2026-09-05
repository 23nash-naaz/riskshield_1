# 🛡️ RiskShield: Enterprise-Grade Fraud & Dispute Engine
**Razorpay Buildathon Track 02: AI Risk Manager**

Most models optimize for F1 scores and fail in production. RiskShield optimizes for Merchant Margin and System Uptime.

## Section 1: The Rupee Impact (Put the money first)

Fraud is a pricing problem. We built a 4-Action economic engine that calculates the exact cost of a false positive (churn) versus a false negative (chargeback fee) in real-time.

| Action | Scenario | Business Impact / Cost |
| :--- | :--- | :--- |
| **Baseline Model** | Optimizes for F1 only | High friction, lost LTV, silent margin drain |
| **RiskShield** | Optimizes for Merchant Margin | Minimized friction, precise cost-benefit calculation per transaction |

## Section 2: The 4-Action Cost Matrix

| Action | Condition | Cost / Friction | Result |
| :--- | :--- | :--- | :--- |
| **ALLOW** | Low risk | ₹0 cost if legitimate | Frictionless checkout |
| **CHALLENGE (OTP)** | Mid risk | ₹15 friction | Drops 95% of fraudsters |
| **REVIEW** | High risk / High ticket | ₹50 analyst time | Human-in-the-loop precision |
| **BLOCK** | Extreme risk | Margin loss + Lifetime Value (LTV) destruction | Hard stop on confirmed threats |

## Section 3: The Evidence Responder (Your Secret Weapon)

Track 02 asked for an evidence responder. While most teams stopped at a fraud score, RiskShield automates the post-fraud dispute lifecycle.

Our `evidence.py` engine calculates the ROI of fighting a dispute and auto-generates a `rebuttal_pack.json` for winning cases.

## Section 4: The Hiring Hook (Production Migration Plan)

RiskShield was built as a scaled-down production system. To migrate this prototype to Razorpay's infrastructure, we execute three strict swaps without touching the API contract:

1. Swap `featurestore.py` dicts with Redis / Flink.
2. Swap `warmup.json` logs with a live Kafka topic.
3. Retain the model artifact and `economics.py` exactly as they are.
