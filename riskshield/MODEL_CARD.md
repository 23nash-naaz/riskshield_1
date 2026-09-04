# Model Card — Rupee-Optimal Chargeback Shield

**Task:** predict whether a card transaction will result in a chargeback, and
choose the rupee-minimising action (allow / step-up / block).

**Not for:** identity verification, credit decisions, account closure, law
enforcement referral, or any adverse action against a person without human
review. A blocked transaction is a friction event, not an accusation.

## Data
IEEE-CIS Fraud Detection schema. Label = chargeback filed on the card.
Temporal split; the most recent 60 days are excluded from training as
label-immature. Synthetic fallback for reproducibility without Kaggle access.

## Metrics (held-out future window)
PR-AUC 0.986 raw / 0.939 calibrated (95% CI 0.900–0.971).
Precision 0.954, recall 0.984, 1.01 FP per 1,000 good customers.
ECE 0.0007 after isotonic calibration.
ROC-AUC deliberately not headlined: at a ~3% base rate it flatters.

## Known failure modes
- **Mid-value transactions** (PR-AUC 0.737) — the weakest segment.
- **New accounts** — 9.5 FP per 1,000 vs 0.5 for returning. Thin history
  causes over-flagging. Consider routing new accounts to step-up by default
  rather than block.
- **Cold-start entities** unseen in the train graph get default features.
- **Concept drift** — adversarial AUC 0.989 means distributions shift.
  Retrain monthly; monitor the ₹-per-1k metric, not just PR-AUC.

## Fairness
No protected attributes are used as features. Slice metrics are reported by
card network, amount band, account tenure, and hour. Account tenure shows a
material disparity (new accounts over-flagged) and is the fairness issue that
needs an owner before deployment.

## Cost assumptions
Chargeback fee ₹1,500; merchant margin 20% default (per-merchant override);
3DS abandonment 8%; 3DS fraud-stop rate 90%. These are estimates and are the
single most influential input to every decision the system makes.

## Human oversight
Blocks should be appealable. Step-up is preferred over block wherever the
expected costs are close. Reason codes are coarse by design and are not an
explanation of a decision to the cardholder.
