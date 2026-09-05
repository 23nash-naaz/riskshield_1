# Architectural Decisions & Bug Log

This document tracks significant technical decisions, bugs encountered during development, and the chosen resolutions.

## 2026-09-05: GRU vs Temporal Features
**Context:** We originally explored using a server-side GRU over a 16-transaction window to embed sequence data for better recall.
**Decision:** We dropped the GRU in favor of purely aggregate Welford features (`history.py` and `featurestore.py`). 
**Rationale:** The O(1) time complexity of maintaining rolling aggregates perfectly matches the strict latency bounds of payment scoring, whereas managing sequence tensors online introduces unnecessary state bloat and PyTorch dependencies.

## 2026-09-05: The "Review" Action State
**Context:** High-ticket transactions near the decision boundary were suffering from hard false-positives (BLOCK), destroying LTV.
**Decision:** We introduced a fourth state: `REVIEW`.
**Rationale:** Sending these specific transactions to a human analyst costs ₹50/ticket but preserves the LTV of a legitimate whale user.

## Future Roadmap: Scale Testing
**Milestone:** We intend to test the engine across a much larger window (100K+ txns) once deployed in a shadow mode on live data to validate the O(1) state management under load.
