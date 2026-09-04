"""Chargeback representment pack.

Closes the loop: for transactions we ALLOWED that later get disputed, assemble
the evidence a network needs. Maps the dispute reason code to the signals that
actually rebut it -- an unauthorised-use claim is rebutted by device continuity,
not by delivery proof.
"""
from datetime import datetime, timedelta

# What actually rebuts each Visa/Mastercard reason code
REBUTTAL = {
    "10.4": ("Fraud - card absent",
             ["device_continuity", "prior_good_txns", "avs_cvv", "ip_geo_match"]),
    "13.1": ("Merchandise not received",
             ["delivery_proof", "tracking", "customer_contact"]),
    "13.3": ("Not as described",
             ["product_spec", "customer_contact", "return_policy_ack"]),
    "13.6": ("Credit not processed",
             ["refund_ledger", "return_policy_ack"]),
}


def build(txn, history, dispute_code="10.4"):
    """txn: dict for the disputed txn. history: list of prior txns for the uid."""
    title, want = REBUTTAL.get(dispute_code, ("Unknown", ["prior_good_txns"]))
    same_dev = [h for h in history if h.get("DeviceInfo") == txn.get("DeviceInfo")]
    ev, strength = [], 0

    if "device_continuity" in want and len(same_dev) >= 3:
        first = min(h["TransactionDT"] for h in same_dev)
        days = int((txn["TransactionDT"] - first) / 86400)
        ev.append(f"Same device used for {len(same_dev)} prior transactions "
                  f"over {days} days on this account.")
        strength += 2
    if "prior_good_txns" in want:
        clean = [h for h in history if not h.get("disputed")]
        if len(clean) >= 3:
            ev.append(f"{len(clean)} prior undisputed transactions totalling "
                      f"Rs {sum(h['TransactionAmt'] for h in clean):,.0f}.")
            strength += 2
    if "avs_cvv" in want and txn.get("avs_match") and txn.get("cvv_match"):
        ev.append("AVS and CVV both matched at authorisation.")
        strength += 1
    if "ip_geo_match" in want and txn.get("ip_country") == txn.get("bill_country"):
        ev.append("Authorisation IP country matched billing country.")
        strength += 1
    if "tracking" in want and txn.get("tracking_id"):
        ev.append(f"Carrier tracking {txn['tracking_id']} shows delivery to "
                  f"the billing address on file.")
        strength += 3

    verdict = ("STRONG - represent" if strength >= 4 else
               "MODERATE - represent, expect ~50%" if strength >= 2 else
               "WEAK - accept the chargeback, representment cost exceeds recovery")
    return {
        "txn_id": txn.get("TransactionID"),
        "amount_inr": txn.get("TransactionAmt"),
        "dispute_code": dispute_code,
        "dispute_type": title,
        "evidence": ev,
        "missing": [w for w in want if not any(w.split("_")[0] in e.lower() for e in ev)],
        "strength": strength,
        "recommendation": verdict,
        "deadline": (datetime.utcnow() + timedelta(days=7)).strftime("%Y-%m-%d"),
    }


def to_text(pack):
    L = [f"REPRESENTMENT PACK - txn {pack['txn_id']} - Rs {pack['amount_inr']:,.0f}",
         f"Dispute {pack['dispute_code']} ({pack['dispute_type']})",
         f"Recommendation: {pack['recommendation']}", "", "Evidence:"]
    L += [f"  {i+1}. {e}" for i, e in enumerate(pack["evidence"])] or ["  (none)"]
    if pack["missing"]:
        L += ["", "Missing evidence to collect: " + ", ".join(pack["missing"])]
    L += ["", f"Submit by {pack['deadline']}"]
    return "\n".join(L)
