"""Abuse-ring sentinel.

Extracts and ranks connected components from the shared-identifier graph.
A ring is suspicious when many UIDs share a small number of devices or email
domains — classic card-testing or refund-abuse topology.

Ranking: component_size × fraud_rate × device_concentration.
Output: structured alerts with severity for the merchant dashboard.
"""
import numpy as np
import pandas as pd
import networkx as nx

IDS = ["DeviceInfo", "P_emaildomain", "card_bin"]

SEVERITY_THRESHOLDS = {
    "CRITICAL": 0.70,   # score >= 0.70
    "HIGH":     0.40,
    "MEDIUM":   0.15,
}


def extract_rings(df, gf, min_size=3, top_k=10):
    """Find suspicious connected components from the fitted graph.

    Args:
        df: DataFrame with uid, DeviceInfo, P_emaildomain, card_bin, isFraud
        gf: graph-fit dict from graph.fit_graph()
        min_size: minimum component UIDs to qualify
        top_k: number of top rings to return

    Returns:
        list of ring dicts sorted by suspiciousness score
    """
    # Reconstruct the graph from gf's component mapping
    G = nx.Graph()
    for col in IDS:
        e = df[["uid", col]].dropna().drop_duplicates()
        G.add_edges_from(zip("u:" + e.uid.astype(str),
                             f"{col}:" + e[col].astype(str)))

    rings = []
    for i, component in enumerate(nx.connected_components(G)):
        uid_nodes = [n for n in component if n.startswith("u:")]
        if len(uid_nodes) < min_size:
            continue

        uids = [n[2:] for n in uid_nodes]
        dev_nodes = [n for n in component if n.startswith("DeviceInfo:")]
        email_nodes = [n for n in component if n.startswith("P_emaildomain:")]
        bin_nodes = [n for n in component if n.startswith("card_bin:")]

        # Fraud rate within this ring
        mask = df.uid.isin(uids)
        ring_df = df[mask]
        if len(ring_df) == 0:
            continue

        n_txns = len(ring_df)
        n_fraud = int(ring_df.isFraud.sum())
        fraud_rate = n_fraud / max(n_txns, 1)

        # Device concentration: few devices shared by many accounts = suspicious
        n_uids = len(uid_nodes)
        n_devices = max(len(dev_nodes), 1)
        device_concentration = n_uids / n_devices

        # Suspiciousness score: size × fraud_rate × device_concentration
        # Normalised to [0, 1] later
        raw_score = (np.log1p(n_uids) * (fraud_rate + 0.01)
                     * np.log1p(device_concentration))

        total_amount = float(ring_df.TransactionAmt.sum())
        avg_amount = float(ring_df.TransactionAmt.mean())

        rings.append({
            "ring_id": i,
            "n_accounts": n_uids,
            "n_transactions": n_txns,
            "n_fraud": n_fraud,
            "fraud_rate": round(fraud_rate, 4),
            "n_devices": len(dev_nodes),
            "n_emails": len(email_nodes),
            "n_bins": len(bin_nodes),
            "device_concentration": round(device_concentration, 2),
            "total_amount_inr": round(total_amount, 0),
            "avg_amount_inr": round(avg_amount, 0),
            "raw_score": raw_score,
            "shared_devices": [n.split(":", 1)[1] for n in dev_nodes[:5]],
            "shared_emails": [n.split(":", 1)[1] for n in email_nodes[:5]],
            "sample_uids": uids[:8],
            "edges": _ring_edges(G, component, max_edges=30),
        })

    if not rings:
        return []

    # Normalise scores to [0, 1]
    max_score = max(r["raw_score"] for r in rings) or 1.0
    for r in rings:
        r["score"] = round(r["raw_score"] / max_score, 4)
        r["severity"] = _severity(r["score"])
        del r["raw_score"]

    rings.sort(key=lambda r: r["score"], reverse=True)
    return rings[:top_k]


def _severity(score):
    for level, thresh in SEVERITY_THRESHOLDS.items():
        if score >= thresh:
            return level
    return "LOW"


def _ring_edges(G, component, max_edges=30):
    """Extract edges for SVG visualization. Returns list of {source, target, type}."""
    edges = []
    sub = G.subgraph(component)
    for u, v in list(sub.edges())[:max_edges]:
        u_type = "account" if u.startswith("u:") else u.split(":")[0]
        v_type = "account" if v.startswith("u:") else v.split(":")[0]
        edges.append({
            "source": u.split(":", 1)[1][:12],
            "target": v.split(":", 1)[1][:12],
            "source_type": u_type,
            "target_type": v_type,
        })
    return edges


def summary_table(rings):
    """DataFrame for printing."""
    if not rings:
        return pd.DataFrame()
    cols = ["ring_id", "severity", "score", "n_accounts", "n_fraud",
            "fraud_rate", "device_concentration", "total_amount_inr"]
    return pd.DataFrame(rings)[cols]
