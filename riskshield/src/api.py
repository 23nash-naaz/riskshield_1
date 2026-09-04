"""Risk platform API. Run from src/:  uvicorn api:app --port 8000

Serves the merchant console at /, scores RAW transactions via the online
feature store (no precomputed features needed), and replays the held-out
window for the live demo.
"""
import os, json, pickle
from typing import Any
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

import decide as dec
import evidence
from economics import COST, decide, thresholds_for_margin, expected_cost
from featurestore import FeatureStore

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")

app = FastAPI(title="Rupee-Optimal Chargeback Shield")

# CORS for any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

M = pickle.load(open(f"{OUT}/model.pkl", "rb"))
REPLAY = json.load(open(f"{OUT}/replay.json"))

# Load ring data if available
RINGS = []
rings_path = f"{OUT}/rings.json"
if os.path.exists(rings_path):
    RINGS = json.load(open(rings_path))

# Load ablation data if available
ABLATION = []
ablation_path = f"{OUT}/ablation_chart.json"
if os.path.exists(ablation_path):
    ABLATION = json.load(open(ablation_path))

# Load metrics if available
METRICS_FILE = {}
metrics_path = f"{OUT}/metrics.json"
if os.path.exists(metrics_path):
    METRICS_FILE = json.load(open(metrics_path))

FS = FeatureStore()

def _warmup():
    """Rebuild online state from the pre-test stream, as a production store
    would from its event log. Without this, every replay account looks new."""
    path = f"{OUT}/warmup.json"
    if not os.path.exists(path):
        return 0
    for r in json.load(open(path)):
        f = FS.features(r["uid"], r["TransactionAmt"], r["TransactionDT"],
                        r["DeviceInfo"], r["P_emaildomain"], r["card_bin"],
                        r["acct_age_days"])
        FS.update(r["uid"], r["TransactionAmt"], r["TransactionDT"],
                  r["DeviceInfo"], r["P_emaildomain"], r["card_bin"],
                  [f[k] for k in SEQ_FEATS])
    return len(FS.acct)

try:
    import torch, seqenc
    HAS_TORCH = M.get("encoder") is not None
except ImportError:
    HAS_TORCH = False

STATE = {"i": 0, "n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
         "cost_model": 0.0, "cost_allow": 0.0, "margin": 0.20}
SEQ_FEATS = ["f_amt_log", "f_amt_z", "f_amt_ratio", "f_dt_prev", "f_dt_ratio",
             "f_txn_last_1h", "f_txn_last_24h"]
N_WARM = _warmup()


class Txn(BaseModel):
    uid: str
    amount: float
    ts: float
    device: Any = "unk"
    email: Any = "unk"
    card_bin: Any = 0
    acct_age_days: float = 0
    merchant_margin: float = 0.20


def _score_raw(uid, amount, ts, device, email, bin_, age, margin):
    f = FS.features(uid, amount, ts, device, email, bin_, age)
    sv = [f[k] for k in SEQ_FEATS]
    if HAS_TORCH and M["ecols"]:
        seq = FS.sequence(uid) + [sv]
        arr = np.zeros((1, seqenc.K, len(SEQ_FEATS)), dtype="float32")
        w = np.clip(np.array(seq[-seqenc.K:], dtype="float32"), -50, 50)
        arr[0, seqenc.K - len(w):] = w
        emb = seqenc.embed(M["encoder"], arr)[0]
        f.update({c: float(v) for c, v in zip(M["ecols"], emb)})
    X = pd.DataFrame([f]).reindex(columns=M["cols"], fill_value=0.0)
    p = float(M["iso"].predict(M["model"].predict_proba(X)[:, 1])[0])
    c = {**COST, "margin": margin}
    action, costs = decide(p, amount, c)
    shap = M["model"].predict(X, pred_contrib=True)[0][:-1]
    FS.update(uid, amount, ts, device, email, bin_, sv)
    return {"risk_score": round(p, 4), "action": action,
            "reason_codes": dec.reasons(shap, M["cols"]),
            "expected_cost_inr": {k: round(v, 2) for k, v in costs.items()},
            "saved_vs_allow_inr": round(costs["allow"] - costs[action], 2)}


@app.get("/")
def home():
    return FileResponse(os.path.join(HERE, "dashboard.html"))


@app.get("/health")
def health():
    return {"ok": True, "sequence_encoder": HAS_TORCH,
            "replay_remaining": len(REPLAY) - STATE["i"],
            "warmed_accounts": N_WARM}


@app.post("/txn")
def score_txn(t: Txn):
    return _score_raw(t.uid, t.amount, t.ts, t.device, t.email,
                      t.card_bin, t.acct_age_days, t.merchant_margin)


@app.post("/simulate/step")
def simulate(n: int = 1):
    out = []
    for _ in range(min(n, len(REPLAY) - STATE["i"])):
        r = REPLAY[STATE["i"]]; STATE["i"] += 1
        res = _score_raw(r["uid"], r["TransactionAmt"], r["TransactionDT"],
                         r["DeviceInfo"], r["P_emaildomain"], r["card_bin"],
                         r["acct_age_days"], STATE["margin"])
        y = int(r["isFraud"])
        flag = res["action"] != "allow"
        STATE["n"] += 1
        STATE["tp" if (flag and y) else "fp" if flag else "fn" if y else "tn"] += 1
        c = {**COST, "margin": STATE["margin"]}
        STATE["cost_model"] += expected_cost(float(y), r["TransactionAmt"], res["action"], c)
        STATE["cost_allow"] += expected_cost(float(y), r["TransactionAmt"], "allow", c)
        out.append({**res, "txn_id": r["TransactionID"], "amount": round(r["TransactionAmt"], 0),
                    "card": r["card4"], "true_fraud": y})
    return {"events": out, "stats": stats()}


@app.get("/stats")
def stats():
    s = STATE
    return {"processed": s["n"],
            "precision": s["tp"] / max(s["tp"] + s["fp"], 1),
            "recall": s["tp"] / max(s["tp"] + s["fn"], 1),
            "fp_per_1k_good": 1000 * s["fp"] / max(s["fp"] + s["tn"], 1),
            "rupees_saved": round(s["cost_allow"] - s["cost_model"], 0),
            "margin": s["margin"]}


@app.post("/margin")
def set_margin(margin: float):
    STATE["margin"] = max(0.01, min(margin, 0.95))
    return {"margin": STATE["margin"],
            "thresholds": thresholds_for_margin(STATE["margin"], 2500.0)}


@app.get("/thresholds")
def thresholds(margin: float = 0.20, amount: float = 2500.0):
    return thresholds_for_margin(margin, amount)


@app.get("/rings")
def get_rings():
    """Return detected abuse rings with severity and graph edges."""
    return {"rings": RINGS, "count": len(RINGS)}


@app.get("/ablation")
def get_ablation():
    """Return ablation study results for the dashboard chart."""
    return {"stages": ABLATION}


@app.get("/offline-metrics")
def get_offline_metrics():
    """Return the full offline metrics computed by run.py."""
    return METRICS_FILE


class DisputeReq(BaseModel):
    txn: dict[str, Any]
    history: list[dict[str, Any]] = []
    dispute_code: str = "10.4"


@app.post("/dispute-pack")
def dispute(r: DisputeReq):
    p = evidence.build(r.txn, r.history, r.dispute_code)
    return {**p, "text": evidence.to_text(p)}
