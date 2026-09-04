"""End-to-end: data -> entity -> history -> graph -> sequence -> model -> rupees.
Run: python run.py
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

import synth, entity, history, graph, labels, drift, train, evaluate, rings, returns
try:
    import seqenc
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[seq  ] torch not installed -- skipping sequence embedding stage (all else runs)")
from economics import COST, decide_vec, thresholds_for_margin

DAY = 86400
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


CANDIDATES = [
    "data/train_transaction.csv",
    "/kaggle/input/ieee-fraud-detection/train_transaction.csv",
    "../input/ieee-fraud-detection/train_transaction.csv",
    "/content/train_transaction.csv",
]


def load():
    for p in CANDIDATES:
        if os.path.exists(p):
            print(f"[data] real IEEE-CIS from {p}")
            df = pd.read_csv(p, usecols=lambda c: c in {
                "TransactionID", "TransactionDT", "TransactionAmt", "isFraud",
                "card1", "card4", "card6", "addr1", "D1", "ProductCD",
                "P_emaildomain", "DeviceInfo"} or c.startswith("C"))
            df["card_bin"] = df.card1 % 400        # BIN proxy; real BIN not in file
            df["DeviceInfo"] = df.DeviceInfo.fillna("unk")
            df["P_emaildomain"] = df.P_emaildomain.fillna("unk")
            return df
    print("[data] synthetic (IEEE-CIS schema)")
    return synth.generate()


def temporal_split(df, val_frac=0.15, test_frac=0.25):
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    a, b = int(n * (1 - val_frac - test_frac)), int(n * (1 - test_frac))
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def main():
    df = load()
    print(f"[data] {len(df):,} rows | fraud {df.isFraud.mean():.3%}")

    # 1. entity resolution
    df = entity.add_uid(df)
    q = entity.uid_quality(df)
    if q: print(f"[uid ] {q['n_uid']:,} accounts | {q['pct_pure']:.1%} single-account pure")

    # 2. leak-safe history
    df = history.add_history(df)

    # 3. delayed labels: train only on matured window
    mature, fresh, cut = labels.maturity_split(df)
    print(f"[label] matured {len(mature):,} | unlabelled-fresh {len(fresh):,} (held out of training)")

    tr, va, te = temporal_split(mature)
    print(f"[split] train {len(tr):,} | val {len(va):,} | test {len(te):,} (temporal, no shuffle)")

    # 4. graph fitted on TRAIN only
    gf = graph.fit_graph(tr)
    tr, va, te = (graph.apply_graph(x, gf) for x in (tr, va, te))

    # 4b. abuse-ring sentinel
    detected_rings = rings.extract_rings(tr, gf, min_size=3, top_k=10)
    if detected_rings:
        rt = rings.summary_table(detected_rings)
        print(f"[rings] {len(detected_rings)} suspicious rings detected")
        print(rt.head(5).to_string(index=False))
    else:
        print("[rings] no suspicious rings above threshold")

    base = ["f_amt_log", "TransactionAmt", "acct_age_days", "is_new_acct", "f_hour"]
    hist = ["h_n_prev", "f_amt_z", "f_amt_ratio", "f_amt_vs_max", "f_dt_prev",
            "f_dt_ratio", "f_txn_last_1h", "f_txn_last_24h"]
    gcols = [c for c in tr.columns if c.startswith("g_")]

    # 5. drift audit
    au = drift.audit(tr, te, base + hist + gcols, tr.isFraud.values)
    print(f"[drift] adversarial AUC {au['auc']:.3f} | dropping {au['unstable'] or 'nothing'}")
    drop = set(au["unstable"])

    # 6. sequence encoder (optional -- needs torch)
    ecols = []
    if HAS_TORCH:
        print("[seq  ] training GRU encoder...")
        Str, Sva, Ste = (seqenc.build_seqs(x.reset_index(drop=True)) for x in (tr, va, te))
        enc = seqenc.train_encoder(Str, tr.isFraud.values)
        ecols = [f"e{i}" for i in range(32)]
        for X, S in ((tr, Str), (va, Sva), (te, Ste)):
            X[ecols] = seqenc.embed(enc, S)

    # 7. ablation
    stages = {
        "baseline (raw row)": base,
        "+ entity history": base + hist,
        "+ graph": base + hist + gcols,
    }
    if ecols:
        stages["+ sequence embed"] = base + hist + gcols + ecols
    y_te, amt_te = te.isFraud.values, te.TransactionAmt.values
    results, final = [], None
    for name, cols in stages.items():
        cols = [c for c in cols if c not in drop]
        m = train.fit(tr[cols], tr.isFraud, va[cols], va.isFraud)
        iso = train.calibrate(m, va[cols], va.isFraud)
        p_raw = m.predict_proba(te[cols])[:, 1]
        p_cal = iso.predict(p_raw)
        results.append({
            "stage": name, "n_feat": len(cols),
            "PR_AUC_raw": average_precision_score(y_te, p_raw),
            "PR_AUC_cal": average_precision_score(y_te, p_cal),
            "ECE_raw": train.ece(p_raw, y_te),
            "ECE_cal": train.ece(p_cal, y_te),
            "rupees_per_1k": evaluate.rupees(y_te, p_cal, amt_te)["model"],
        })
        r = results[-1]
        final = (m, iso, cols, p_cal)
        print(f"  {name:22s} PR-AUC raw {r['PR_AUC_raw']:.4f} / cal {r['PR_AUC_cal']:.4f}"
              f"   ECE {r['ECE_raw']:.4f}->{r['ECE_cal']:.4f}"
              f"   Rs {r['rupees_per_1k']:,.0f}/1k")

    m, iso, cols, p = final
    ab = pd.DataFrame(results)

    # 8. honest reporting
    mean, lo, hi = evaluate.boot_ap(y_te, p)
    op = evaluate.op_point(y_te, p, amt_te)
    ru = evaluate.rupees(y_te, p, amt_te)
    sl = evaluate.slices(te, y_te, p, amt_te)

    print("\n=== HELD-OUT TEST (future window, never seen) ===")
    print(f"PR-AUC (cal)  {mean:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"PR-AUC (raw)  {ab.PR_AUC_raw.iloc[-1]:.4f}   "
          f"(isotonic trades a little ranking AP for usable probabilities)")
    print(f"Precision     {op['precision']:.3f}")
    print(f"Recall        {op['recall']:.3f}")
    print(f"FP / 1k good  {op['fp_per_1k_good']:.2f}")
    print(f"Actions       allow {1-op['review_rate']:.1%} | step-up {op['pct_stepup']:.1%} | block {op['pct_block']:.1%}")
    print("\n--- Rupees lost per 1,000 transactions ---")
    for k, v in sorted(ru.items(), key=lambda x: x[1]):
        print(f"  {k:12s} Rs {v:10,.0f}" + ("   <-- ours" if k == "model" else ""))
    saved = ru["allow_all"] - ru["model"]
    print(f"\n  Saved vs do-nothing: Rs {saved:,.0f} per 1,000 txns "
          f"= Rs {saved/1000/te.TransactionAmt.mean()*1e7:,.0f} per Rs 1 crore processed")

    print("\n--- Worst slices (where we are weakest) ---")
    print(sl.head(5).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n--- Per-merchant thresholds (score at which action flips) ---")
    for mg in (0.05, 0.20, 0.60):
        t = thresholds_for_margin(mg, te.TransactionAmt.median())
        print(f"  margin {mg:.0%}: " + " ".join(f"{k}>={v:.3f}" for k, v in t.items()))

    # 9. return-risk scoring
    print("\n--- Return-Risk Scorer ---")
    ret_df = returns.generate_return_data(df)
    ret_model, ret_iso, ret_metrics = returns.train_return_scorer(ret_df)
    if ret_metrics.get("pr_auc", 0) > 0:
        print(f"  Return abuse PR-AUC: {ret_metrics['pr_auc']:.4f}")
        print(f"  Abuse rate: {ret_metrics['abuse_rate']:.2%}")
        print(f"  Avg cost per abusive return: Rs {ret_metrics['avg_abuse_cost_inr']:,.0f}")
    else:
        print(f"  {ret_metrics.get('note', 'skipped')}")

    ab.to_csv(f"{OUT}/ablation.csv", index=False)
    sl.to_csv(f"{OUT}/slices.csv", index=False)
    json.dump({"pr_auc_cal": mean, "pr_auc_raw": ab.PR_AUC_raw.iloc[-1], "ci": [lo, hi], **op, "rupees": ru,
               "drift_auc": au["auc"], "dropped": au["unstable"],
               "return_metrics": ret_metrics},
              open(f"{OUT}/metrics.json", "w"), indent=2, default=float)
    pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)\
      .head(20).to_csv(f"{OUT}/importance.csv")

    # export ring data for dashboard
    json.dump(detected_rings, open(f"{OUT}/rings.json", "w"), indent=2, default=float)

    # export ablation in chart-friendly format
    json.dump(results, open(f"{OUT}/ablation_chart.json", "w"), indent=2, default=float)

    import pickle
    art = {"model": m, "iso": iso, "cols": cols,
           "encoder": enc if ecols else None, "ecols": ecols}
    pickle.dump(art, open(f"{OUT}/model.pkl", "wb"))
    # replay stream for the live demo: raw fields + true label, held-out window
    replay_cols = ["TransactionID", "TransactionDT", "TransactionAmt", "uid",
                   "DeviceInfo", "P_emaildomain", "card_bin", "acct_age_days",
                   "card4", "isFraud"]
    te[replay_cols].to_json(f"{OUT}/replay.json", orient="records")
    # warmup stream: everything BEFORE the test window, so the online store
    # has each account's history when replay begins (matches training view)
    warm = pd.concat([tr, va]).sort_values("TransactionDT")
    warm[replay_cols].to_json(f"{OUT}/warmup.json", orient="records")
    print(f"\n[out] {OUT}/ ablation.csv slices.csv metrics.json importance.csv model.pkl")


if __name__ == "__main__":
    main()
