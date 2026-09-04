# EXECUTION GUIDE — Rupee-Optimal Chargeback Shield

One document. Follow top to bottom for your platform.

```
WORKFLOW
========
develop locally (VS Code, synthetic data, ~3 min runs)
        |
        v
push results are stable -> run on Kaggle (real IEEE-CIS, 590k rows)
        |
        v
demo: run.py output + out/*.csv + API on localhost
```

---

## PART A — VS Code (local, synthetic data)

### A1. Open the project
- Unzip `riskshield`
- VS Code -> `File` -> `Open Folder` -> select the folder that directly
  contains `run.py` (not its parent)
- Sanity check: the Explorer sidebar shows `run.py`, `requirements.txt`, `src/`

### A2. Terminal + environment
Open the built-in terminal: `` Ctrl+` ``

```bash
python --version        # must print 3.10 or higher
python -m venv .venv
```

Activate:
```bash
source .venv/bin/activate        # Mac / Linux
.venv\Scripts\activate           # Windows PowerShell
```
Prompt now starts with `(.venv)`.

Point VS Code at it: `Ctrl+Shift+P` -> `Python: Select Interpreter` ->
choose the entry containing `.venv`.

### A3. Install
```bash
pip install -r requirements.txt
```
torch is the slow line (~2 GB). If it fails or you're impatient, delete the
`torch>=2.0` line and install again — the pipeline detects the absence and
skips only the sequence-embedding stage. Nothing else changes.

### A4. Run
```bash
python run.py
```
Run from the project root, never from inside `src/`.
Runtime: ~3 min with torch, ~1 min without.

### A5. Verify success
First line of output:
```
[data] synthetic (IEEE-CIS schema)
```
Last line:
```
[out] .../out/ ablation.csv slices.csv metrics.json importance.csv model.pkl
```
And `out/` now contains those five files. Expected headline numbers
(deterministic, seeds are fixed): PR-AUC raw 0.9856, precision 0.954,
recall 0.984, FP/1k 1.01, model cost Rs 1,253/1k.

### A6. Launch the risk platform (the demo)
```bash
cd src
uvicorn api:app --port 8000
```
Open **http://localhost:8000** -> the merchant risk console.
1. Click **Start live traffic** — held-out transactions stream through the
   online feature store and decision engine in real time
2. Drag the **margin slider** — watch the allow/step-up/block bands move
3. Red dots = confirmed chargebacks from held-out labels, revealed only
   AFTER the decision
Startup takes ~10 s (warms the feature store from the pre-test stream).
API docs at http://localhost:8000/docs. `Ctrl+C` to stop.

---

## PART B — Kaggle (real IEEE-CIS data)

### B1. Notebook + dataset
1. kaggle.com -> `Create` -> `New Notebook`
2. Right panel -> `+ Add Input` -> search `IEEE-CIS Fraud Detection`
   (the official competition dataset) -> click `+`
3. It mounts read-only at `/kaggle/input/ieee-fraud-detection/`

### B2. Upload the code
1. On your machine, zip the `riskshield` folder
2. In the notebook: `+ Add Input` -> `Upload` -> `New Dataset` ->
   drag the zip -> title it `riskshield-code` -> `Create`
3. It mounts at `/kaggle/input/riskshield-code/`

### B3. Settings (right panel)
- Internet: **Off** (nothing needs downloading; lightgbm/torch/networkx are preinstalled)
- Accelerator: **None** (GPU doesn't help; LightGBM is CPU-bound)

### B4. Cell 1 — copy code to the writable area
```python
!mkdir -p /kaggle/working/riskshield
!cp -r /kaggle/input/riskshield-code/riskshield/* /kaggle/working/riskshield/ 2>/dev/null \
 || cp -r /kaggle/input/riskshield-code/* /kaggle/working/riskshield/
!ls /kaggle/working/riskshield
```
Must list `run.py` and `src`. If not, find the true nesting:
`!find /kaggle/input/riskshield-code -name run.py` and fix the `cp` path.

### B5. Cell 2 — run
```python
%cd /kaggle/working/riskshield
!python run.py
```
First line must be:
```
[data] real IEEE-CIS from /kaggle/input/ieee-fraud-detection/train_transaction.csv
```
If it says `synthetic`, the dataset input from B1 is missing.

Runtime on the full 590k rows: 10–20 min. Fine for iteration:

### B6. (optional) Cell 0 — subsample for faster iteration
```python
import re
p = "/kaggle/working/riskshield/run.py"
s = open(p).read()
s = s.replace("df = load()",
    "df = load().sample(200000, random_state=0).sort_values('TransactionDT')")
open(p, "w").write(s)
print("subsampled to 200k")
```
Run this BEFORE cell 2 (and after cell 1). Remove for the final full run.

### B7. Collect results
```python
import json
print(json.load(open("/kaggle/working/riskshield/out/metrics.json")))
```
`out/` files are downloadable from the right panel under `Output` after
`Save Version` -> `Save & Run All`.

### B8. Expectations on real data
PR-AUC lands around 0.5–0.6, not 0.98 — real fraud is messier than the
synthetic injection. That is 15x+ better than random at a 3.5% base rate,
and the rupee comparison and ablation shape are the results that matter.
The entity-history stage should still be the single biggest jump; if it
isn't, check the `[uid]` line printed at the start.

---

## PART C — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'entity'` | ran from inside `src/` | `cd ..`, run from project root |
| `TypeError ... 'type' object is not subscriptable` | Python 3.9 | use Python 3.10+ (`python3.11 run.py`) |
| torch install fails / too slow | 2 GB wheel | delete torch from requirements.txt; pipeline auto-skips that stage |
| `FileNotFoundError: ../out/model.pkl` on API start | model not trained yet | run `python run.py` first |
| API `/score` returns risk_score 0.0 | embedding columns e0–e31 not supplied | expected; see README "Known limitations" |
| Kaggle prints `[data] synthetic` | competition dataset not attached | redo B1 |
| Kaggle cell killed / OOM | full 590k + all features | use B6 subsample |
| VS Code runs wrong Python | interpreter not selected | `Ctrl+Shift+P` -> Python: Select Interpreter -> `.venv` |
| `pip: command not found` inside venv (rare) | broken venv | `python -m pip install -r requirements.txt` |

---

## PART D — Command summary (copy-paste)

Local, end to end:
```bash
cd riskshield
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py
cd src && uvicorn api:app --reload
```

Kaggle, cell 1 then cell 2:
```python
!mkdir -p /kaggle/working/riskshield && cp -r /kaggle/input/riskshield-code/riskshield/* /kaggle/working/riskshield/
%cd /kaggle/working/riskshield
!python run.py
```
