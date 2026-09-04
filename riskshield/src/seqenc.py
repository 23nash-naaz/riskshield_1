"""GRU over each account's last-K transactions -> 32-d behavioural embedding.

Fed into LightGBM as extra columns, not used as the final classifier: GBDT
stays better on the tabular part, the GRU contributes temporal shape GBDT
can't see (ramping, bursting, drift away from baseline).
"""
import numpy as np
import torch
import torch.nn as nn

K = 16
FEATS = ["f_amt_log", "f_amt_z", "f_amt_ratio", "f_dt_prev", "f_dt_ratio",
         "f_txn_last_1h", "f_txn_last_24h"]


def build_seqs(df, key="uid"):
    """(N, K, F) tensor of the K txns ENDING AT and including row i."""
    X = df[FEATS].fillna(0).clip(-50, 50).values.astype("float32")
    out = np.zeros((len(df), K, len(FEATS)), dtype="float32")
    pos = {v: i for i, v in enumerate(df.index)}
    for _, idx in df.groupby(key, sort=False).indices.items():
        for j, r in enumerate(idx):
            lo = max(0, j - K + 1)
            w = X[idx[lo:j + 1]]
            out[r, K - len(w):] = w
    return out


class Enc(nn.Module):
    def __init__(self, f, h=32):
        super().__init__()
        self.gru = nn.GRU(f, h, batch_first=True)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        _, hn = self.gru(x)
        e = hn[-1]
        return self.head(e).squeeze(-1), e


def train_encoder(Xtr, ytr, epochs=6, bs=1024, seed=0):
    torch.manual_seed(seed)
    m = Enc(Xtr.shape[2])
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    pw = torch.tensor([(1 - ytr.mean()) / max(ytr.mean(), 1e-6)], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            logit, _ = m(Xt[b])
            lossf(logit, yt[b]).backward()
            opt.step()
    return m


@torch.no_grad()
def embed(m, X, bs=4096):
    m.eval()
    return np.vstack([m(torch.tensor(X[i:i + bs]))[1].numpy()
                      for i in range(0, len(X), bs)])
