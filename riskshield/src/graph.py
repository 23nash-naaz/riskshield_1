"""Shared-identifier graph features (abuse-ring signal).

Bipartite: account <-> {device, email domain, card BIN}.
Degree and component size are computed on the TRAIN window only, then mapped
onto test rows. Building the graph over test data would leak.
"""
import numpy as np
import pandas as pd
import networkx as nx

IDS = ["DeviceInfo", "P_emaildomain", "card_bin"]


def fit_graph(train):
    G = nx.Graph()
    for col in IDS:
        e = train[["uid", col]].dropna().drop_duplicates()
        G.add_edges_from(zip("u:" + e.uid.astype(str),
                             f"{col}:" + e[col].astype(str)))
    comp = {n: i for i, c in enumerate(nx.connected_components(G)) for n in c}
    csize = pd.Series(comp).groupby(lambda n: comp[n]).size()
    return {"deg": dict(G.degree()), "comp": comp,
            "csize": {n: int(csize.get(comp[n], 1)) for n in G}}


def apply_graph(df, gf):
    out = df.copy()
    for col in IDS:
        k = f"{col}:" + df[col].astype(str)
        out[f"g_deg_{col}"] = k.map(gf["deg"]).fillna(0)
    u = "u:" + df.uid.astype(str)
    out["g_comp_size"] = u.map(gf["csize"]).fillna(1)
    out["g_uid_deg"] = u.map(gf["deg"]).fillna(0)
    # devices touching many distinct cards = classic card-testing signature
    out["g_dev_per_uid"] = out.g_deg_DeviceInfo / (out.g_uid_deg + 1)
    return out
