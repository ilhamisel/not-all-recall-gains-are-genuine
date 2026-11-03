#!/usr/bin/env python3
"""Rev.4 Task 1 supplement — surrogate-violation audit of ALL synthetic minority
points (not only the boundary block), same folds/seeds as the CV runs.
For each training fold and minority class: re-fit the (deterministic) OvR
XGB-30 surrogate and report the fraction with p(c|x) < 0.5 among
  CC-strict core block (GS-0.85), CC-strict boundary block, CC-strict core
  replacements, GeoB boundary block, and the whole CC-strict synthetic set.
Output: results/rev4_core_audit.csv (per dataset, pooled over folds/classes)"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import rev4_methods as R
from rev4_run_suite import load
from augment_xai_v4 import _ovr_model
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
rows = []
for suite in ("primary", "external", "real", "synth"):
    ds, _ = load(suite)
    for name, (X, y) in ds.items():
        acc = {}
        for rep in range(3):
            for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42 + rep).split(X, y)):
                Xs = StandardScaler().fit_transform(X[tr]); ytr = y[tr]; seed = 42 + rep * 10 + fold
                R.STATS.clear(); R.CTX.clear()
                Xa, ya = R.CCSTRICT(Xs.copy(), ytr.copy(), seed=seed)
                Xg, yg = R.SGEO(Xs.copy(), ytr.copy(), seed=seed)
                sx, sy = Xa[len(Xs):], ya[len(Xs):]; gx, gy = Xg[len(Xs):], yg[len(Xs):]
                for st in R.STATS:
                    c = st["cls"]; S = sx[sy == c]; G = gx[gy == c]
                    nb = st["n_bnd"]; nc = len(S) - nb; nd = st["n_bnd_delivered"]
                    clf = _ovr_model(Xs, ytr, c, seed)
                    v = lambda Z: (clf.predict_proba(Z)[:, 1] < .5) if len(Z) else np.zeros(0, bool)
                    for k, Z in (("core", S[:nc]), ("bnd", S[nc:nc + nd]), ("repl", S[nc + nd:]),
                                 ("all", S), ("geob_bnd", G[len(G) - nb:] if nb else G[:0]),
                                 ("geob_all", G)):
                        a = acc.setdefault(k, [0, 0]); vv = v(Z); a[0] += vv.sum(); a[1] += len(vv)
        r = dict(suite=suite, dataset=name)
        for k, (nv, n) in acc.items():
            r[f"viol_{k}"] = nv / n if n else np.nan; r[f"n_{k}"] = n
        rows.append(r); print(r, flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "rev4_core_audit.csv"), index=False)
print(df.groupby("suite")[[c for c in df.columns if c.startswith("viol_")]].mean().round(3).to_string())
