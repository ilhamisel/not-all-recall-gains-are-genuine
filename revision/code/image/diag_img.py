#!/usr/bin/env python3
"""Rev.4 — embedding-space analogue of the paper's certification diagnostic
(76% surrogate rejection of GeoB proposals; 1-NN / 5-NN real-neighbour proxy).

For every task, rep-0 CV folds (5), same fold pipeline as run_img.py
(L2 -> PCA fit on train fold -> StandardScaler on train fold), every
non-majority class c: 2,000 GeoB proposals (seed in c, partner among 5 nearest
majority, depth U(0.25,0.75)); admission = OvR XGB-30 surrogate p>=0.5
(exactly CC-SG / CC-strict's test). Surrogate-independent proxies on REAL
training instances: nearest real neighbour is non-c (nn1_maj), share of
non-c among 5 nearest real neighbours (nn5_maj).
Output: results/diag_<tag>.csv
Usage: python diag_img.py <tag> <pca_dim> [tasks...]
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors
from augment_xai_v4 import _ovr_model, _minority_iter
from tasks_img import TASKS, DATA

N_CAND = 2000


def main():
    tag, d = sys.argv[1], int(sys.argv[2])
    names = sys.argv[3:] or list(TASKS)
    rows = []
    for name in names:
        z = np.load(os.path.join(DATA, f"emb_{name}.npz"))
        Eall, y = normalize(z["E"]), z["y"]
        nmax = pd.Series(y).value_counts().max()
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        for fold, (tr, _) in enumerate(skf.split(Eall, y)):
            pca = PCA(n_components=d, random_state=0).fit(Eall[tr])
            X = StandardScaler().fit_transform(pca.transform(Eall[tr]))
            yt = y[tr]
            rng = np.random.RandomState(42 + fold)
            nn_all = NearestNeighbors(n_neighbors=5).fit(X)
            for c, _diff in _minority_iter(X, yt):
                X_c, X_maj = X[yt == c], X[yt != c]
                kk = min(5, len(X_maj))
                _, nbr = NearestNeighbors(n_neighbors=kk).fit(
                    X_maj).kneighbors(X_c)
                si = rng.randint(len(X_c), size=N_CAND)
                mi = nbr[si, rng.randint(kk, size=N_CAND)]
                P, Q = X_c[si], X_maj[mi]
                t = rng.uniform(0.25, 0.75, size=N_CAND)
                cand = P + t[:, None] * (Q - P)
                p = _ovr_model(X, yt, c, seed=42).predict_proba(cand)[:, 1]
                ok = p >= 0.5
                _, i5 = nn_all.kneighbors(cand)
                nn1 = yt[i5[:, 0]] != c
                nn5 = (yt[i5] != c).mean(1)
                ir = nmax / (y == c).sum()
                for grp, m in (("admit", ok), ("reject", ~ok)):
                    rows.append(dict(task=name, fold=fold, cls=int(c),
                                     IR_c=float(ir), grp=grp, n=int(m.sum()),
                                     nn1_maj=float(nn1[m].mean()) if m.any()
                                     else np.nan,
                                     nn5_maj=float(nn5[m].mean()) if m.any()
                                     else np.nan))
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results", f"diag_{tag}.csv"), index=False)

    def summ(g):
        a, r = g[g.grp == "admit"], g[g.grp == "reject"]
        w = lambda gg, col: np.nansum(gg[col] * gg.n) / max(gg.n.sum(), 1)
        return pd.Series(dict(rej=r.n.sum() / (a.n.sum() + r.n.sum()),
                              nn1_admit=w(a, "nn1_maj"),
                              nn1_reject=w(r, "nn1_maj"),
                              nn5_admit=w(a, "nn5_maj"),
                              nn5_reject=w(r, "nn5_maj")))
    print(df.groupby("task", sort=False).apply(summ).round(3).to_string())
    print("ALL:", summ(df).round(3).to_dict())
    print("IR_c>=50 only:", summ(df[df.IR_c >= 50]).round(3).to_dict())


if __name__ == "__main__":
    main()
