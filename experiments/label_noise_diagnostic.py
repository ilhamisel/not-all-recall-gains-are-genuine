"""Model-independent label-noise diagnostic (paper Section 5.3, Fig. 8).

For each minority class, geometric boundary proposals are generated and split
into those the surrogate admits vs. rejects. For each proposal we then record,
using the *real* training data only (independent of the surrogate), whether its
nearest neighbour belongs to the majority class. Rejected proposals should sit
near majority observations much more often than admitted ones -- evidence that
the admission test responds to genuine class-region structure rather than to
surrogate idiosyncrasies.

    python experiments/label_noise_diagnostic.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccsg.core import majority_neighbors, minority_iter, ovr_surrogate  # noqa
from ccsg.datasets import load_primary                                  # noqa

N_CAND = 2000
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    rows = []
    for ds, (X, y) in load_primary().items():
        X = StandardScaler().fit_transform(np.asarray(X, float))
        y = np.asarray(y)
        rng = np.random.RandomState(42)
        for c, _ in minority_iter(X, y):
            X_c, X_maj = X[y == c], X[y != c]
            if len(X_c) < 3:
                continue
            kk, nbr = majority_neighbors(X_c, X_maj)
            clf = ovr_surrogate(X, y, c, seed=42)
            si = rng.randint(len(X_c), size=N_CAND)
            mi = nbr[si, rng.randint(kk, size=N_CAND)]
            P, Q = X_c[si], X_maj[mi]
            t = rng.uniform(0.25, 0.75, size=N_CAND)
            cand = P + t[:, None] * (Q - P)
            admitted = clf.predict_proba(cand)[:, 1] >= 0.5
            _, i1 = NearestNeighbors(n_neighbors=1).fit(X).kneighbors(cand)
            nn_is_majority = (y[i1[:, 0]] != c)
            for grp, mask in [("admitted", admitted), ("rejected", ~admitted)]:
                if mask.sum():
                    rows.append(dict(dataset=ds, cls=int(c), group=grp,
                                     n=int(mask.sum()),
                                     rejected_frac=float((~admitted).mean()),
                                     nn_majority=float(nn_is_majority[mask].mean())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "label_noise_diagnostic.csv"), index=False)

    w = lambda g, col: (g[col] * g.n).sum() / max(g.n.sum(), 1)
    a, r = df[df.group == "admitted"], df[df.group == "rejected"]
    print(f"overall rejection rate : {r.n.sum() / (a.n.sum() + r.n.sum()):.1%}")
    print(f"nearest neighbour is majority -- admitted: {w(a,'nn_majority'):.1%}"
          f"  rejected: {w(r,'nn_majority'):.1%}")


if __name__ == "__main__":
    main()
