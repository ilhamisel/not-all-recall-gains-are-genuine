#!/usr/bin/env python3
"""Rev.4 Task 5c — exhaustive row-overlap lineage check: every primary dataset
vs every external (rev_external) and real-extreme (rev3) dataset with equal d.
Also: for overlapping pairs, which primary classes the other task's minority
rows carry. Output: results/rev4_lineage_all.csv"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from run_importance_v3 import load_all
from run_rev_external import load_external
from datasets_rev3 import load_real_extreme
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
prim = load_all(); others = {**{f"EXT:{k}": v for k, v in load_external().items()},
                             **{f"XTR:{k}": v for k, v in load_real_extreme().items()}}
key = lambda r: tuple(np.round(r, 6))
rows = []
for pn, (Xp, yp) in prim.items():
    idx = {}
    for i, r in enumerate(Xp):
        idx.setdefault(key(r), set()).add(int(yp[i]))
    for on, (Xo, yo) in others.items():
        if Xo.shape[1] != Xp.shape[1]:
            continue
        hit = np.array([key(r) in idx for r in Xo])
        rr = dict(primary=pn, other=on, d=Xp.shape[1], n_other=len(Xo),
                  frac_other_rows_in_primary=hit.mean())
        if hit.mean() > 0.5:
            mino = pd.Series(yo).value_counts().index[-1]
            labs = [tuple(sorted(idx[key(r)])) for r, h, l in zip(Xo, hit, yo) if h and l == mino]
            rr["other_minority_rows_primary_labels"] = str(pd.Series(labs).value_counts().to_dict())
            rr["primary_class_counts"] = str(pd.Series(yp).value_counts().sort_index().to_dict())
        rows.append(rr)
df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "rev4_lineage_all.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print(df[df.frac_other_rows_in_primary > 0].to_string(index=False))
