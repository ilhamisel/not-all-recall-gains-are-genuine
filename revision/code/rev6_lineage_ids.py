#!/usr/bin/env python3
"""Observation-level lineage between the primary suite and the external / real-extreme suites.
No model is trained: the datasets are loaded with the same loaders as in the experiments and rows are
matched on all feature values rounded to 6 decimals (as in rev4_lineage_all.py).

Output: results/lineage_observation_ids.csv, one row per matched (other-task row, primary row) pair:
  pair, other_task, other_row, other_label, other_is_minority,
  primary_task, primary_row, primary_label, n_primary_matches
Row indices are 0-based positions in the matrices returned by the experiment loaders
(run_importance_v3.load_all, run_rev_external.load_external, datasets_rev3.load_real_extreme).
The source data carry no observation IDs, so a feature-vector match identifies identical
feature rows, not necessarily the same physical observation when duplicates exist.
Also writes results/lineage_summary.csv."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from run_importance_v3 import load_all
from run_rev_external import load_external
from datasets_rev3 import load_real_extreme

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
prim = load_all()
ext = load_external(); xtr = load_real_extreme()
PAIRS = [("Wine_White", "wine_quality_x", xtr), ("Wine_Red", "wine_quality_x", xtr),
         ("abalone_19", "abalone_x", xtr), ("car_eval_4", "car_eval_34", ext)]
key = lambda r: tuple(np.round(r, 6))
rows, summ = [], []
for pn, on, src in PAIRS:
    Xp, yp = prim[pn]; Xo, yo = src[on]
    idx = {}
    for i, r in enumerate(Xp):
        idx.setdefault(key(r), []).append(i)
    mino = pd.Series(yo).value_counts().idxmin()
    n_hit = 0; n_dup_p = 0
    for j, r in enumerate(Xo):
        m = idx.get(key(r), [])
        if not m:
            continue
        n_hit += 1; n_dup_p += len(m) > 1
        for i in m:
            rows.append(dict(pair=f"{on}~{pn}", other_task=on, other_row=j, other_label=int(yo[j]),
                             other_is_minority=bool(yo[j] == mino), primary_task=pn, primary_row=i,
                             primary_label=int(yp[i]), n_primary_matches=len(m)))
    mp = set(i for r in Xo for i in idx.get(key(r), []))
    summ.append(dict(other_task=on, primary_task=pn, n_other=len(Xo), n_primary=len(Xp),
                     other_rows_matched=n_hit, frac_other_rows_matched=n_hit / len(Xo),
                     primary_rows_covered=len(mp), frac_primary_rows_covered=len(mp) / len(Xp),
                     other_rows_with_multiple_primary_matches=n_dup_p,
                     other_minority=int((yo == mino).sum()),
                     other_minority_matched=int(sum(1 for j, r in enumerate(Xo) if yo[j] == mino and key(r) in idx))))
pd.DataFrame(rows).to_csv(os.path.join(RES, "lineage_observation_ids.csv"), index=False)
S = pd.DataFrame(summ); S.to_csv(os.path.join(RES, "lineage_summary.csv"), index=False)
pd.set_option("display.width", 250)
print(S.to_string(index=False))
# Wine_Red vs wine_quality_x: are the matched rows also present in Wine_White?
Xw, _ = prim["Wine_White"]; wk = set(key(r) for r in Xw)
D = pd.DataFrame(rows); red = D[D.primary_task == "Wine_Red"]
for _, r in red.iterrows():
    v = xtr["wine_quality_x"][0][r.other_row]
    print("Wine_Red match: other_row", r.other_row, "red_row", r.primary_row, "also in Wine_White:", key(v) in wk)
