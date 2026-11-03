#!/usr/bin/env python3
"""Rev.4 Task 5 — documentation facts (no new model runs).
(a) k actually used by SMOTE-family rivals on Wine_White per training fold
(b) failed augmentation runs recorded in all paper result CSVs (n_fail>0)
(c) data lineage: exact feature-row overlap of primary vs extreme/external
Output: results/rev4_doc_checks.txt, results/rev4_failures_summary.csv,
        results/rev4_lineage.csv"""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.datasets import fetch_datasets
from run_importance_v3 import load_all
from datasets_rev3 import load_real_extreme

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
L = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); L.append(s)

prim = load_all()
# (a)
X, y = prim["Wine_White"]
vc = pd.Series(y).value_counts().sort_index()
p("Wine_White class counts (label-encoded):", dict(vc))
ks = []
for rep in range(3):
    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42 + rep).split(X, y)):
        c = pd.Series(y[tr]).value_counts()
        kmin = int(c.min()); ks.append((rep, f, kmin, max(1, min(5, kmin - 1)),
                                        int(pd.Series(y[te]).value_counts().reindex(vc.index).fillna(0).min())))
p("per fold (rep, fold, min train class count, k used by SMOTE/Borderline/ADASYN/SMOTE-ENN/Tomek/SMOTE-IPF, min test class count):")
for k in ks: p("  ", k)
for n, (Xd, yd) in prim.items():
    kmins = []
    for rep in range(3):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=42 + rep).split(Xd, yd):
            kmins.append(int(pd.Series(yd[tr]).value_counts().min()))
    p(f"  {n:12s} min-train-count range {min(kmins)}-{max(kmins)} -> k range "
      f"{max(1,min(5,min(kmins)-1))}-{max(1,min(5,max(kmins)-1))}")
# (b)
rows = []
for f in sorted(glob.glob(os.path.join(RES, "results_*.csv"))) + sorted(glob.glob(os.path.join(RES, "rev4_*.csv"))):
    try:
        d = pd.read_csv(f)
    except Exception:
        continue
    if "n_fail" not in d.columns:
        continue
    g = d[d.n_fail > 0].groupby(["dataset", "method"]).n_fail.max().reset_index()
    g["file"] = os.path.basename(f); rows.append(g)
fs = pd.concat(rows) if rows else pd.DataFrame()
fs.to_csv(os.path.join(RES, "rev4_failures_summary.csv"), index=False)
p("\nFailed augmentation folds (n_fail>0, out of 15 folds) in result CSVs:")
p(fs.to_string(index=False) if len(fs) else "  none")
for f in sorted(glob.glob(os.path.join(RES, "failures_*.csv"))):
    try:
        d = pd.read_csv(f)
    except Exception:
        continue
    if len(d):
        p(f"  {os.path.basename(f)}: {len(d)} rows; ",
          d.groupby(["dataset", "method", "exc_type"]).size().to_dict(),
          "| msg:", d.exc_msg.iloc[0][:120])
# (c) lineage
def rowset(Xm):
    return set(map(lambda r: tuple(np.round(r, 6)), Xm))
raw = fetch_datasets(filter_data=("wine_quality", "abalone", "abalone_19"))
ext = load_real_extreme()
pairs = [("Wine_White", prim["Wine_White"][0], "imblearn wine_quality (full)", np.asarray(raw["wine_quality"].data, float)),
         ("Wine_White", prim["Wine_White"][0], "wine_quality_x (extreme suite)", ext["wine_quality_x"][0]),
         ("Wine_Red", prim["Wine_Red"][0], "wine_quality_x (extreme suite)", ext["wine_quality_x"][0]),
         ("abalone_19", prim["abalone_19"][0], "imblearn abalone (full)", np.asarray(raw["abalone"].data, float)),
         ("abalone_19", prim["abalone_19"][0], "abalone_x (extreme suite)", ext["abalone_x"][0])]
lin = []
for a, Xa, b, Xb in pairs:
    sa, sb = rowset(Xa), rowset(Xb)
    inter = len(sa & sb)
    r = dict(primary=a, other=b, n_primary=len(Xa), n_other=len(Xb), d_primary=Xa.shape[1],
             d_other=Xb.shape[1], uniq_primary=len(sa), uniq_other=len(sb), exact_overlap_unique=inter,
             frac_other_rows_in_primary=np.mean([tuple(np.round(r_, 6)) in sa for r_ in Xb]) if Xa.shape[1] == Xb.shape[1] else np.nan)
    lin.append(r); p(r)
pd.DataFrame(lin).to_csv(os.path.join(RES, "rev4_lineage.csv"), index=False)
p("imblearn wine_quality target:", dict(pd.Series(raw["wine_quality"].target).value_counts()))
p("imblearn abalone target:", dict(pd.Series(raw["abalone"].target).value_counts()),
  "| abalone_19 target:", dict(pd.Series(raw["abalone_19"].target).value_counts()))
p("abalone_x / wine_quality_x sizes:", {k: (len(v[1]), int(v[1].sum())) for k, v in ext.items() if k in ("abalone_x", "wine_quality_x")})
open(os.path.join(RES, "rev4_doc_checks.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
