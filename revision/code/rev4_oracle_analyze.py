#!/usr/bin/env python3
"""Rev.4 Task 3 analysis. Inputs: rev4_oracle_{proposals,delivered,downstream}.csv
Outputs: results/rev4_oracle_summary.csv, rev4_oracle_downstream_wide.csv,
         rev4_oracle_report.txt"""
import os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
L = []
def p(*a):
    s = " ".join(map(str, a)); print(s); L.append(s)
pp = pd.read_csv(os.path.join(RES, "rev4_oracle_proposals.csv"))
dv = pd.read_csv(os.path.join(RES, "rev4_oracle_delivered.csv"))
dn = pd.read_csv(os.path.join(RES, "rev4_oracle_downstream.csv"))
rows = []
for (prob, c), g in pp.groupby(["problem", "cls"]):
    r = dict(problem=prob, cls=c, n_prop=g.n_prop.sum(), admit_rate=(g.admit_rate * g.n_prop).sum() / g.n_prop.sum())
    for ref in ("bayes", "bal"):
        TP, FP, FN, TN = (g[f"{ref}_{k}_rej"].sum() for k in ("TP", "FP", "FN", "TN"))
        n = TP + FP + FN + TN
        r[f"{ref}_geob_frac_true_maj"] = (TP + FN) / n
        r[f"{ref}_rej_precision"] = TP / (TP + FP) if TP + FP else np.nan
        r[f"{ref}_rej_recall"] = TP / (TP + FN) if TP + FN else np.nan
        r[f"{ref}_agreement"] = (TP + TN) / n
        r[f"{ref}_post_adm_mean"] = g[f"{ref}_post_adm_mean"].mean()
        r[f"{ref}_post_rej_mean"] = g[f"{ref}_post_rej_mean"].mean()
        r[f"{ref}_seeds_in_maj"] = g[f"{ref}_frac_minority_seeds_in_maj"].mean()
        for m in ("CC-SG", "CC-strict", "CC-batch"):
            h = dv[(dv.problem == prob) & (dv.cls == c) & (dv.method == m)]
            w = h.n_delivered
            r[f"{ref}_{m}_bnd_true_maj"] = (h[f"{ref}_viol_delivered"] * w).sum() / max(w.sum(), 1)
            if m == "CC-strict":
                wr = h.n_repl
                r[f"{ref}_CC-strict_repl_true_maj"] = (h[f"{ref}_viol_repl"].fillna(0) * wr).sum() / max(wr.sum(), 1)
                r["CC-strict_frac_repl"] = wr.sum() / h.n_bnd.sum()
    rows.append(r)
S = pd.DataFrame(rows); S.to_csv(os.path.join(RES, "rev4_oracle_summary.csv"), index=False)
pd.set_option("display.width", 250)
cols1 = ["problem", "cls", "admit_rate", "bayes_geob_frac_true_maj", "bayes_rej_precision", "bayes_rej_recall",
         "bayes_agreement", "bayes_post_adm_mean", "bayes_post_rej_mean", "bayes_CC-strict_bnd_true_maj",
         "bayes_CC-SG_bnd_true_maj", "bayes_CC-batch_bnd_true_maj"]
cols2 = ["problem", "cls", "bal_geob_frac_true_maj", "bal_rej_precision", "bal_rej_recall", "bal_agreement",
         "bal_post_adm_mean", "bal_post_rej_mean", "bal_CC-strict_bnd_true_maj", "bal_CC-SG_bnd_true_maj",
         "bal_CC-strict_repl_true_maj", "CC-strict_frac_repl", "bal_seeds_in_maj"]
p("== (A) Proposal audit vs TRUE-PRIOR Bayes posterior (true-majority region = P(c|x)<0.5)")
p(S[cols1].round(3).to_string(index=False))
p("\n== (A) Proposal audit vs BALANCED (equal-prior) posterior")
p(S[cols2].round(3).to_string(index=False))
p("\nMeans over problem x class:")
p(S.drop(columns=["problem", "cls"]).mean().round(3).to_string())
# downstream
p("\n== (B) Downstream RF (5x3 CV), mean over problems")
W = dn.pivot_table(index=["dataset", "method"], columns="metric", values="mean").reset_index()
W.to_csv(os.path.join(RES, "rev4_oracle_downstream_wide.csv"), index=False)
mets = ["min_recall", "gmean", "f1_macro", "mcc"]
p(W.groupby("method")[mets].mean().round(4).to_string())
for grp, pat in (("binary", "bin_"), ("multiclass", "mc3_")):
    p(f"\n-- {grp}")
    p(W[W.dataset.str.startswith(pat)].groupby("method")[mets].mean().round(4).to_string())
p("\nPaired over 21 problems (Wilcoxon):")
for a, b in [("CC-strict", "Sgeo"), ("CC-SG", "Sgeo"), ("CC-batch", "CC-strict"), ("Oracle-Bayes", "CC-strict"),
             ("Oracle-bal", "CC-strict"), ("Oracle-bal", "Sgeo"), ("Oracle-Bayes", "Sgeo")]:
    out = []
    for m in mets:
        P = W.pivot_table(index="dataset", columns="method", values=m)
        d = P[a] - P[b]
        pv = wilcoxon(P[a], P[b]).pvalue if not np.allclose(d, 0) else 1.0
        out.append(f"{m} {d.mean():+.4f} ({(d>1e-9).sum()}/{len(d)}, p={pv:.3f})")
    p(f"  {a} - {b}: " + " | ".join(out))
fl = os.path.join(RES, "rev4_oracle_failures.csv")
if os.path.exists(fl):
    p("\nfailures:", len(pd.read_csv(fl)))
open(os.path.join(RES, "rev4_oracle_report.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
