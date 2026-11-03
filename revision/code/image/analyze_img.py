#!/usr/bin/env python3
"""Rev.4 — aggregate image-domain results.
Task-level statistic (as in paper): per-task mean over 15 folds; across the
10 tasks a paired two-sided Wilcoxon signed-rank test, Holm-corrected within
each (classifier, metric) family of comparisons against the reference method.
Writes results/summary_<tag>.md and results/task_means_<tag>.csv.
Usage: python analyze_img.py <tag> [ref=CC-strict]"""
import os, sys, glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
from tasks_img import TASKS, DATA

tag = sys.argv[1]
REF = sys.argv[2] if len(sys.argv) > 2 else "CC-strict"
ORDER = ["Base", "Weighted", "PixAug", "ROS", "SMOTE", "Borderline", "ADASYN",
         "SMOTE-ENN", "CalKDE", "GeoB", "CC-SG", "CC-strict"]
MET = ["min_recall", "gmean", "f1_macro", "mcc"]
TORDER = list(TASKS)

df = pd.concat([pd.read_csv(f) for f in
                glob.glob(os.path.join(RES, f"folds_{tag}_*.csv"))])
tm = df.groupby(["task", "clf", "method", "metric"]).value.agg(
    ["mean", "std"]).reset_index()
tm.to_csv(os.path.join(RES, f"task_means_{tag}.csv"), index=False)
tasks = [t for t in TORDER if t in set(df.task)]
L = [f"# Rev.4 image-domain results ({tag}); tasks={len(tasks)}\n"]


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1, run)
    return adj


# ── task table
L.append("## Tasks\n\n| task | n | K | IR (max/min) | minority counts | description |")
L.append("|---|---|---|---|---|---|")
for t in tasks:
    y = np.load(os.path.join(DATA, f"emb_{t}.npz"))["y"]
    vc = pd.Series(y).value_counts()
    mins = ",".join(str(v) for v in sorted(vc[vc < vc.max()].values))
    L.append(f"| {t} | {len(y)} | {len(vc)} | {vc.max()/vc.min():.1f} | "
             f"{mins} | {TASKS[t][2]} |")

for clf in ["RF", "LR"]:
    sub = tm[(tm.clf == clf)]
    piv = sub.pivot_table(index=["task", "method"], columns="metric",
                          values="mean")
    meth = [m for m in ORDER if m in set(sub.method)]
    L.append(f"\n## {clf}: mean over {len(tasks)} tasks (per-task 5x3 CV means)"
             f"; Wilcoxon vs {REF} across tasks, Holm within metric\n")
    L.append("| method | " + " | ".join(MET) +
             " | wins F1/MCC vs " + REF + " (of %d) |" % len(tasks))
    L.append("|---" * (len(MET) + 2) + "|")
    pv = {}
    for met in MET:
        A = piv[met].unstack()                       # task x method
        ref = A.loc[tasks, REF]
        others = [m for m in meth if m != REF]
        ps = []
        for m in others:
            d = A.loc[tasks, m] - ref
            ps.append(1.0 if np.allclose(d, 0) else wilcoxon(A.loc[tasks, m], ref).pvalue)
        for m, p in zip(others, holm(ps)):
            pv[(m, met)] = p
    for m in meth:
        cells = []
        for met in MET:
            A = piv[met].unstack()
            v = A.loc[tasks, m].mean()
            s = f"{v:.3f}"
            if m != REF:
                s += f" (p={pv[(m, met)]:.3f})"
            cells.append(s)
        A1 = piv["f1_macro"].unstack(); A2 = piv["mcc"].unstack()
        w = "" if m == REF else (
            f"{(A1.loc[tasks, REF] > A1.loc[tasks, m]).sum()}/"
            f"{(A2.loc[tasks, REF] > A2.loc[tasks, m]).sum()}")
        L.append(f"| {m} | " + " | ".join(cells) + f" | {w} |")

    # per-task key table
    L.append(f"\n### {clf}: per-task mean±std (15 folds) — Base, Weighted, PixAug, SMOTE, SMOTE-ENN, CalKDE, GeoB, CC-SG, CC-strict\n")
    for met in ["min_recall", "f1_macro", "mcc"]:
        keym = ["Base", "Weighted", "PixAug", "SMOTE", "SMOTE-ENN", "CalKDE",
                "GeoB", "CC-SG", "CC-strict"]
        L.append(f"\n**{met}**\n\n| task | " + " | ".join(keym) + " |")
        L.append("|---" * (len(keym) + 1) + "|")
        for t in tasks:
            r = sub[(sub.task == t) & (sub.metric == met)].set_index("method")
            L.append(f"| {t} | " + " | ".join(
                f"{r.loc[m,'mean']:.3f}±{r.loc[m,'std']:.3f}" for m in keym) + " |")

    # GeoB vs CC-strict / CC-SG paired deltas
    L.append(f"\n### {clf}: gate effect (method − GeoB), per task\n")
    L.append("| task | ΔminRec strict | ΔF1 strict | ΔMCC strict | ΔminRec CC-SG | ΔF1 CC-SG | ΔMCC CC-SG |")
    L.append("|---|---|---|---|---|---|---|")
    D = {}
    for met in ["min_recall", "f1_macro", "mcc"]:
        A = piv[met].unstack().loc[tasks]
        D[("CC-strict", met)] = A["CC-strict"] - A["GeoB"]
        D[("CC-SG", met)] = A["CC-SG"] - A["GeoB"]
    for t in tasks:
        L.append(f"| {t} | " + " | ".join(
            f"{D[(m, met)][t]:+.3f}" for m in ["CC-strict", "CC-SG"]
            for met in ["min_recall", "f1_macro", "mcc"]) + " |")
    for m in ["CC-strict", "CC-SG"]:
        cells = []
        for met in ["min_recall", "f1_macro", "mcc"]:
            d = D[(m, met)]
            p = 1.0 if np.allclose(d, 0) else wilcoxon(d).pvalue
            cells.append(f"{met}: mean {d.mean():+.4f}, +{(d>0).sum()}/-{(d<0).sum()}, p={p:.3f}")
        L.append(f"\n{m} vs GeoB: " + "; ".join(cells))

# CC-strict bookkeeping
fb = [pd.read_csv(f) for f in glob.glob(os.path.join(RES, f"fbstats_{tag}_*.csv"))
      if os.path.getsize(f) > 5]
if fb:
    fb = pd.concat(fb)
    L.append("\n## CC-strict boundary budget bookkeeping (all folds)\n")
    L.append("| task | boundary budget | certified in 6 rounds | fallback | fallback violating (→ core) | share of budget delivered as certified boundary |")
    L.append("|---|---|---|---|---|---|")
    for t in tasks:
        g = fb[fb.task == t]
        nb, nc, nf, nv = g.n_bnd.sum(), g.n_cert.sum(), g.n_fb.sum(), g.n_viol.sum()
        L.append(f"| {t} | {nb} | {nc/nb:.1%} | {nf/nb:.1%} | {nv/max(nf,1):.1%} | {(nc+nf-nv)/nb:.1%} |")
fl = [pd.read_csv(f) for f in glob.glob(os.path.join(RES, f"failures_{tag}_*.csv"))
      if os.path.getsize(f) > 5]
if fl:
    fl = pd.concat(fl)
    L.append("\n## Failures (method fell back to Base)\n")
    L.append(fl.groupby(["task", "method", "exc"]).size().to_string())

open(os.path.join(RES, f"summary_{tag}.md"), "w", encoding="utf-8").write(
    "\n".join(L))
print("\n".join(L))
