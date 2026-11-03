#!/usr/bin/env python3
"""Rev.4 analysis for Tasks 1, 2, 4.
Task 1: CC-strict as reference; dataset-level two-sided Wilcoxon signed-rank
        (scipy default, as in make_tables_ccsg.t_stats) + Holm within each
        (suite, classifier, metric, family). Families stated explicitly.
        Quota / violation statistics of the delivered boundary set.
Task 2: CC-batch vs CC-strict vs GeoB (+GeoF) on primary + extreme suites.
Task 4: paired per-dataset differences CC-strict - rival, bootstrap 95% CI of
        the mean difference (10,000 resamples over datasets, percentile) and
        Hodges-Lehmann estimate (median of Walsh averages), RF, 4 metrics.
Outputs (results/): rev4_task1_holm.csv, rev4_task1_quota.csv,
        rev4_task1_repro.csv, rev4_task1_means.csv, rev4_task2_batch.csv,
        rev4_task2_quota.csv, rev4_task4_bootstrap.csv, rev4_stats_log.txt"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
LOG = []
METR4 = ["min_recall", "gmean", "f1_macro", "mcc"]
METR5 = ["min_recall", "rare_recall", "gmean", "f1_macro", "mcc"]
REF = "CC-strict"


def p(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


def rd(f):
    return pd.read_csv(os.path.join(RES, f))


def piv(df, clf, metric):
    return (df[(df.clf == clf) & (df.metric == metric)]
            .pivot_table(index="method", columns="dataset", values="mean"))


# ── assemble per-suite long tables: method x dataset per (clf, metric) ─────

def suite_tables():
    """Returns {suite: (df_long, families)}; df_long columns dataset, method,
    clf, metric, mean. CC-strict / CC-batch / GeoF / re-runs from rev4."""
    S = {}
    # primary
    v5 = rd("results_xai_v5.csv")
    rb = rd("results_rev_baselines.csv"); rb = rb[rb.method != "Base"].copy()
    rb["clf"] = rb["clf"].str.replace("-W", "", regex=False)
    r3p = rd("results_rev3_primary.csv")
    mv = rd("results_master_v2.csv")
    mv = mv[mv.method.isin(["ROS", "SVM-SMOTE", "KMeans-SMOTE", "PlainKDE", "GMM-S"])]
    brf = rd("results_rev2_brf.csv"); brf = brf[brf.method == "BalancedRF"].copy()
    brf["clf"] = "RF"
    xl = os.path.join(RES, "rev4_primaryXL.csv")
    r4 = pd.concat([rd("rev4_primary.csv")] + ([pd.read_csv(xl)] if os.path.exists(xl) else []),
                   ignore_index=True)
    r4n = r4[r4.method.isin(["CC-strict", "CC-batch", "GeoF"])].copy()
    r4n["method"] = r4n["method"].replace({"CC-SG": "CC-SG"})
    long = pd.concat([v5, rb, r3p, mv, brf, r4n], ignore_index=True)
    fam = {"A_main": ["Base", "Weighted", "SMOTE", "Borderline", "ADASYN",
                      "SMOTE-ENN", "SMOTE-Tomek", "GS-0.85", "Sgeo", "CC-SG"],
           "B_extended": ["Base", "Weighted", "SMOTE", "Borderline", "ADASYN",
                          "SMOTE-ENN", "SMOTE-Tomek", "GS-0.85", "Sgeo", "CC-SG",
                          "SMOTE-IPF", "MWMOTE", "VAE", "ThreshMove", "ROS",
                          "SVM-SMOTE", "KMeans-SMOTE", "PlainKDE", "GMM-S",
                          "BalancedRF"]}
    S["primary"] = (long, fam, r4)
    # external
    ex = rd("results_rev_external.csv")
    w = ex[ex.clf == "RF-W"].copy(); w["clf"] = "RF"; w["method"] = "Weighted"
    ex = pd.concat([ex[ex.clf == "RF"], w])
    r4 = rd("rev4_external.csv")
    long = pd.concat([ex, r4[r4.method.isin(["CC-strict", "CC-batch", "GeoF"])]])
    S["external"] = (long, {"A_main": ["Base", "Weighted", "SMOTE", "SMOTE-ENN",
                                       "SMOTE-Tomek", "GS-0.85", "Sgeo", "CC-SG"]}, r4)
    for tag in ("real", "synth"):
        r3 = rd(f"results_rev3_{tag}.csv")
        r3 = r3[r3.method != "CC-strict"]           # replaced by the rev4 re-run
        r4 = rd(f"rev4_{tag}.csv")
        long = pd.concat([r3, r4[r4.method.isin(["CC-strict", "CC-batch", "GeoF"])]])
        S[tag] = (long, {"A_main": ["Base", "Weighted", "ThreshMove", "SMOTE",
                                    "SMOTE-ENN", "SMOTE-IPF", "MWMOTE", "VAE",
                                    "GS-0.85", "Sgeo", "CC-SG"]}, r4)
    # lineage sensitivity: drop tasks sharing parent rows with primary data
    for tag, drop in (("external", ["car_eval_34"]), ("real", ["wine_quality_x", "abalone_x"])):
        long, fam, r4 = S[tag]
        S[tag + "_indep"] = (long[~long.dataset.isin(drop)], fam, r4[~r4.dataset.isin(drop)])
    return S


def holm(pvals):
    keys = sorted(pvals, key=lambda k: (np.nan_to_num(pvals[k], nan=1.0)))
    m, prev, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        pv = 1.0 if np.isnan(pvals[k]) else pvals[k]
        adj = min(1.0, max(prev, (m - i) * pv)); out[k] = adj; prev = adj
    return out


def wil(a, b):
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        return np.nan


def hodges_lehmann(d):
    d = np.asarray(d); n = len(d)
    w = [(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)]
    return float(np.median(w))


def boot_ci(d, B=10000, seed=0):
    rng = np.random.RandomState(seed)
    d = np.asarray(d)
    idx = rng.randint(len(d), size=(B, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    S = suite_tables()
    # ── reproduction check: rev4 re-runs vs published CSVs ────────────────
    p("== Reproduction check (rev4 re-run minus published, mean over datasets; max |diff|)")
    rep_rows = []
    pubs = {"primary": [("results_xai_v5.csv", ["Base", "Sgeo", "CC-SG"], ["RF", "XGB", "LR"]),
                        ("results_rev2_fallback.csv", ["CC-strict"], ["RF"])],
            "external": [("results_rev_external.csv", ["Base", "Sgeo", "CC-SG"], ["RF"])],
            "real": [("results_rev3_real.csv", ["Base", "Sgeo", "CC-SG", "CC-strict"], ["RF"])],
            "synth": [("results_rev3_synth.csv", ["Base", "Sgeo", "CC-SG", "CC-strict"], ["RF"])]}
    for suite, lst in pubs.items():
        r4 = S[suite][2]
        for f, ms, clfs in lst:
            pub = rd(f)
            for clf in clfs:
                for m in ms:
                    if m not in piv(r4, clf, "mcc").index:
                        continue
                    for met in METR4:
                        a = piv(r4, clf, met).loc[m]; b = piv(pub, clf, met).loc[m][a.index]
                        rep_rows.append(dict(suite=suite, src=f, clf=clf, method=m, metric=met,
                                             max_abs_diff=float((a - b).abs().max())))
    rr = pd.DataFrame(rep_rows); rr.to_csv(os.path.join(RES, "rev4_task1_repro.csv"), index=False)
    p(rr.groupby(["suite", "src", "method"]).max_abs_diff.max().to_string())

    # ── Task 1: means table + Holm ───────────────────────────────────────
    hrows, mrows = [], []
    for suite, (long, fams, _) in S.items():
        clfs = ["RF", "XGB", "LR"] if suite == "primary" else ["RF"]
        mets = METR5 if suite == "primary" else METR4
        for clf in clfs:
            for met in mets:
                P = piv(long, clf, met)
                if REF not in P.index:
                    continue
                for m in P.index:
                    mrows.append(dict(suite=suite, clf=clf, metric=met, method=m,
                                      mean_over_datasets=P.loc[m].mean(),
                                      n_datasets=int(P.loc[m].notna().sum())))
                for fname, rivals in fams.items():
                    rivals = [r for r in rivals if r in P.index]
                    if clf != "RF" and fname != "A_main":
                        continue
                    if clf != "RF":   # XGB/LR: only rivals that exist for that clf
                        rivals = [r for r in rivals if P.loc[r].notna().all()]
                    raw = {}
                    for r in rivals:
                        a, b = P.loc[REF], P.loc[r]
                        ok = a.notna() & b.notna()
                        raw[r] = wil(a[ok].values, b[ok].values)
                    adj = holm(raw)
                    for r in rivals:
                        a, b = P.loc[REF], P.loc[r]; ok = a.notna() & b.notna()
                        d = (a - b)[ok]
                        hrows.append(dict(suite=suite, clf=clf, metric=met, family=fname,
                                          n_tests=len(rivals), rival=r, n_datasets=int(ok.sum()),
                                          mean_diff=d.mean(), wins=int((d > 1e-9).sum()),
                                          ties=int((d.abs() <= 1e-9).sum()),
                                          losses=int((d < -1e-9).sum()),
                                          p_raw=raw[r], p_holm=adj[r]))
    H = pd.DataFrame(hrows); H.to_csv(os.path.join(RES, "rev4_task1_holm.csv"), index=False)
    pd.DataFrame(mrows).to_csv(os.path.join(RES, "rev4_task1_means.csv"), index=False)
    p("\n== Task 1: CC-strict vs rivals (Wilcoxon over datasets, Holm within suite x clf x metric x family)")
    for (suite, clf, fam), g in H.groupby(["suite", "clf", "family"], sort=False):
        p(f"\n-- {suite} / {clf} / family {fam} (m = {g.n_tests.iloc[0]} tests per metric)")
        t = g.pivot_table(index="rival", columns="metric",
                          values=["mean_diff", "p_holm"], aggfunc="first")
        cells = {}
        for r in g.rival.unique():
            row = []
            for met in g.metric.unique():
                x = g[(g.rival == r) & (g.metric == met)].iloc[0]
                row.append(f"{x.mean_diff:+.3f} ({x.wins}/{x.n_datasets}, {x.p_holm:.3f}{'*' if x.p_holm < .05 else ''})")
            cells[r] = row
        p(pd.DataFrame(cells, index=g.metric.unique()).T.to_string())

    # ── Task 1/2: quota statistics ────────────────────────────────────────
    qrows = []
    for suite in ("primary", "external", "real", "synth"):
        q = rd(f"rev4_{suite}_quota.csv")
        q = q[q.method.isin(["CC-SG", "CC-strict", "CC-batch"])].copy()
        for (m, dsn), g in q.groupby(["method", "dataset"]):
            nb = g.n_bnd.sum()
            if nb == 0:
                continue
            qrows.append(dict(suite=suite, method=m, dataset=dsn, n_bnd=nb,
                              frac_inloop=g.n_inloop.sum() / nb,
                              frac_fallback_delivered=(g.n_bnd_delivered.sum() - g.n_inloop.sum()) / nb,
                              frac_core_replaced=g.n_core_repl.sum() / nb,
                              fallback_fail_rate=(g.n_fb.sum() - g.n_fb_pass.sum()) / max(g.n_fb.sum(), 1),
                              final_viol_rate_delivered_boundary=g.n_bnd_delivered_viol.sum() / max(g.n_bnd_delivered.sum(), 1),
                              viol_rate_core_replacements=g.n_core_repl_viol.sum() / max(g.n_core_repl.sum(), 1),
                              proposals_per_quota=g.n_proposals.sum() / nb,
                              mean_rounds=g.rounds.mean(),
                              frac_calls_needing_fallback=(g.n_fb > 0).mean()))
    Q = pd.DataFrame(qrows); Q.to_csv(os.path.join(RES, "rev4_task1_quota.csv"), index=False)
    p("\n== Quota / violation statistics (pooled over 15 folds x minority classes, per dataset; then mean over datasets)")
    cols = ["frac_inloop", "frac_fallback_delivered", "frac_core_replaced",
            "final_viol_rate_delivered_boundary", "viol_rate_core_replacements",
            "proposals_per_quota", "frac_calls_needing_fallback"]
    p(Q.groupby(["suite", "method"])[cols].mean().round(4).to_string())
    p("\nmax final violation rate of delivered boundary set, CC-strict / CC-batch (should be 0):",
      Q[Q.method != "CC-SG"].final_viol_rate_delivered_boundary.max())
    p("\nCC-strict per dataset:")
    p(Q[Q.method == "CC-strict"][["suite", "dataset"] + cols].round(3).to_string(index=False))

    # ── Task 2: CC-batch vs CC-strict vs GeoB ─────────────────────────────
    p("\n== Task 2: matched generate-filter-refill control (RF)")
    t2 = []
    pairs = [("CC-batch", "CC-strict"), ("CC-batch", "Sgeo"), ("CC-strict", "Sgeo"),
             ("GeoF", "CC-strict"), ("GeoF", "Sgeo"), ("CC-batch", "CC-SG")]
    pooled = {}
    for suite in ("primary", "real", "synth", "external"):
        r4 = S[suite][2]
        for met in METR4:
            P = piv(r4, "RF", met)
            for a, b in pairs:
                d = P.loc[a] - P.loc[b]
                pooled.setdefault((a, b, met), []).extend(list(d.values) if suite != "external" else [])
                lo, hi = boot_ci(d.values)
                t2.append(dict(suite=suite, metric=met, A=a, B=b, n=len(d),
                               mean_A=P.loc[a].mean(), mean_B=P.loc[b].mean(),
                               mean_diff=d.mean(), hl=hodges_lehmann(d.values),
                               ci_lo=lo, ci_hi=hi, wins=int((d > 1e-9).sum()),
                               losses=int((d < -1e-9).sum()), p_wilcoxon=wil(P.loc[a].values, P.loc[b].values)))
    for (a, b, met), d in pooled.items():
        d = np.array(d); lo, hi = boot_ci(d)
        t2.append(dict(suite="pooled_primary+real+synth", metric=met, A=a, B=b, n=len(d),
                       mean_diff=d.mean(), hl=hodges_lehmann(d), ci_lo=lo, ci_hi=hi,
                       wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()),
                       p_wilcoxon=float(wilcoxon(d).pvalue) if not np.allclose(d, 0) else 1.0))
    T2 = pd.DataFrame(t2); T2.to_csv(os.path.join(RES, "rev4_task2_batch.csv"), index=False)
    for (a, b), g in T2.groupby(["A", "B"], sort=False):
        p(f"\n-- {a} minus {b}")
        p(g[["suite", "metric", "n", "mean_diff", "hl", "ci_lo", "ci_hi", "wins", "losses", "p_wilcoxon"]]
          .round(4).to_string(index=False))

    # ── Task 4: bootstrap CI + HL, CC-strict vs each rival, RF ─────────────
    t4 = []
    for suite, (long, fams, _) in S.items():
        rivals = list(dict.fromkeys(sum(fams.values(), []))) + ["CC-batch", "GeoF"]
        for met in METR4:
            P = piv(long, "RF", met)
            for r in rivals:
                if r not in P.index:
                    continue
                a, b = P.loc[REF], P.loc[r]; ok = a.notna() & b.notna()
                d = (a - b)[ok].values
                lo, hi = boot_ci(d)
                t4.append(dict(suite=suite, metric=met, rival=r, n_datasets=len(d),
                               mean_diff=d.mean(), boot_ci_lo=lo, boot_ci_hi=hi,
                               hodges_lehmann=hodges_lehmann(d), median_diff=np.median(d),
                               ci_excludes_0=bool(lo > 0 or hi < 0)))
    T4 = pd.DataFrame(t4); T4.to_csv(os.path.join(RES, "rev4_task4_bootstrap.csv"), index=False)
    p("\n== Task 4: CC-strict minus rival, RF; mean [95% bootstrap CI over datasets] / HL")
    for suite, g in T4.groupby("suite", sort=False):
        p(f"\n-- {suite}")
        cells = {}
        for r in g.rival.unique():
            cells[r] = [f"{x.mean_diff:+.3f} [{x.boot_ci_lo:+.3f},{x.boot_ci_hi:+.3f}]{'*' if x.ci_excludes_0 else ''} HL{x.hodges_lehmann:+.3f}"
                        for _, x in g[g.rival == r].iterrows()]
        p(pd.DataFrame(cells, index=METR4).T.to_string())
    open(os.path.join(RES, "rev4_stats_log.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    main()
