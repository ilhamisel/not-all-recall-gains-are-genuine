#!/usr/bin/env python3
"""Rev.5 analysis.
(1) Corrected SMOTE-IPF (rev5_ipf) replaces the rev3 SMOTE-IPF everywhere
    (old one kept only as 'SMOTE-IPF-old' for the comparison table).
(2) Non-independent tasks flagged: wine_quality_x (= Wine_White rows),
    abalone_x (= abalone_19 rows), car_eval_34 (= car_eval_4 rows).
    Every external/extreme summary is recomputed (a) on all tasks and
    (b) excluding the flagged tasks.  CC-strict is the reference ("CC-SG (ours)").
Statistics (RF, per-dataset means of 5x3 CV):
  dataset-level two-sided Wilcoxon signed-rank CC-strict vs rival (scipy
  default; exactly-zero differences -> p=1), Holm within each
  (suite, subset, metric) family; family size m = number of rivals listed;
  paired bootstrap 95% percentile CI (10,000 resamples of datasets, seed 0)
  for mean(CC-strict - rival); Hodges-Lehmann estimate.
Helper functions (holm, wil, boot_ci, hodges_lehmann, piv, suite_tables) are
imported unchanged from rev4_stats.
Outputs (results/): rev5_means.csv, rev5_holm.csv, rev5_bootstrap.csv,
  rev5_ipf_vs_smote.csv, rev5_ipf_removal_summary.csv, rev5_stats_log.txt,
  rev5_tables.tex"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from rev4_stats import (holm, wil, boot_ci, hodges_lehmann, piv, suite_tables,
                        rd, RES)

LOG = []
REF = "CC-strict"
FLAG = {"external": ["car_eval_34"], "real": ["wine_quality_x", "abalone_x"],
        "synth": []}
FLAG_PARENT = {"car_eval_34": "car_eval_4", "wine_quality_x": "Wine_White",
               "abalone_x": "abalone_19"}
M4 = ["min_recall", "gmean", "f1_macro", "mcc"]
M5 = ["min_recall", "rare_recall", "gmean", "f1_macro", "mcc"]
FAMS = {
    "external": ["Base", "Weighted", "SMOTE", "SMOTE-ENN", "SMOTE-Tomek",
                 "SMOTE-IPF", "GS-0.85", "Sgeo", "CC-SG"],
    "real": ["Base", "Weighted", "ThreshMove", "SMOTE", "SMOTE-ENN",
             "SMOTE-IPF", "MWMOTE", "VAE", "GS-0.85", "Sgeo", "CC-SG"],
    "synth": ["Base", "Weighted", "ThreshMove", "SMOTE", "SMOTE-ENN",
              "SMOTE-IPF", "MWMOTE", "VAE", "GS-0.85", "Sgeo", "CC-SG"],
    "primary": ["Base", "Weighted", "SMOTE", "Borderline", "ADASYN",
                "SMOTE-ENN", "SMOTE-Tomek", "GS-0.85", "Sgeo", "CC-SG",
                "SMOTE-IPF", "MWMOTE", "VAE", "ThreshMove", "ROS",
                "SVM-SMOTE", "KMeans-SMOTE", "PlainKDE", "GMM-S",
                "BalancedRF"]}
METS = {"external": M5, "real": M4, "synth": M4, "primary": M5}
LABEL = {"Base": "Base", "Weighted": "Weighted clf.", "ThreshMove": "ThreshMove",
         "SMOTE": "SMOTE", "SMOTE-ENN": "SMOTE-ENN", "SMOTE-Tomek": "SMOTE-Tomek",
         "SMOTE-IPF": "SMOTE-IPF", "MWMOTE": "MWMOTE", "VAE": "VAE",
         "GS-0.85": "CalKDE", "Sgeo": "GeoB", "CC-SG": "CC-SG (non-strict)",
         "CC-strict": "CC-SG (ours)", "ROS": "ROS", "SVM-SMOTE": "SVM-SMOTE",
         "KMeans-SMOTE": "KMeans-SMOTE", "PlainKDE": "Plain KDE",
         "GMM-S": "GMM sampling", "BalancedRF": "Balanced RF"}
MLAB = {"min_recall": "Min.\\ recall", "rare_recall": "Rarest recall",
        "gmean": "G-mean", "f1_macro": "Macro-F1", "mcc": "MCC"}


def p(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


def new_ipf(suite):
    d = rd(f"rev5_ipf_{suite}.csv")
    return d[d.method == "SMOTE-IPF"].copy()


def assemble():
    S0 = suite_tables()
    S = {}
    for suite in ("primary", "external", "real", "synth"):
        long = S0[suite][0]
        long = long[long.clf == "RF"].copy()
        long.loc[long.method == "SMOTE-IPF", "method"] = "SMOTE-IPF-old"
        parts = [long, new_ipf(suite)]
        fc = os.path.join(RES, f"rev5_ipfcons_{suite}.csv")
        if os.path.exists(fc):          # sensitivity: consensus voting scheme
            c = pd.read_csv(fc); c = c[c.method == "SMOTE-IPF"].copy()
            c["method"] = "SMOTE-IPF-cons"; parts.append(c)
        S[suite] = pd.concat(parts, ignore_index=True)
    return S


def stats_block(long, suite, subset, rows_m, rows_h, rows_b):
    rivals = FAMS[suite]
    for met in METS[suite]:
        P = piv(long, "RF", met)
        for m in P.index:
            rows_m.append(dict(suite=suite, subset=subset, metric=met, method=m,
                               mean=P.loc[m].mean(),
                               n_datasets=int(P.loc[m].notna().sum())))
        riv = [r for r in rivals if r in P.index]
        raw = {}
        for r in riv:
            a, b = P.loc[REF], P.loc[r]; ok = a.notna() & b.notna()
            raw[r] = wil(a[ok].values, b[ok].values)
        adj = holm(raw)
        for r in riv:
            a, b = P.loc[REF], P.loc[r]; ok = a.notna() & b.notna()
            d = (a - b)[ok].values
            lo, hi = boot_ci(d)
            rows_h.append(dict(suite=suite, subset=subset, metric=met,
                               family_size=len(riv), rival=r, n_datasets=len(d),
                               mean_ref=a[ok].mean(), mean_rival=b[ok].mean(),
                               mean_diff=d.mean(), wins=int((d > 1e-9).sum()),
                               ties=int((np.abs(d) <= 1e-9).sum()),
                               losses=int((d < -1e-9).sum()),
                               p_raw=raw[r], p_holm=adj[r]))
            rows_b.append(dict(suite=suite, subset=subset, metric=met, rival=r,
                               n_datasets=len(d), mean_diff=d.mean(),
                               boot_ci_lo=lo, boot_ci_hi=hi,
                               hodges_lehmann=hodges_lehmann(d),
                               ci_excludes_0=bool(lo > 0 or hi < 0)))


def tex_body(M, suite, subset, methods, title, relabel=None):
    lab = dict(LABEL, **(relabel or {}))
    mets = METS[suite]
    g = M[(M.suite == suite) & (M.subset == subset)]
    T = g.pivot_table(index="method", columns="metric", values="mean")
    methods = [m for m in methods if m in T.index]
    best = {met: T.loc[methods, met].max() for met in mets}
    out = [f"% {title}",
           "% Method & " + " & ".join(MLAB[m] for m in mets) + " \\\\"]
    for m in methods:
        cells = []
        for met in mets:
            v = T.loc[m, met]; s = f"{v:.3f}"
            if round(v, 3) == round(best[met], 3):
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        out.append(f"{lab[m]} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def main():
    S = assemble()
    rm, rh, rb = [], [], []
    for suite in ("primary", "external", "real", "synth"):
        long = S[suite]
        stats_block(long, suite, "all", rm, rh, rb)
        if suite != "primary":
            stats_block(long[~long.dataset.isin(FLAG[suite])], suite,
                        "excl_flagged", rm, rh, rb)
    M, H, Bt = pd.DataFrame(rm), pd.DataFrame(rh), pd.DataFrame(rb)
    M.to_csv(os.path.join(RES, "rev5_means.csv"), index=False)
    H.to_csv(os.path.join(RES, "rev5_holm.csv"), index=False)
    Bt.to_csv(os.path.join(RES, "rev5_bootstrap.csv"), index=False)

    # ── IPF: removal + new vs old vs SMOTE ────────────────────────────────
    p("== Corrected SMOTE-IPF: removal statistics (per augmentation call = fold)")
    rem = []
    for suite in ("primary", "external", "real", "synth"):
        L = rd(f"rev5_ipf_{suite}_removal.csv")
        L["frac_removed_of_augmented"] = L.n_removed_total / L.n_after_smote
        L["frac_orig_removed"] = L.n_removed_original / L.n_orig
        L["n_synth"] = L.n_after_smote - L.n_orig
        L["frac_synth_removed"] = L.n_removed_synth / L.n_synth.clip(lower=1)
        g = L.groupby("dataset").agg(n_calls=("n_orig", "size"), n_orig=("n_orig", "mean"),
                                     n_after_smote=("n_after_smote", "mean"),
                                     removed_total=("n_removed_total", "mean"),
                                     removed_original=("n_removed_original", "mean"),
                                     removed_synth=("n_removed_synth", "mean"),
                                     frac_removed_of_augmented=("frac_removed_of_augmented", "mean"),
                                     frac_orig_removed=("frac_orig_removed", "mean"),
                                     frac_synth_removed=("frac_synth_removed", "mean"),
                                     iterations=("iterations", "mean")).reset_index()
        g.insert(0, "suite", suite); rem.append(g)
        p(f"\n-- {suite}: mean over folds per dataset")
        p(g.round(3).to_string(index=False))
        p(f"   suite mean over datasets: removed {g.removed_total.mean():.1f} samples/fold "
          f"({100*g.frac_removed_of_augmented.mean():.1f}% of the SMOTE-augmented set; "
          f"{g.removed_original.mean():.1f} original = {100*g.frac_orig_removed.mean():.1f}% of real rows, "
          f"{g.removed_synth.mean():.1f} synthetic = {100*g.frac_synth_removed.mean():.1f}% of synthetic rows), "
          f"{g.iterations.mean():.1f} iterations")
    R = pd.concat(rem); R.to_csv(os.path.join(RES, "rev5_ipf_removal_summary.csv"), index=False)

    p("\n== SMOTE-IPF: corrected vs rev3 (old) vs SMOTE, RF, mean over datasets")
    cmp_rows = []
    for suite in ("primary", "external", "real", "synth"):
        long = S[suite]
        r5 = rd(f"rev5_ipf_{suite}.csv")
        for met in METS[suite]:
            P = piv(long, "RF", met)
            Q = piv(r5, "RF", met)
            rep_smote = float((Q.loc["SMOTE"] - P.loc["SMOTE"][Q.columns]).abs().max()) \
                if "SMOTE" in P.index else np.nan
            row = dict(suite=suite, metric=met, SMOTE=P.loc["SMOTE"].mean() if "SMOTE" in P.index else Q.loc["SMOTE"].mean(),
                       IPF_old=P.loc["SMOTE-IPF-old"].mean() if "SMOTE-IPF-old" in P.index else np.nan,
                       IPF_new=P.loc["SMOTE-IPF"].mean(),
                       smote_rerun_max_abs_diff_vs_published=rep_smote)
            d = (Q.loc["SMOTE-IPF"] - Q.loc["SMOTE"]).values
            row.update(new_minus_smote=d.mean(), max_abs_new_minus_smote=np.abs(d).max(),
                       wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()),
                       p_wilcoxon=wil(Q.loc["SMOTE-IPF"].values, Q.loc["SMOTE"].values))
            cmp_rows.append(row)
    C = pd.DataFrame(cmp_rows); C.to_csv(os.path.join(RES, "rev5_ipf_vs_smote.csv"), index=False)
    p(C.round(4).to_string(index=False))

    # ── recomputed summaries ──────────────────────────────────────────────
    for suite in ("external", "real", "synth", "primary"):
        for subset in (("all", "excl_flagged") if suite != "primary" else ("all",)):
            g = H[(H.suite == suite) & (H.subset == subset)]
            b = Bt[(Bt.suite == suite) & (Bt.subset == subset)]
            mm = M[(M.suite == suite) & (M.subset == subset)]
            n = int(mm.n_datasets.max())
            fl = FLAG.get(suite, [])
            p(f"\n== {suite} / {subset} (n = {n} datasets"
              + (f"; excluded {fl}" if subset != "all" and fl else "") + ")")
            T = mm.pivot_table(index="method", columns="metric", values="mean")[METS[suite]]
            order = [m for m in FAMS[suite] + [REF, "CC-batch", "GeoF", "SMOTE-IPF-old", "SMOTE-IPF-cons"] if m in T.index]
            p("RF means:"); p(T.loc[order].round(3).to_string())
            p(f"CC-strict minus rival: mean diff (W/T/L) Holm p [family size m = {g.family_size.iloc[0]} per metric]; "
              f"95% paired bootstrap CI")
            cells = {}
            for r in g.rival.unique():
                row = []
                for met in METS[suite]:
                    x = g[(g.rival == r) & (g.metric == met)].iloc[0]
                    y = b[(b.rival == r) & (b.metric == met)].iloc[0]
                    row.append(f"{x.mean_diff:+.3f} ({x.wins}/{x.ties}/{x.losses}) p={x.p_holm:.3f}"
                               f"{'*' if x.p_holm < .05 else ''} [{y.boot_ci_lo:+.3f},{y.boot_ci_hi:+.3f}]"
                               f"{'#' if y.ci_excludes_0 else ''}")
                cells[r] = row
            p(pd.DataFrame(cells, index=METS[suite]).T.to_string())
    p("\n(* Holm-adjusted p < 0.05; # bootstrap CI excludes 0)")
    p("\n== Sensitivity: SMOTE-IPF with CONSENSUS voting (not in Holm families); "
      "CC-strict minus IPF-cons, mean diff (W/L) raw Wilcoxon p [95% bootstrap CI]")
    for suite in ("primary", "external", "real", "synth"):
        long = S[suite]
        if "SMOTE-IPF-cons" not in long.method.unique():
            continue
        for subset in (("all", "excl_flagged") if suite in ("external", "real") else ("all",)):
            L = long if subset == "all" else long[~long.dataset.isin(FLAG[suite])]
            p(f"-- {suite}/{subset}")
            for met in METS[suite]:
                P = piv(L, "RF", met)
                a, c = P.loc[REF], P.loc["SMOTE-IPF-cons"]
                d = (a - c).values; lo, hi = boot_ci(d)
                p(f"   {met:11s} IPF-maj={P.loc['SMOTE-IPF'].mean():.3f} IPF-cons={c.mean():.3f} "
                  f"SMOTE={P.loc['SMOTE'].mean():.3f}  CC-strict-IPFcons={d.mean():+.3f} "
                  f"({int((d > 1e-9).sum())}/{int((d < -1e-9).sum())}) p_raw={wil(a.values, c.values):.3f} "
                  f"[{lo:+.3f},{hi:+.3f}]")
        Lr = rd(f"rev5_ipfcons_{suite}_removal.csv")
        p(f"   IPF-cons removal ({suite}): {Lr.n_removed_total.mean():.1f} samples/fold, "
          f"{100 * (Lr.n_removed_total / Lr.n_after_smote).mean():.1f}% of augmented set, "
          f"{100 * (Lr.n_removed_original / Lr.n_orig).mean():.1f}% of real rows")

    # ── LaTeX table bodies ────────────────────────────────────────────────
    dag = "$^\\dagger$"
    tex = ["% rev5_tables.tex -- generated by rev5_stats.py. RF averages over datasets.",
           "% CC-SG (ours) = CC-strict; CC-SG (non-strict) = original CC-SG.",
           "% SMOTE-IPF = corrected re-implementation (rev5_ipf.py; Saez et al. 2015 defaults).",
           "% Bold = best value per column among the listed rows.", ""]
    ext_rows = ["Base", "Weighted", "SMOTE", "SMOTE-ENN", "SMOTE-Tomek", "SMOTE-IPF",
                "GS-0.85", "Sgeo", "CC-SG", "CC-strict"]
    xr_rows = ["Base", "Weighted", "ThreshMove", "SMOTE", "SMOTE-ENN", "SMOTE-IPF",
               "MWMOTE", "VAE", "GS-0.85", "Sgeo", "CC-SG", "CC-strict"]
    xt_rows = ["ROS", "SVM-SMOTE", "KMeans-SMOTE", "PlainKDE", "GMM-S", "SMOTE-IPF",
               "MWMOTE", "VAE", "ThreshMove", "BalancedRF", "CC-SG", "CC-strict"]
    tex.append("%%%% tab:external -- (a) all 10 datasets")
    tex.append(tex_body(M, "external", "all", ext_rows, "tab:external, all 10 datasets"))
    tex.append("% Note: \\textsuperscript{\\dagger}car\\_eval\\_34 shares 100\\% of its rows with the primary task "
               "car\\_eval\\_4 (its 134 minority rows contain all 65 car\\_eval\\_4 minority rows); "
               "it is kept in the suite but is not independent.")
    tex.append("% Suggested caption text: \"... 10 datasets (car\\_eval\\_34" + dag + ") ...\"\n")
    tex.append("%%%% tab:external -- (b) sensitivity: 9 datasets, excluding car_eval_34" + "")
    tex.append(tex_body(M, "external", "excl_flagged", ext_rows, "tab:external, excluding flagged (n=9)"))
    tex.append("")
    tex.append("%%%% tab:extreme_real -- (a) all 9 datasets")
    tex.append(tex_body(M, "real", "all", xr_rows, "tab:extreme_real, all 9 datasets"))
    tex.append("% Note: \\textsuperscript{\\dagger}wine\\_quality\\_x uses 100\\% rows of primary Wine\\_White "
               "(minority = quality $\\le$4); \\textsuperscript{\\dagger}abalone\\_x uses 100\\% rows of primary "
               "abalone\\_19 (different label). Both kept in the suite but not independent.\n")
    tex.append("%%%% tab:extreme_real -- (b) sensitivity: 7 datasets, excluding wine_quality_x and abalone_x")
    tex.append(tex_body(M, "real", "excl_flagged", xr_rows, "tab:extreme_real, excluding flagged (n=7)"))
    tex.append("")
    tex.append("%%%% tab:extreme_synth -- 9 synthetic datasets (no flagged task; (a) == (b))")
    tex.append(tex_body(M, "synth", "all", xr_rows, "tab:extreme_synth, 9 datasets"))
    tex.append("")
    tex.append("%%%% tab:extended -- primary suite, 10 datasets, extended rival set")
    tex.append(tex_body(M, "primary", "all", xt_rows, "tab:extended, 10 primary datasets",
                        relabel={"VAE": "VAE (deep gen.)", "ThreshMove": "Threshold-moving"}))
    tex.append("% Note: SMOTE-IPF row = corrected implementation; the previously reported row was "
               "numerically ~identical to SMOTE because the earlier filter removed almost nothing.")
    open(os.path.join(RES, "rev5_tables.tex"), "w", encoding="utf-8").write("\n".join(tex) + "\n")
    open(os.path.join(RES, "rev5_stats_log.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50)
    main()
