#!/usr/bin/env python3
"""
Literatür Kıyaslaması — kanonik oversampling aileleri vs E-auto (SF-KDE)
════════════════════════════════════════════════════════════════════════
Referanslar:
  ROS            Random Over-Sampling (naif kopyalama)
  SMOTE          Chawla+ 2002 — doğrusal interpolasyon
  Borderline     Han+ 2005 — sınır-odaklı SMOTE
  ADASYN         He+ 2008 — yoğunluk-adaptif SMOTE
  SVM-SMOTE      Nguyen+ 2011 — SVM sınırında SMOTE
  C1-Filter      bizim GMM-filtre ailesi (GMF-SMOTE esinli)
  E-auto         önerilen nihai model (SF-KDE + otomatik kapı)

Ekstrem dengesizlikte SMOTE-ailesi k_neighbors dinamik ayarlanır.
4 set × RF+XGB × 3 tekrar. Çıktı: results/results_literature.csv,
results/LITERATURE.md, results/literature_figure.png
"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import benchmark_math_models as B
import augment_filtered as F
import eval_framework as E
from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, ADASYN,
                                    SVMSMOTE, RandomOverSampler)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
CORE = ["Date_Fruit", "Soybean_Diseases", "Wine_Red", "Wine_White"]


def _kmin(y):
    return int(pd.Series(y).value_counts().min())


def wrap(sampler_cls, needs_k=True, extra=None):
    """imblearn sampler'ı ortak (X, y, seed) imzasına sarar; k dinamik.
    Hatalar YUTULMAZ: istisna yukarı fırlatılır; eval_framework.run_full
    hatayı türü/mesajıyla LAST_FAILURES'a kaydeder ve Base'e döner."""
    def fn(X, y, seed=42):
        kw = dict(random_state=seed)
        if extra:
            kw.update(extra)
        if needs_k:
            k = max(1, min(5, _kmin(y) - 1))
            kw["k_neighbors"] = k
        Xr, yr = sampler_cls(**kw).fit_resample(X, y)
        return np.asarray(Xr), np.asarray(yr)
    return fn


METHODS = {
    "Base":       B.augment_base,
    "ROS":        wrap(RandomOverSampler, needs_k=False),
    "SMOTE":      wrap(SMOTE),
    "Borderline": wrap(BorderlineSMOTE),
    "ADASYN":     wrap(ADASYN, needs_k=False, extra=None),  # ADASYN: n_neighbors
    "SVM-SMOTE":  wrap(SVMSMOTE),
    "C1-Filter":  F.augment_C1_filter,
    "E-auto":     F.augment_E_auto,
}
# ADASYN k parametresi farklı isimde → özel sarım (hata yutulmaz)
def _adasyn(X, y, seed=42):
    k = max(1, min(5, _kmin(y) - 1))
    Xr, yr = ADASYN(random_state=seed, n_neighbors=k).fit_resample(X, y)
    return np.asarray(Xr), np.asarray(yr)
METHODS["ADASYN"] = _adasyn

CLFS = ["RF", "XGB"]


def main():
    t0 = time.time()
    datasets = {n: B.load_dataset(B.DATASETS[n]) for n in CORE}
    df = E.run_full(datasets, METHODS, classifiers=CLFS,
                    n_splits=5, n_repeats=3, seed=42, skip_large=None, verbose=True)
    p = os.path.join(OUT, "results_literature.csv")
    df.to_csv(p, index=False)
    print(f"\n[OK] {p}")
    write_md(df, os.path.join(OUT, "LITERATURE.md"))
    make_figure(df, os.path.join(OUT, "literature_figure.png"))
    print(f"Toplam: {(time.time()-t0)/60:.1f} dk")


def write_md(df, path):
    order = list(METHODS.keys())
    L = []; A = L.append
    A("# Literatür Kıyaslaması — Kanonik Oversampling vs E-auto (SF-KDE)\n")
    A("*ROS, SMOTE, Borderline-SMOTE, ADASYN, SVM-SMOTE = literatür; "
      "C1-Filter, E-auto = bizim aile*\n")
    A("- RF+XGB, 5-fold × 3 tekrar, Wilcoxon vs Base (`*` = p<0.05)\n")
    for clf in CLFS:
        A(f"\n## {clf}\n")
        for metric in ["min_recall", "gmean", "f1_macro"]:
            A(f"\n### {metric}\n")
            sub = df[(df.clf == clf) & (df.metric == metric)]
            piv = sub.pivot_table(index="method", columns="dataset",
                                  values="mean").reindex(order)[CORE]
            sig = sub.pivot_table(index="method", columns="dataset",
                                  values="wilcoxon_p").reindex(order)[CORE]
            piv["ORT"] = piv.mean(axis=1)
            A("| Yöntem | " + " | ".join(CORE) + " | ORT |")
            A("|" + "---|" * (len(CORE) + 2))
            for m in order:
                cells = []
                for ds in CORE:
                    star = "*" if (sig.loc[m, ds] < 0.05 and m != "Base") else ""
                    cells.append(f"{piv.loc[m, ds]:.4f}{star}")
                A(f"| {m} | " + " | ".join(cells) + f" | {piv.loc[m,'ORT']:.4f} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[OK] {path}")


def make_figure(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = list(METHODS.keys())
    lit = {"Base", "ROS", "SMOTE", "Borderline", "ADASYN", "SVM-SMOTE"}
    colors = ["#9aa0a6" if m == "Base" else
              ("#e07a1f" if m == "E-auto" else
               ("#7b3fa0" if m == "C1-Filter" else "#4895c9"))
              for m in order]
    metrics = [("min_recall", "Azınlık recall"), ("gmean", "Makro G-mean")]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey="row")
    for r, (metric, ml) in enumerate(metrics):
        for c, ds in enumerate(CORE):
            ax = axes[r, c]
            sub = df[(df.clf == "RF") & (df.metric == metric) & (df.dataset == ds)]
            vals = [float(sub[sub.method == m]["mean"].values[0]) for m in order]
            ax.bar(range(len(order)), vals, color=colors, edgecolor="white")
            for i, v in enumerate(vals):
                ax.text(i, v + max(vals) * 0.02, f"{v:.2f}", ha="center", fontsize=7.5)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
            if r == 0:
                vc = pd.Series(B.load_dataset(B.DATASETS[ds])[1]).value_counts()
                ax.set_title(f"{ds}\nIR={vc.max()/vc.min():.0f}", fontweight="bold")
            if c == 0:
                ax.set_ylabel(ml, fontweight="bold")
            ax.grid(axis="y", ls=":", alpha=0.4)
    fig.suptitle("Literatür oversampling yöntemleri vs E-auto (RF)  "
                 "[turuncu=önerilen, mor=GMM-filtre, mavi=SMOTE ailesi]",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[OK] {path}")


if __name__ == "__main__":
    main()
