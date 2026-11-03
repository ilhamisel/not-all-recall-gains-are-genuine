#!/usr/bin/env python3
"""Revizyon deney (a) — Roadmap #2 + #3: uret-sonra-temizle akrabalari
(SMOTE-ENN, SMOTE-Tomek) ve maliyet-duyarli agirlikli siniflandiricilar,
orijinal 10 veri kumesi, v5 protokolu (5x3, seed=42, fold-ici olcekleme).
Cikti: results_rev_baselines.csv (v5 ile birlestirilebilir)."""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE

import benchmark_math_models as B
import eval_framework as E
from run_importance_v3 import load_all

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def _kmin(y):
    return int(pd.Series(y).value_counts().min())


def _safe_smote(y, seed):
    k = max(1, min(5, _kmin(y) - 1))
    return SMOTE(k_neighbors=k, random_state=seed)


def smote_enn(X, y, seed=42):
    Xr, yr = SMOTEENN(random_state=seed,
                      smote=_safe_smote(y, seed)).fit_resample(X, y)
    return np.asarray(Xr), np.asarray(yr)


def smote_tomek(X, y, seed=42):
    Xr, yr = SMOTETomek(random_state=seed,
                        smote=_safe_smote(y, seed)).fit_resample(X, y)
    return np.asarray(Xr), np.asarray(yr)


def main():
    t0 = time.time()
    datasets = load_all()
    dfs = []
    # (a1) uret-sonra-temizle hibritleri, 3 siniflandirici
    df1 = E.run_full(datasets,
                     {"Base": B.augment_base,
                      "SMOTE-ENN": smote_enn,
                      "SMOTE-Tomek": smote_tomek},
                     classifiers=["RF", "XGB", "LR"],
                     n_splits=5, n_repeats=3, seed=42, verbose=True, scale=True)
    dfs.append(df1)
    f1 = pd.DataFrame(E.LAST_FAILURES)
    # (a2) agirlikli siniflandiricilar (augmentasyon YOK)
    df2 = E.run_full(datasets, {"Base": B.augment_base},
                     classifiers=["RF-W", "XGB-W", "LR-W"],
                     n_splits=5, n_repeats=3, seed=42, verbose=True, scale=True)
    df2 = df2.copy()
    df2["method"] = "Weighted"          # tablolarda "Weighted clf" satiri
    dfs.append(df2)
    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(os.path.join(OUT, "results_rev_baselines.csv"), index=False)
    f1.to_csv(os.path.join(OUT, "failures_rev_baselines.csv"), index=False)
    print(f"\n[OK] results_rev_baselines.csv  {(time.time()-t0)/60:.1f} dk")


if __name__ == "__main__":
    main()
