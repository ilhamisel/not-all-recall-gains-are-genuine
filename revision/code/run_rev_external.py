#!/usr/bin/env python3
"""Revizyon deney (c) — Roadmap #1: DIS DOGRULAMA.
CC-SG konfigurasyonu DONDURULMUS halde (s=0.85, beta=0.5, U(0.25,0.75),
tau=0.5, XGB-30 verici) hicbir tasarim kararinda kullanilmamis 10 veri
kumesinde kosulur. Hicbir hiperparametre bu kumelere gore ayarlanmamistir.
Kumeler: imblearn fetch_datasets'ten, orijinal suitle kesisimsiz.
Cikti: results_rev_external.csv"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.datasets import fetch_datasets

import benchmark_math_models as B
import eval_framework as E
from run_importance_v3b import GS
import augment_xai_v4 as X4
from run_rev_baselines import smote_enn, smote_tomek
from run_literature import wrap
from imblearn.over_sampling import SMOTE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

EXTERNAL = ["satimage", "sick_euthyroid", "spectrometer", "car_eval_34",
            "libras_move", "thyroid_sick", "solar_flare_m0", "yeast_ml8",
            "coil_2000", "arrhythmia"]

CCSG = lambda X, y, seed=42: X4.augment_xsemi(
    X, y, mode="cfgate", frac=0.5, depth_lo=0.25, depth_hi=0.75,
    p_thr=0.5, seed=seed)

METHODS = {
    "Base":        B.augment_base,
    "SMOTE":       wrap(SMOTE),
    "SMOTE-ENN":   smote_enn,
    "SMOTE-Tomek": smote_tomek,
    "GS-0.85":     GS(0.85),
    "Sgeo":        lambda X, y, seed=42: X4.augment_xsemi(
        X, y, mode="geo", frac=0.5, depth_lo=0.25, depth_hi=0.75, seed=seed),
    "CC-SG":       CCSG,
}


def load_external():
    raw = fetch_datasets(filter_data=tuple(EXTERNAL))
    d = {}
    for n in EXTERNAL:
        d[n] = (np.asarray(raw[n].data, float),
                LabelEncoder().fit_transform(raw[n].target))
    return d


def main():
    t0 = time.time()
    datasets = load_external()
    dfs = []
    df1 = E.run_full(datasets, METHODS, classifiers=["RF"],
                     n_splits=5, n_repeats=3, seed=42, verbose=True, scale=True)
    dfs.append(df1)
    fails = pd.DataFrame(E.LAST_FAILURES)
    # agirlikli RF taban cizgisi
    df2 = E.run_full(datasets, {"Base": B.augment_base}, classifiers=["RF-W"],
                     n_splits=5, n_repeats=3, seed=42, verbose=True, scale=True)
    df2 = df2.copy(); df2["method"] = "Weighted"
    dfs.append(df2)
    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(os.path.join(OUT, "results_rev_external.csv"), index=False)
    fails.to_csv(os.path.join(OUT, "failures_rev_external.csv"), index=False)
    print(f"\n[OK] results_rev_external.csv  {(time.time()-t0)/60:.1f} dk")


if __name__ == "__main__":
    main()
