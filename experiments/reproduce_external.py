"""Reproduce the external validations (paper Tables 7-10, Section 4.6-4.7).

Evaluates the frozen CC-SG configuration and the key rivals on three unseen
suites: 10 moderate-imbalance datasets, 9 real extreme-imbalance datasets, and
9 controlled synthetic datasets with noise-free labels.

    python experiments/reproduce_external.py
"""
import os
import sys

import pandas as pd
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccsg import CCSG, calkde, geob, smote_ipf, mwmote, vae_oversample  # noqa
from ccsg.datasets import (load_external_moderate, load_external_extreme,      # noqa
                           make_synthetic_extreme)
from ccsg.evaluation import run_full, no_augmentation                   # noqa
from experiments.reproduce_primary import _combine                      # noqa
from imblearn.combine import SMOTEENN

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def _kmin(y):
    return int(pd.Series(y).value_counts().min())


def _smote(X, y, seed=42):
    k = max(1, min(5, _kmin(y) - 1))
    return SMOTE(k_neighbors=k, random_state=seed).fit_resample(X, y)


METHODS = {
    "Base":        no_augmentation,
    "SMOTE":       _smote,
    "SMOTE-ENN":   _combine(SMOTEENN),
    "SMOTE-IPF":   smote_ipf,
    "MWMOTE":      mwmote,
    "VAE":         vae_oversample,
    "CalKDE":      calkde,
    "GeoB":        geob,
    "CC-SG":       lambda X, y, seed=42: CCSG(random_state=seed).fit_resample(X, y),
    "CC-SG-strict": lambda X, y, seed=42: CCSG(strict=True, random_state=seed).fit_resample(X, y),
}


def _run(datasets, tag):
    df = run_full(datasets, METHODS, classifiers=["RF"], seed=42)
    df2 = run_full(datasets, {"Base": no_augmentation},
                   classifiers=["RF-W", "RF-THR"], seed=42)
    df2["method"] = df2["clf"].map({"RF-W": "Weighted", "RF-THR": "ThreshMove"})
    df2["clf"] = "RF"
    out = pd.concat([df, df2[df2.method.notna()]], ignore_index=True)
    path = os.path.join(OUT, f"external_{tag}.csv")
    out.to_csv(path, index=False)
    print("saved", path)


def main():
    _run(load_external_moderate(), "moderate")
    _run(load_external_extreme(), "extreme_real")
    _run(make_synthetic_extreme(), "synthetic")


if __name__ == "__main__":
    main()
