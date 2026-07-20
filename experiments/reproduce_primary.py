"""Reproduce the primary-suite results (paper Tables 2-6, Fig. 4-8).

Runs CC-SG, its internal references (CalKDE, GeoB), the classical SMOTE family,
generate-then-clean hybrids, selective/deep-generative rivals, and the
cost-sensitive / threshold-moving baselines on the 10 primary datasets with
four downstream classifiers.

    python experiments/reproduce_primary.py
"""
import os
import sys

import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccsg import CCSG, calkde, geob, smote_ipf, mwmote, vae_oversample  # noqa
from ccsg.datasets import load_primary                                  # noqa
from ccsg.evaluation import run_full, no_augmentation                   # noqa

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


def _kmin(y):
    return int(pd.Series(y).value_counts().min())


def _smote(cls):
    def fn(X, y, seed=42):
        k = max(1, min(5, _kmin(y) - 1))
        Xr, yr = cls(k_neighbors=k, random_state=seed).fit_resample(X, y)
        return Xr, yr
    return fn


def _combine(cls):
    def fn(X, y, seed=42):
        k = max(1, min(5, _kmin(y) - 1))
        Xr, yr = cls(random_state=seed,
                     smote=SMOTE(k_neighbors=k, random_state=seed)
                     ).fit_resample(X, y)
        return Xr, yr
    return fn


METHODS = {
    "Base":         no_augmentation,
    "SMOTE":        _smote(SMOTE),
    "Borderline":   _smote(BorderlineSMOTE),
    "SMOTE-ENN":    _combine(SMOTEENN),
    "SMOTE-Tomek":  _combine(SMOTETomek),
    "SMOTE-IPF":    smote_ipf,
    "MWMOTE":       mwmote,
    "VAE":          vae_oversample,
    "CalKDE":       calkde,
    "GeoB":         geob,
    "CC-SG":        lambda X, y, seed=42: CCSG(random_state=seed).fit_resample(X, y),
    "CC-SG-strict": lambda X, y, seed=42: CCSG(strict=True, random_state=seed).fit_resample(X, y),
}


def main():
    datasets = load_primary()
    df = run_full(datasets, METHODS, classifiers=["RF", "XGB", "LR"],
                  n_splits=5, n_repeats=3, seed=42)
    # cost-sensitive / threshold-moving baselines (no augmentation)
    df2 = run_full(datasets, {"Base": no_augmentation},
                   classifiers=["RF-W", "RF-THR"], n_splits=5, n_repeats=3,
                   seed=42)
    df2["method"] = df2["clf"].map({"RF-W": "Weighted", "RF-THR": "ThreshMove"})
    df2["clf"] = "RF"
    out = pd.concat([df, df2[df2.method.notna()]], ignore_index=True)
    path = os.path.join(OUT, "primary.csv")
    out.to_csv(path, index=False)
    print("saved", path)


if __name__ == "__main__":
    main()
