"""Dataset loaders for the three evaluation suites used in the paper.

* :func:`load_primary`           -- the 10 primary datasets (imbalance ratio
  9-440): 8 binary KEEL/UCI benchmarks fetched by ``imbalanced-learn`` plus the
  two multi-class wine-quality datasets (run ``python data/download.py`` first).
* :func:`load_external_moderate` -- 10 unseen moderate-imbalance datasets
  (IR 9-20) for the first frozen-configuration external validation.
* :func:`load_external_extreme`  -- 9 unseen real datasets sub-sampled to
  IR 111-400 for the extreme-regime external validation.
* :func:`make_synthetic_extreme` -- 9 controlled synthetic datasets (IR 100-444,
  noise-free labels) for the mechanism-level external validation.

All real data is fetched from its original public source; nothing is
redistributed. See ``data/README.md`` for source links.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.datasets import fetch_datasets

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(_HERE, "..", "data", "raw"))

# ---- suite definitions ---------------------------------------------------- #
PRIMARY_KEEL = ["ecoli", "us_crime", "oil", "car_eval_4", "yeast_me2",
                "ozone_level", "mammography", "abalone_19"]
PRIMARY_WINE = ["Wine_Red", "Wine_White"]        # multi-class, from UCI

EXTERNAL_MODERATE = ["satimage", "sick_euthyroid", "spectrometer",
                     "car_eval_34", "libras_move", "thyroid_sick",
                     "solar_flare_m0", "yeast_ml8", "coil_2000", "arrhythmia"]

EXTREME_REAL_SRC = ["webpage", "letter_img", "pen_digits", "optical_digits",
                    "scene", "isolet", "wine_quality", "abalone"]
EXTREME_TARGET_IR = {"webpage": 200, "letter_img": 300, "pen_digits": 400,
                     "optical_digits": 300, "scene": 150, "isolet": 200,
                     "wine_quality": 150, "abalone": 200}
_MAJ_CAP = 8000


def _binize(y):
    y = np.asarray(y)
    return (y == 1).astype(int) if set(np.unique(y)) <= {-1, 1} else \
        LabelEncoder().fit_transform(y)


def _load_wine(name):
    path = os.path.join(DATA_DIR, name + ".csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python data/download.py` first to fetch "
            "the UCI wine-quality datasets.")
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values.astype(float)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    if np.isnan(X).any():
        X = SimpleImputer(strategy="median").fit_transform(X)
    return X, y


def load_primary():
    """Return ``{name: (X, y)}`` for the 10 primary datasets."""
    out = {}
    raw = fetch_datasets(filter_data=tuple(PRIMARY_KEEL))
    for n in PRIMARY_KEEL:
        out[n] = (np.asarray(raw[n].data, float),
                  LabelEncoder().fit_transform(raw[n].target))
    for n in PRIMARY_WINE:
        out[n] = _load_wine(n)
    return out


def load_external_moderate():
    """Return ``{name: (X, y)}`` for the 10 moderate-imbalance external sets."""
    raw = fetch_datasets(filter_data=tuple(EXTERNAL_MODERATE))
    return {n: (np.asarray(raw[n].data, float),
                LabelEncoder().fit_transform(raw[n].target))
            for n in EXTERNAL_MODERATE}


def load_external_extreme(seed=42):
    """Return ``{name: (X, y)}`` for the 9 real extreme-IR external sets.

    Built by sub-sampling the minority class of unused public benchmarks to a
    target imbalance ratio, plus the natively extreme ``protein_homo``.
    """
    rng = np.random.RandomState(seed)
    raw = fetch_datasets(filter_data=tuple(EXTREME_REAL_SRC))
    out = {}
    for n in EXTREME_REAL_SRC:
        X = np.asarray(raw[n].data, float)
        y = _binize(raw[n].target)
        maj_i = np.where(y == 0)[0]
        min_i = np.where(y == 1)[0]
        ir = EXTREME_TARGET_IR[n]
        if len(maj_i) > _MAJ_CAP:
            maj_i = rng.choice(maj_i, _MAJ_CAP, replace=False)
        n_min = max(5, len(maj_i) // ir)
        if n_min < len(min_i):
            min_i = rng.choice(min_i, n_min, replace=False)
        idx = np.concatenate([maj_i, min_i])
        rng.shuffle(idx)
        out[f"{n}_x"] = (X[idx], y[idx])
    p = fetch_datasets(filter_data=("protein_homo",))["protein_homo"]
    Xp, yp = np.asarray(p.data, float), _binize(p.target)
    mi = rng.choice(np.where(yp == 1)[0], 150, replace=False)
    ma = rng.choice(np.where(yp == 0)[0], 150 * 120, replace=False)
    idx = np.concatenate([ma, mi]); rng.shuffle(idx)
    out["protein_homo_x"] = (Xp[idx], yp[idx])
    return out


def make_synthetic_extreme(seed=42):
    """Return ``{name: (X, y)}`` for the 9 controlled synthetic datasets
    (IR 100/200/400 x three separability levels, noise-free labels)."""
    out = {}
    for ir in (100, 200, 400):
        for name, sep, ncl in [("easy", 2.0, 1), ("med", 1.0, 2),
                               ("hard", 0.6, 3)]:
            n_maj = 4000
            n_min = max(10, n_maj // ir)
            X, y = make_classification(
                n_samples=n_maj + n_min, n_features=20, n_informative=8,
                n_redundant=4, n_clusters_per_class=ncl, class_sep=sep,
                weights=[n_maj / (n_maj + n_min)], flip_y=0.0,
                random_state=seed + ir + ncl)
            out[f"syn_ir{ir}_{name}"] = (X.astype(float), y.astype(int))
    return out
