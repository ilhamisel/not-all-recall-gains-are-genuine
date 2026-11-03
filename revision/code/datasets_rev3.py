#!/usr/bin/env python3
"""Rev.3 aşırı-IR (IR>=100) GÖRÜLMEMİŞ veri suite'leri (dis hakem #4).
(A) real_extreme: hicbir yerde kullanilmamis imblearn gercek setlerinin
    azinligini alt-orneklemeyle IR 150-400'e cekerek olusturulmus gercek veri
    + native aşırı-IR protein_homo (alt-olcekli).
(B) synth_extreme: bilinen üretici yapıya sahip kontrollü sentetik setler
    (make_classification, IR 100/200/400) — teşhis argümanı için de kullanılır.
Hicbir hiperparametre bu setlere gore ayarlanmadi (dondurulmus CC-SG)."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from imblearn.datasets import fetch_datasets

# birincil/ilk-dis suite'lerde KULLANILMAMIS gercek setler
REAL_SRC = ["webpage", "letter_img", "pen_digits", "optical_digits",
            "scene", "isolet", "wine_quality", "abalone"]
TARGET_IR = {"webpage": 200, "letter_img": 300, "pen_digits": 400,
             "optical_digits": 300, "scene": 150, "isolet": 200,
             "wine_quality": 150, "abalone": 200}
MAJ_CAP = 8000                # buyuk setleri sinirla (hiz)


def _binize(y):
    """imblearn hedefleri {-1,1}; azinlik=1."""
    y = np.asarray(y)
    return (y == 1).astype(int) if set(np.unique(y)) <= {-1, 1} else \
        LabelEncoder().fit_transform(y)


def load_real_extreme(seed=42):
    rng = np.random.RandomState(seed)
    raw = fetch_datasets(filter_data=tuple(REAL_SRC))
    out = {}
    for n in REAL_SRC:
        X = np.asarray(raw[n].data, float)
        y = _binize(raw[n].target)
        maj_i = np.where(y == 0)[0]
        min_i = np.where(y == 1)[0]
        ir = TARGET_IR[n]
        # cogunlugu sinirla, sonra IR'yi tutturacak azinligi sec
        if len(maj_i) > MAJ_CAP:
            maj_i = rng.choice(maj_i, MAJ_CAP, replace=False)
        n_min = max(5, len(maj_i) // ir)
        if n_min < len(min_i):
            min_i = rng.choice(min_i, n_min, replace=False)
        idx = np.concatenate([maj_i, min_i])
        rng.shuffle(idx)
        out[f"{n}_x"] = (X[idx], y[idx])
    # native aşırı-IR: protein_homo (alt-olcekli, IR ~120 korunur)
    p = fetch_datasets(filter_data=("protein_homo",))["protein_homo"]
    Xp = np.asarray(p.data, float); yp = _binize(p.target)
    mi = np.where(yp == 1)[0]; ma = np.where(yp == 0)[0]
    mi = rng.choice(mi, 150, replace=False)
    ma = rng.choice(ma, 150 * 120, replace=False)
    idx = np.concatenate([ma, mi]); rng.shuffle(idx)
    out["protein_homo_x"] = (Xp[idx], yp[idx])
    return out


def make_synth_extreme(seed=42):
    """9 kontrollü set: IR ∈ {100,200,400} × yapı ∈ {kolay,orta,zor}.
    Zorluk class_sep + n_clusters_per_class ile ayarlanir; Bayes sınırı
    üretici tarafından bilinir (teşhis için ideal)."""
    out = {}
    cfgs = [("easy", 2.0, 1), ("med", 1.0, 2), ("hard", 0.6, 3)]
    for ir in (100, 200, 400):
        for name, sep, ncl in cfgs:
            n_maj = 4000
            n_min = max(10, n_maj // ir)
            X, y = make_classification(
                n_samples=n_maj + n_min, n_features=20, n_informative=8,
                n_redundant=4, n_clusters_per_class=ncl, class_sep=sep,
                weights=[n_maj / (n_maj + n_min)], flip_y=0.0,
                random_state=seed + ir + ncl)
            out[f"syn_ir{ir}_{name}"] = (X.astype(float), y.astype(int))
    return out


if __name__ == "__main__":
    import pandas as pd
    for lbl, fn in [("REAL", load_real_extreme), ("SYNTH", make_synth_extreme)]:
        print(f"\n=== {lbl} ===")
        for n, (X, y) in fn().items():
            vc = pd.Series(y).value_counts()
            print(f"  {n:22} n={len(X):6} d={X.shape[1]:3} "
                  f"IR={vc.max()/vc.min():6.1f} nmin={vc.min()}")
