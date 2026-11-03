#!/usr/bin/env python3
"""Rev.4 (Reviewer 1) — does CC-strict / CC-SG transfer to medical images?

Pipeline per CV fold (5x3 repeated stratified CV, seed 42+rep, as in paper):
  frozen ResNet-50 embedding (2048-d, precomputed, label-free)
  -> per-sample L2 normalisation (label-free)
  -> PCA(d) fit on the TRAINING fold only
  -> StandardScaler fit on the TRAINING fold only (paper's fold-internal scaling)
  -> oversampling in this embedding space (training fold only)
  -> classifier (RF primary; LR = linear probe)
Image-native reference: PixAug = embeddings of pixel-augmented views of the
training-fold minority images (modality-aware flips + affine + intensity),
projected with the same fold PCA/scaler, used to balance each class.
Weighted = no augmentation, class_weight="balanced" (RF-W / LR-W).
Frozen paper configuration for every generator (no tuning on images).

Output: results/folds_<tag>.csv (long format, one row per fold x metric),
        results/fbstats_<tag>.csv (CC-strict certification bookkeeping),
        results/failures_<tag>.csv
Usage: python run_img.py <tag> <pca_dim> <task> [<task> ...]
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # SG-KDE/
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, ADASYN,
                                    RandomOverSampler)

import benchmark_math_models as B
import eval_framework as E
import augment_xai_v4 as X4
import augment_rev as AR
from run_importance_v3b import GS
from run_literature import wrap, _adasyn
from run_rev_baselines import smote_enn
from tasks_img import DATA

CCSG = lambda X, y, seed=42: X4.augment_xsemi(
    X, y, mode="cfgate", frac=0.5, depth_lo=0.25, depth_hi=0.75,
    p_thr=0.5, seed=seed)
GEOB = lambda X, y, seed=42: X4.augment_xsemi(
    X, y, mode="geo", frac=0.5, depth_lo=0.25, depth_hi=0.75, seed=seed)
STRICT = lambda X, y, seed=42: AR.augment_ccsg_strict(X, y, strict=True,
                                                      seed=seed)
GEN = {
    "Base":       B.augment_base,
    "ROS":        wrap(RandomOverSampler, needs_k=False),
    "SMOTE":      wrap(SMOTE),
    "Borderline": wrap(BorderlineSMOTE),
    "ADASYN":     _adasyn,
    "SMOTE-ENN":  smote_enn,
    "CalKDE":     GS(0.85),
    "GeoB":       GEOB,
    "CC-SG":      CCSG,
    "CC-strict":  STRICT,
}
CLFS = ["RF", "LR"]


def pixaug(Ztr, ytr, Zaug, yaug, seed):
    """Balance each non-majority class with embedded pixel-augmented views of
    its own training-fold images (without replacement while the pool lasts)."""
    rng = np.random.RandomState(seed)
    cnt = pd.Series(ytr).value_counts(); tgt = cnt.max()
    Xa, ya = [Ztr], [ytr]
    for c, nc in cnt.items():
        diff = int(tgt - nc)
        if diff <= 0:
            continue
        pool = np.where(yaug == c)[0]
        if len(pool) == 0:
            continue
        pick = rng.choice(pool, diff, replace=diff > len(pool))
        Xa.append(Zaug[pick]); ya.append(np.full(diff, c))
    return np.vstack(Xa), np.concatenate(ya)


def run_task(name, pca_dim, tag, n_splits=5, n_repeats=3, seed=42):
    z = np.load(os.path.join(DATA, f"emb_{name}.npz"))
    Eall = normalize(z["E"]); y = z["y"]
    Eaug = normalize(z["E_aug"]); asrc = z["aug_src"]
    aloc = asrc                      # task-local position of each view's source image
    rows, fails = [], []
    AR.FB_STATS.clear()
    fbrows = []
    vc = pd.Series(y).value_counts()
    print(f"\n=== {name} n={len(y)} K={len(vc)} IR={vc.max()/vc.min():.1f} "
          f"PCA={pca_dim}", flush=True)
    t0 = time.time()
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + rep)
        for fold, (tr, te) in enumerate(skf.split(Eall, y)):
            pca = PCA(n_components=pca_dim, random_state=0).fit(Eall[tr])
            sc = StandardScaler().fit(pca.transform(Eall[tr]))
            proj = lambda A: sc.transform(pca.transform(A))
            Ztr, Zte, ytr, yte = proj(Eall[tr]), proj(Eall[te]), y[tr], y[te]
            trset = np.zeros(len(y), bool); trset[tr] = True
            am = trset[aloc]
            Zaug, yaug = proj(Eaug[am]), y[aloc[am]]
            fseed = seed + rep * 10 + fold
            runs = {}
            for m, fn in GEN.items():
                try:
                    nb = len(AR.FB_STATS)
                    runs[m] = fn(Ztr.copy(), ytr.copy(), seed=fseed)
                    if m == "CC-strict":
                        for r in AR.FB_STATS[nb:]:
                            fbrows.append(dict(task=name, rep=rep, fold=fold,
                                               **r))
                except Exception as e:
                    runs[m] = (Ztr, ytr)
                    fails.append(dict(task=name, method=m, rep=rep, fold=fold,
                                      exc=type(e).__name__, msg=str(e)[:200]))
            runs["PixAug"] = pixaug(Ztr, ytr, Zaug, yaug, fseed)
            jobs = [(m, c, c) for m in runs for c in CLFS]
            jobs += [("Weighted", "RF", "RF-W"), ("Weighted", "LR", "LR-W")]
            for m, clab, cname in jobs:
                Xf, yf = runs["Base" if m == "Weighted" else m]
                clf = E.make_classifier(cname, seed)
                clf.fit(Xf, yf)
                yp = clf.predict(Zte)
                for met, mfn in E.METRICS.items():
                    rows.append(dict(task=name, method=m, clf=clab, rep=rep,
                                     fold=fold, metric=met,
                                     value=float(mfn(yte, yp)),
                                     n_train=len(yf)))
        print(f"  rep {rep} done  {(time.time()-t0)/60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    g = df[df.metric.isin(["f1_macro", "mcc", "min_recall"])].groupby(
        ["clf", "method", "metric"]).value.mean().unstack()
    print(g.round(4).to_string(), flush=True)
    return df, pd.DataFrame(fbrows), pd.DataFrame(fails)


def main():
    tag, pca_dim, names = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    out = os.path.join(HERE, "results")
    for name in names:
        df, fb, fl = run_task(name, pca_dim, tag)
        df.to_csv(os.path.join(out, f"folds_{tag}_{name}.csv"), index=False)
        fb.to_csv(os.path.join(out, f"fbstats_{tag}_{name}.csv"), index=False)
        fl.to_csv(os.path.join(out, f"failures_{tag}_{name}.csv"),
                  index=False)


if __name__ == "__main__":
    main()
