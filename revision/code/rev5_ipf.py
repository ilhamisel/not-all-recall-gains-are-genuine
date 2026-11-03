#!/usr/bin/env python3
"""Rev.5 — corrected SMOTE-IPF (Saez, Luengo, Stefanowski & Herrera 2015,
Information Sciences 291:184-203).

Why a new implementation: augment_rev3.smote_ipf trained an UNPRUNED 25-tree
RF on 5/6 of the data per partition and let it vote on its own training rows,
so (memorised) points were never majority-misclassified and the filter removed
~nothing (SMOTE-IPF == SMOTE to 1e-3).  augment_rev3.py is left untouched.

Algorithm (published defaults, Saez et al. 2015, Sect. 3 / Table 3):
  1. SMOTE to full balance; inner SMOTE identical to all other SMOTE rivals in
     this study (augment_rev3._safe_smote: k = max(1, min(5, n_rarest - 1))).
  2. Iterative-Partitioning Filter (Khoshgoftaar & Rebours 2007) on the
     augmented set E (original + synthetic; IPF filters both):
       a. split current E into n = 9 disjoint stratified partitions;
       b. build one C4.5-like tree on EACH PARTITION (1/9 of E, not 8/9);
       c. every tree predicts ALL samples of E;
       d. majority scheme: remove a sample if more than half of the n trees
          misclassify it (#correct votes < n/2);
       e. stop when, for k = 3 consecutive iterations, the number removed is
          < p = 1 % of the ORIGINAL training-set size (|X| before SMOTE).
  C4.5 surrogate: sklearn DecisionTreeClassifier(criterion="entropy",
  min_samples_leaf=2) = C4.5's information-gain splits with Weka's -M 2
  minimum leaf size; C4.5's pessimistic error pruning (CF = 0.25) has no
  sklearn equivalent, so we add light cost-complexity pruning ccp_alpha=0.005
  (documented, fixed a priori, not tuned).
  Safety guards (not in the paper, needed for tiny classes): stop if any class
  in E has fewer than n samples (stratified partitioning impossible) or if a
  removal would delete an entire class (that removal is skipped); hard cap of
  50 iterations.
Signature (X, y, seed) like every other augmenter.  LOG records per call:
n_orig, n_after_smote, n_removed_total, n_removed_original, n_removed_synth,
iterations.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from augment_rev3 import _safe_smote

N_PART, K_ITER, P_FRAC, CCP, MSL, MAX_IT = 9, 3, 0.01, 0.005, 2, 50
LOG = []
CTX = {}
CONSENSUS = [False]   # sensitivity switch (published default: majority)


def _c45(seed):
    return DecisionTreeClassifier(criterion="entropy", min_samples_leaf=MSL,
                                  ccp_alpha=CCP, random_state=seed)


def ipf_filter(X, y, n_orig, seed=42, n_part=N_PART, k=K_ITER, p=P_FRAC,
               consensus=False):
    """Returns boolean keep-mask over (X, y) and number of iterations."""
    rng = np.random.RandomState(seed)
    keep = np.ones(len(X), bool)
    calm, it = 0, 0
    while it < MAX_IT:
        idx = np.where(keep)[0]
        Xk, yk = X[idx], y[idx]
        if pd.Series(yk).value_counts().min() < n_part:
            break
        skf = StratifiedKFold(n_splits=n_part, shuffle=True,
                              random_state=rng.randint(1 << 30))
        correct = np.zeros(len(idx))
        for _, part in skf.split(Xk, yk):          # train on ONE partition
            t = _c45(seed).fit(Xk[part], yk[part])
            correct += (t.predict(Xk) == yk)
        thr = n_part if consensus else n_part / 2.0
        bad = (correct == 0) if consensus else (correct < thr)
        # never delete a whole class
        for c in np.unique(yk):
            if bad[yk == c].all():
                bad[yk == c] = False
        n_rm = int(bad.sum())
        keep[idx[bad]] = False
        it += 1
        calm = calm + 1 if n_rm < p * n_orig else 0
        if calm >= k:
            break
    return keep, it


def smote_ipf(X, y, seed=42):
    Xr, yr = _safe_smote(y, seed).fit_resample(X, y)
    Xr, yr = np.asarray(Xr), np.asarray(yr)
    n0 = len(X)
    keep, it = ipf_filter(Xr, yr, n0, seed=seed, consensus=CONSENSUS[0])
    syn = np.arange(len(Xr)) >= n0          # imblearn appends synthetic rows
    rec = dict(n_orig=n0, n_after_smote=len(Xr),
               n_removed_total=int((~keep).sum()),
               n_removed_original=int((~keep & ~syn).sum()),
               n_removed_synth=int((~keep & syn).sum()), iterations=it)
    rec.update(CTX); LOG.append(rec)
    return Xr[keep], yr[keep]


# ── unit check on a toy dataset ────────────────────────────────────────────
def _toy_check():
    from sklearn.datasets import make_classification
    import augment_rev3 as A3
    from run_literature import wrap
    from imblearn.over_sampling import SMOTE
    out = []
    for sep, flip in [(s_, f_) for s_ in (2.0, 1.2) for f_ in (0.0, 0.05, 0.15)]:
        X, y = make_classification(n_samples=600, n_features=6, n_informative=4,
                                   n_redundant=0, n_classes=3,
                                   weights=[0.80, 0.15, 0.05], flip_y=0,
                                   class_sep=sep, random_state=1)
        rng = np.random.RandomState(0)
        noisy = rng.rand(len(y)) < flip                 # inject label noise
        y = y.copy(); y[noisy] = (y[noisy] + rng.randint(1, 3, noisy.sum())) % 3
        Xs, ys = wrap(SMOTE)(X, y, seed=7)
        Xo, yo = A3.smote_ipf(X, y, seed=7)
        LOG.clear()
        Xn, yn = smote_ipf(X, y, seed=7)
        r = LOG[-1]
        # how many of the injected noisy ORIGINAL rows were removed
        kept_orig = {tuple(np.round(v, 10)) for v in Xn}
        rm_noisy = sum(tuple(np.round(X[i], 10)) not in kept_orig
                       for i in np.where(noisy)[0])
        rm_clean = sum(tuple(np.round(X[i], 10)) not in kept_orig
                       for i in np.where(~noisy)[0])
        out.append(dict(class_sep=sep, label_noise=flip, n_orig=len(X), n_smote=len(Xs),
                        n_old_ipf=len(Xo), n_new_ipf=len(Xn),
                        removed_total=r["n_removed_total"],
                        removed_synth=r["n_removed_synth"],
                        removed_orig=r["n_removed_original"],
                        injected_noisy=int(noisy.sum()), noisy_removed=rm_noisy,
                        frac_noisy_removed=rm_noisy / max(noisy.sum(), 1),
                        frac_clean_orig_removed=rm_clean / max((~noisy).sum(), 1),
                        iterations=r["iterations"],
                        same_as_smote=bool(len(Xn) == len(Xs) and np.allclose(Xn, Xs))))
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    df = _toy_check()
    print(df.to_string(index=False))
    import os
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "rev5_ipf_toycheck.csv"), index=False)
