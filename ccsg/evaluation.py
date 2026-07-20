"""Evaluation framework: metrics, classifiers, and repeated cross-validation.

Augmentation is applied to the training fold only; standardization is fitted on
the training fold and applied to the test fold, so there is no leakage.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _per_class_recall(y_true, y_pred):
    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return labels, np.diag(cm) / np.clip(cm.sum(axis=1), 1, None)


def gmean_macro(y_true, y_pred, eps=1e-6):
    """Macro geometric mean of per-class recalls (recalls floored at ``eps``)."""
    _, r = _per_class_recall(y_true, y_pred)
    return float(np.exp(np.mean(np.log(np.clip(r, eps, None)))))


def minority_recall(y_true, y_pred, frac=0.25):
    """Mean recall over the smallest ``frac`` fraction of classes."""
    vc = pd.Series(y_true).value_counts()
    small = list(vc.tail(max(1, int(round(len(vc) * frac)))).index)
    labels, rec = _per_class_recall(y_true, y_pred)
    rmap = dict(zip(labels, rec))
    return float(np.mean([rmap[s] for s in small]))


def rare_recall(y_true, y_pred):
    """Recall of the single rarest class."""
    vc = pd.Series(y_true).value_counts()
    labels, rec = _per_class_recall(y_true, y_pred)
    return float(dict(zip(labels, rec))[vc.index[-1]])


METRICS = {
    "min_recall": minority_recall,
    "rare_recall": rare_recall,
    "gmean": gmean_macro,
    "f1_macro": lambda yt, yp: f1_score(yt, yp, average="macro"),
    "mcc": matthews_corrcoef,
}


# --------------------------------------------------------------------------- #
# Classifiers (including the cost-sensitive and threshold-moving baselines)
# --------------------------------------------------------------------------- #
class _WeightedXGB(XGBClassifier):
    """XGBoost with class-balanced sample weights."""
    def fit(self, X, y, **kw):
        from sklearn.utils.class_weight import compute_sample_weight
        return super().fit(X, y, sample_weight=compute_sample_weight(
            "balanced", y), **kw)


class ThresholdMoveRF:
    """Threshold-moving baseline: an unaugmented random forest whose per-class
    one-vs-rest thresholds are tuned on an inner split to maximise F1."""
    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.clf = RandomForestClassifier(n_estimators=100,
                                          random_state=self.seed, n_jobs=-1)
        oof = np.zeros((len(X), len(self.classes_)))
        try:
            skf = StratifiedKFold(n_splits=3, shuffle=True,
                                  random_state=self.seed)
            for tr, te in skf.split(X, y):
                c = RandomForestClassifier(n_estimators=100,
                                           random_state=self.seed, n_jobs=-1)
                c.fit(X[tr], y[tr])
                cols = np.searchsorted(self.classes_, c.classes_)
                oof[te[:, None], cols] = c.predict_proba(X[te])
            thr = np.full(len(self.classes_), 0.5)
            for j, cl in enumerate(self.classes_):
                yb = (y == cl).astype(int)
                best_f1, best_t = -1, 0.5
                for t in np.linspace(0.05, 0.9, 18):
                    f1 = f1_score(yb, (oof[:, j] >= t).astype(int),
                                  zero_division=0)
                    if f1 > best_f1:
                        best_f1, best_t = f1, t
                thr[j] = max(best_t, 1e-3)
            self.thr_ = thr
        except Exception:
            self.thr_ = np.full(len(self.classes_), 0.5)
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        cols = np.searchsorted(self.classes_, self.clf.classes_)
        Pf = np.zeros((len(X), len(self.classes_)))
        Pf[:, cols] = self.clf.predict_proba(X)
        return self.classes_[np.argmax(Pf / self.thr_[None, :], axis=1)]


def make_classifier(name, seed=42):
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=seed,
                                      n_jobs=-1)
    if name == "XGB":
        return XGBClassifier(n_estimators=100, verbosity=0, random_state=seed,
                             n_jobs=-1)
    if name == "LR":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000,
                                                random_state=seed))
    if name == "MLP":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=300,
                             random_state=seed)
    if name == "RF-W":
        return RandomForestClassifier(n_estimators=100, random_state=seed,
                                      n_jobs=-1, class_weight="balanced")
    if name == "LR-W":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000,
                                                random_state=seed,
                                                class_weight="balanced"))
    if name == "XGB-W":
        return _WeightedXGB(n_estimators=100, verbosity=0, random_state=seed,
                            n_jobs=-1)
    if name == "RF-THR":
        return ThresholdMoveRF(seed)
    if name == "BRF":
        from imblearn.ensemble import BalancedRandomForestClassifier
        return BalancedRandomForestClassifier(n_estimators=100,
                                              random_state=seed, n_jobs=-1)
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# Repeated stratified cross-validation
# --------------------------------------------------------------------------- #
def run_full(datasets, methods, classifiers=("RF",), n_splits=5, n_repeats=3,
             seed=42, scale=True, verbose=True):
    """Evaluate ``methods`` (name -> ``fn(X, y, seed=...)``) on ``datasets``
    (name -> ``(X, y)``) with the given classifiers.

    Returns a long-format DataFrame with columns
    ``dataset, method, clf, metric, mean, std``.
    """
    rows = []
    for ds_name, (X, y) in datasets.items():
        X = np.asarray(X, float)
        y = np.asarray(y)
        if verbose:
            vc = pd.Series(y).value_counts()
            print(f"[{ds_name}] n={len(X)} d={X.shape[1]} "
                  f"K={len(vc)} IR={vc.max() / vc.min():.1f}", flush=True)
        scores = {}
        for rep in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=seed + rep)
            for fold, (tr, te) in enumerate(skf.split(X, y)):
                Xtr0, ytr0, Xte, yte = X[tr], y[tr], X[te], y[te]
                if scale:
                    sc = StandardScaler().fit(Xtr0)
                    Xtr0, Xte = sc.transform(Xtr0), sc.transform(Xte)
                for mname, fn in methods.items():
                    try:
                        Xtr, ytr = fn(Xtr0.copy(), ytr0.copy(),
                                      seed=seed + rep * 10 + fold)
                    except Exception:
                        Xtr, ytr = Xtr0, ytr0
                    for clf_name in classifiers:
                        clf = make_classifier(clf_name, seed)
                        clf.fit(Xtr, ytr)
                        yp = clf.predict(Xte)
                        for m, mfn in METRICS.items():
                            scores.setdefault((mname, clf_name, m),
                                              []).append(mfn(yte, yp))
        for clf_name in classifiers:
            for mname in methods:
                for m in METRICS:
                    s = np.array(scores[(mname, clf_name, m)])
                    rows.append(dict(dataset=ds_name, method=mname,
                                     clf=clf_name, metric=m,
                                     mean=float(s.mean()), std=float(s.std())))
    return pd.DataFrame(rows)


def no_augmentation(X, y, seed=42):
    """Identity 'method' for the unaugmented baseline."""
    return X, y
