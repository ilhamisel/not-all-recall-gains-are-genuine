"""Core building blocks for CC-SG.

This module contains the low-level components shared by the oversampler:
a robust Gaussian-mixture fit for discovering minority sub-concepts, the
calibrated component-local kernel-density core sampler, the one-vs-rest
surrogate used for the in-loop admission test, and small geometric helpers.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Default hyper-parameters (fixed once on the primary suite; see the paper).
SHRINK = 0.85          # uniform bandwidth shrinkage factor s
CF_GRID = 20           # grid resolution for the counterfactual flip point


# --------------------------------------------------------------------------- #
# Minority-class iteration
# --------------------------------------------------------------------------- #
def minority_iter(X, y):
    """Yield ``(class_label, budget)`` for every minority class.

    ``budget`` is the number of synthetic samples needed to bring the class
    up to the size of the majority class.
    """
    counts = pd.Series(y).value_counts()
    target = counts.max()
    for c in counts.index:
        if counts[c] < target:
            yield c, int(target - counts[c])


# --------------------------------------------------------------------------- #
# Gaussian mixture / component-local KDE core
# --------------------------------------------------------------------------- #
def fit_gmm(X_c, max_k=3, seed=42):
    """Fit a robust Gaussian mixture to a single class.

    Uses ``K = min(max_k, n/2)`` components with a scale-aware covariance
    regularization and a ``full -> diag -> spherical`` fallback so the fit
    never fails on highly correlated or scarce classes.
    """
    k = min(max_k, max(1, len(X_c) // 2))
    if len(X_c) <= 2:
        k = 1
    scale = float(np.mean(np.var(X_c, axis=0))) + 1e-9
    reg = max(1e-4 * scale, 1e-6)
    types = (["full", "diag", "spherical"]
             if len(X_c) > X_c.shape[1] + 2 else ["diag", "spherical"])
    for ct in types:
        try:
            return GaussianMixture(n_components=k, covariance_type=ct,
                                   random_state=seed, reg_covar=reg,
                                   max_iter=100).fit(X_c)
        except Exception:
            continue
    return GaussianMixture(n_components=1, covariance_type="spherical",
                           random_state=seed, reg_covar=max(reg, 1.0)).fit(X_c)


def _silverman(Xk, d):
    """Silverman-type reference bandwidth per feature."""
    return (Xk.std(0) + 1e-6) * (len(Xk) ** (-1.0 / (d + 4)))


def kde_components(X_c, resp, K, d, shrink=SHRINK):
    """Map each GMM component to ``(members, shrunk_bandwidth)``.

    Empty components fall back to the whole class. The bandwidth is the
    Silverman rule multiplied by the uniform shrinkage factor ``s``.
    """
    out = {}
    for k in range(K):
        Xk = X_c[resp == k] if (resp == k).sum() else X_c
        out[k] = (Xk, _silverman(Xk, d) * shrink)
    return out


def core_samples(rng, gmm, comps, n):
    """Draw ``n`` samples from the component-local Gaussian KDE core."""
    out = []
    for _ in range(n):
        k = rng.choice(gmm.n_components, p=gmm.weights_)
        Xk, h = comps[k]
        out.append(Xk[rng.randint(len(Xk))] + rng.normal(0, h))
    return out


# --------------------------------------------------------------------------- #
# Surrogate for the in-loop admission test
# --------------------------------------------------------------------------- #
def ovr_surrogate(X, y, c, seed, n_estimators=30):
    """Fit a one-vs-rest gradient-boosted surrogate for class ``c``."""
    clf = XGBClassifier(verbosity=0, random_state=seed,
                        n_estimators=n_estimators)
    clf.fit(X, (y == c).astype(int))
    return clf


def flip_points(clf, P, Q, d, tau=0.5):
    """First decision-flip point ``t_flip`` along each segment ``P -> Q``.

    Evaluated on a ``CF_GRID``-point grid. Returns 1.0 when the surrogate
    never drops below ``tau`` along the segment.
    """
    n = len(P)
    ts = np.linspace(0.05, 1.0, CF_GRID)
    G = (P[:, None, :] + ts[None, :, None] * (Q - P)[:, None, :]).reshape(-1, d)
    pr = clf.predict_proba(G)[:, 1].reshape(n, CF_GRID)
    flip = np.where(pr < tau, 1, 0).argmax(1)
    return np.where((pr >= tau).all(1), 1.0, ts[flip])


def majority_neighbors(X_c, X_maj, k=5):
    """Return a ``(k, neighbour-index)`` table from each minority point to
    its nearest majority neighbours."""
    kk = min(k, len(X_maj))
    nn = NearestNeighbors(n_neighbors=kk).fit(X_maj)
    _, nbr = nn.kneighbors(X_c)
    return kk, nbr
