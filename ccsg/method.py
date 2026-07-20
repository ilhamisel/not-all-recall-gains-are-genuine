"""CC-SG: consistency-controlled, surrogate-gated Gaussian oversampling.

CC-SG balances each training fold class by class. Half of the synthetic
budget is drawn from a calibrated component-local Gaussian KDE core; the other
half is generated as boundary samples that must pass an *in-loop admission
test* -- a one-vs-rest surrogate must still assign each candidate to the
minority class, otherwise the candidate is rejected and resampled rather than
deleted. Any budget left unfilled after a bounded number of rounds is
completed by counterfactual truncation.

The public entry point is :class:`CCSG`, which follows the
``imbalanced-learn`` convention (``fit_resample``). A functional form,
:func:`oversample`, is also provided.
"""
from __future__ import annotations

import numpy as np

from .core import (SHRINK, core_samples, fit_gmm, flip_points, kde_components,
                   majority_neighbors, minority_iter, ovr_surrogate)


def _oversample_class(rng, X, y, c, diff, *, beta, tau, depth_lo, depth_hi,
                      shrink, rounds, strict, seed):
    """Generate ``diff`` synthetic samples for a single minority class."""
    d = X.shape[1]
    X_c, X_maj = X[y == c], X[y != c]
    n_boundary = int(diff * beta) if len(X_maj) else 0
    n_core = diff - n_boundary

    gmm = fit_gmm(X_c, seed=seed)
    comps = kde_components(X_c, gmm.predict(X_c), gmm.n_components, d, shrink)
    syn = core_samples(rng, gmm, comps, n_core)

    if n_boundary == 0:
        return np.asarray(syn)

    kk, nbr = majority_neighbors(X_c, X_maj)
    clf = ovr_surrogate(X, y, c, seed)

    # Phases 2-3: propose toward a majority neighbour, admit if the surrogate
    # still calls it minority, otherwise reject and resample.
    admitted, need = [], n_boundary
    for _ in range(rounds):
        if need <= 0:
            break
        si = rng.randint(len(X_c), size=need)
        mi = nbr[si, rng.randint(kk, size=need)]
        P, Q = X_c[si], X_maj[mi]
        t = rng.uniform(depth_lo, depth_hi, size=need)
        cand = P + t[:, None] * (Q - P)
        ok = clf.predict_proba(cand)[:, 1] >= tau
        admitted.extend(cand[ok])
        need = n_boundary - len(admitted)

    # Phase 4: counterfactual-truncation fallback for any unfilled quota.
    if need > 0:
        si = rng.randint(len(X_c), size=need)
        mi = nbr[si, rng.randint(kk, size=need)]
        P, Q = X_c[si], X_maj[mi]
        t = rng.uniform(depth_lo, depth_hi, size=need) * flip_points(clf, P, Q,
                                                                     d, tau)
        fb = P + t[:, None] * (Q - P)
        if strict:
            # Re-check the fallback candidates; replace violators with core
            # samples so that every admitted sample passes the surrogate.
            keep = fb[clf.predict_proba(fb)[:, 1] >= tau]
            admitted.extend(keep)
            admitted.extend(core_samples(rng, gmm, comps,
                                         n_boundary - len(admitted)))
        else:
            admitted.extend(fb)

    syn.extend(admitted)
    return np.asarray(syn)


def oversample(X, y, *, beta=0.5, tau=0.5, depth=(0.25, 0.75), shrink=SHRINK,
               rounds=6, strict=False, seed=42):
    """Balance ``(X, y)`` with CC-SG and return the augmented ``(X, y)``.

    Parameters
    ----------
    beta : float
        Fraction of each class's synthetic budget assigned to the certified
        boundary sampler (the rest goes to the KDE core). Default ``0.5``.
    tau : float
        Admission threshold; a candidate is admitted iff the surrogate scores
        it ``>= tau``. Default ``0.5``.
    depth : tuple(float, float)
        Uniform interval for the boundary depth ``t``. Default ``(0.25, 0.75)``.
    shrink : float
        Uniform bandwidth shrinkage factor ``s`` for the KDE core.
    rounds : int
        Maximum reject-and-resample rounds before the fallback. Default ``6``.
    strict : bool
        If ``True``, re-check fallback candidates so that *every* admitted
        sample passes the surrogate (the fully certified variant).
    seed : int
        Random seed.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    Xa, ya = [X], [y]
    lo, hi = depth
    for c, diff in minority_iter(X, y):
        syn = _oversample_class(rng, X, y, c, diff, beta=beta, tau=tau,
                                depth_lo=lo, depth_hi=hi, shrink=shrink,
                                rounds=rounds, strict=strict, seed=seed)
        Xa.append(syn)
        ya.append(np.full(len(syn), c))
    return np.vstack(Xa), np.concatenate(ya)


class CCSG:
    """CC-SG oversampler with an ``imbalanced-learn``-style API.

    Example
    -------
    >>> from ccsg import CCSG
    >>> X_res, y_res = CCSG(random_state=42).fit_resample(X, y)
    """

    def __init__(self, beta=0.5, tau=0.5, depth=(0.25, 0.75), shrink=SHRINK,
                 rounds=6, strict=False, random_state=42):
        self.beta = beta
        self.tau = tau
        self.depth = depth
        self.shrink = shrink
        self.rounds = rounds
        self.strict = strict
        self.random_state = random_state

    def fit_resample(self, X, y):
        return oversample(X, y, beta=self.beta, tau=self.tau, depth=self.depth,
                          shrink=self.shrink, rounds=self.rounds,
                          strict=self.strict, seed=self.random_state)
