"""Baselines and matched controls used in the paper.

Included here:

* :func:`calkde`  -- the calibrated component-local KDE core alone (no
  boundary sampler); the "CalKDE" internal reference.
* :func:`geob`    -- the identical boundary sampler with the admission test
  removed (purely geometric boundary samples); the "GeoB" matched control that
  isolates the effect of in-loop screening.
* :func:`smote_ipf`, :func:`mwmote`, :func:`vae_oversample` -- faithful
  implementations of three modern rivals (a generate-then-clean filter, a
  selective oversampler, and a deep generative baseline).

The classical SMOTE family (SMOTE, Borderline-SMOTE, ADASYN, SMOTE-ENN,
SMOTE-Tomek) is used directly from ``imbalanced-learn`` in the experiment
scripts.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE

from .core import (core_samples, fit_gmm, kde_components, majority_neighbors,
                   minority_iter)
from .method import oversample

warnings.filterwarnings("ignore")


def _kmin(y):
    return int(pd.Series(y).value_counts().min())


def calkde(X, y, seed=42):
    """KDE core only (CalKDE): the Phase-1 calibrated component-local sampler."""
    return oversample(X, y, beta=0.0, seed=seed)


def geob(X, y, seed=42, depth=(0.25, 0.75)):
    """Geometric boundary control (GeoB): boundary samples with *no* admission
    test. Differs from CC-SG in the admission test alone."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    lo, hi = depth
    d = X.shape[1]
    Xa, ya = [X], [y]
    for c, diff in minority_iter(X, y):
        X_c, X_maj = X[y == c], X[y != c]
        n_bnd = int(diff * 0.5) if len(X_maj) else 0
        gmm = fit_gmm(X_c, seed=seed)
        comps = kde_components(X_c, gmm.predict(X_c), gmm.n_components, d)
        syn = core_samples(rng, gmm, comps, diff - n_bnd)
        if n_bnd:
            kk, nbr = majority_neighbors(X_c, X_maj)
            si = rng.randint(len(X_c), size=n_bnd)
            mi = nbr[si, rng.randint(kk, size=n_bnd)]
            P, Q = X_c[si], X_maj[mi]
            t = rng.uniform(lo, hi, size=n_bnd)
            syn.extend(P + t[:, None] * (Q - P))
        Xa.append(np.asarray(syn))
        ya.append(np.full(len(syn), c))
    return np.vstack(Xa), np.concatenate(ya)


# --------------------------------------------------------------------------- #
# SMOTE-IPF (Saez et al., 2015): SMOTE + iterative partitioning filter
# --------------------------------------------------------------------------- #
def smote_ipf(X, y, seed=42, n_part=6, k_iter=3, p=0.01):
    """SMOTE followed by an iterative ensemble noise filter that removes
    synthetic samples misclassified by a majority vote of partition-trained
    forests. Only synthetic samples may be removed."""
    rng = np.random.RandomState(seed)
    sm = SMOTE(k_neighbors=max(1, min(5, _kmin(y) - 1)), random_state=seed)
    Xr, yr = sm.fit_resample(X, y)
    Xr, yr = np.asarray(Xr), np.asarray(yr)
    synth = np.arange(len(Xr)) >= len(X)
    keep = np.ones(len(Xr), bool)
    for _ in range(k_iter):
        idx = np.where(keep)[0]
        if len(idx) < n_part * 2:
            break
        votes = np.zeros(len(Xr))
        counts = np.zeros(len(Xr))
        skf = StratifiedKFold(n_splits=n_part, shuffle=True,
                              random_state=rng.randint(1 << 30))
        Xk, yk = Xr[idx], yr[idx]
        try:
            splits = list(skf.split(Xk, yk))
        except ValueError:
            break
        for tr, _te in splits:
            clf = RandomForestClassifier(n_estimators=25, random_state=seed,
                                         n_jobs=-1).fit(Xk[tr], yk[tr])
            votes += (clf.predict(Xr) == yr).astype(float)
            counts += 1.0
        agree = np.divide(votes, np.maximum(counts, 1))
        drop = synth & keep & (agree < 0.5)
        changed = drop.sum() / max(len(Xr), 1)
        keep &= ~drop
        if changed < p:
            break
    return Xr[keep], yr[keep]


# --------------------------------------------------------------------------- #
# MWMOTE-style (Barua et al., 2014): weighted informative-minority synthesis
# --------------------------------------------------------------------------- #
def mwmote(X, y, seed=42, k1=5, k2=3):
    """Select informative (borderline) minority samples, weight them by
    proximity to the majority class, and synthesise along minority neighbours."""
    rng = np.random.RandomState(seed)
    counts = pd.Series(y).value_counts()
    target = counts.max()
    Xa, ya = [X], [y]
    yflat = np.asarray(y)
    for c in counts.index:
        diff = int(target - counts[c])
        if diff <= 0:
            continue
        Xc, Xoth = X[y == c], X[y != c]
        if len(Xc) < 2 or len(Xoth) < 1:
            si = rng.randint(len(Xc), size=diff)
            Xa.append(Xc[si]); ya.append(np.full(diff, c)); continue
        kk = min(k1, len(X) - 1)
        _, nb = NearestNeighbors(n_neighbors=kk + 1).fit(X).kneighbors(Xc)
        border = np.array([np.any(yflat[row[1:]] != c) for row in nb])
        Xinfo = Xc[border] if border.sum() >= 2 else Xc
        dmaj, _ = NearestNeighbors(n_neighbors=1).fit(Xoth).kneighbors(Xinfo)
        w = 1.0 / (dmaj[:, 0] + 1e-6)
        w = w / w.sum()
        kc = min(k2, max(1, len(Xc) - 1))
        _, nbmin = NearestNeighbors(n_neighbors=kc + 1).fit(Xc).kneighbors(Xinfo)
        syn = []
        for _ in range(diff):
            i = rng.choice(len(Xinfo), p=w)
            j = nbmin[i, rng.randint(1, nbmin.shape[1])]
            syn.append(Xinfo[i] + rng.uniform(0, 1) * (Xc[j] - Xinfo[i]))
        Xa.append(np.asarray(syn)); ya.append(np.full(diff, c))
    return np.vstack(Xa), np.concatenate(ya)


# --------------------------------------------------------------------------- #
# Deep generative baseline: a per-class variational autoencoder
# --------------------------------------------------------------------------- #
def vae_oversample(X, y, seed=42, epochs=200, latent=8, hidden=32):
    """Train a small VAE per minority class and sample from it. Under extreme
    scarcity (fewer than 8 samples) it falls back to a KDE-style perturbation.
    Requires PyTorch."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    counts = pd.Series(y).value_counts()
    target = counts.max()
    d = X.shape[1]
    Xa, ya = [X], [y]

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
            self.mu = nn.Linear(hidden, latent)
            self.lv = nn.Linear(hidden, latent)
            self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                     nn.Linear(hidden, d))

        def forward(self, x):
            h = self.enc(x)
            mu, lv = self.mu(h), self.lv(h)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            return self.dec(z), mu, lv

    for c in counts.index:
        diff = int(target - counts[c])
        if diff <= 0:
            continue
        Xc = X[y == c].astype(np.float32)
        if len(Xc) < 8:
            std = Xc.std(0) + 1e-6
            si = rng.randint(len(Xc), size=diff)
            syn = Xc[si] + rng.normal(0, std * 0.5, size=(diff, d))
            Xa.append(syn.astype(float)); ya.append(np.full(diff, c)); continue
        Xt = torch.tensor(Xc)
        vae = VAE()
        opt = torch.optim.Adam(vae.parameters(), lr=1e-2)
        for _ in range(epochs):
            opt.zero_grad()
            xh, mu, lv = vae(Xt)
            rec = ((xh - Xt) ** 2).mean()
            kld = -0.5 * torch.mean(1 + lv - mu ** 2 - torch.exp(lv))
            (rec + 0.01 * kld).backward()
            opt.step()
        with torch.no_grad():
            syn = vae.dec(torch.randn(diff, latent)).numpy()
        Xa.append(syn.astype(float)); ya.append(np.full(diff, c))
    return np.vstack(Xa), np.concatenate(ya)
