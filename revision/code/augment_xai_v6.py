#!/usr/bin/env python3
"""v6 — CC-SG'yi gecmeye yonelik varyantlar (CC-SG = cfgate tau=0.5 sabit ref).
  Sfront  : sertifikali cepheye hedefleme — t = U(0.75,0.95) × t_flip
  CC-ens  : 3-vekil medyan-p sertifikasi
  CC-adapt: sinif-bazli sinir payi = f(kabul orani)
  CC-core : cekirdek KDE orneklerine de sertifika
Hepsi ayni cekirdegi (GS-0.85 GMM-KDE) ve ayni sinir gramerini paylasir."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import benchmark_math_models as B
from augment_xai_v4 import (_ovr_model, _minority_iter, _kde_components,
                            _flip_points, CF_GRID)

P_THR = 0.5


def _boundary_setup(X, y, c, X_c, X_maj):
    kk = min(5, len(X_maj))
    nn = NearestNeighbors(n_neighbors=kk).fit(X_maj)
    _, nbr = nn.kneighbors(X_c)
    return kk, nbr


def _core_samples(rng, gmm, comps, n):
    out = []
    for _ in range(n):
        k = rng.choice(gmm.n_components, p=gmm.weights_)
        Xk, h = comps[k]
        out.append(Xk[rng.randint(len(Xk))] + rng.normal(0, h))
    return out


# ── Sfront: sertifikali cepheye hedefleme ───────────────────────────────────

def augment_sfront(X, y, lo=0.75, hi=0.95, frac=0.5, seed=42):
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    Xa, ya = [X], [y]
    for c, diff in _minority_iter(X, y):
        X_c, X_maj = X[y == c], X[y != c]
        n_bnd = int(diff * frac); n_core = diff - n_bnd
        gmm = B.fit_gmm(X_c, seed=seed)
        comps = _kde_components(X_c, gmm.predict(X_c), gmm.n_components, d)
        syn = _core_samples(rng, gmm, comps, n_core)
        if n_bnd > 0 and len(X_maj):
            kk, nbr = _boundary_setup(X, y, c, X_c, X_maj)
            clf = _ovr_model(X, y, c, seed)
            si = rng.randint(len(X_c), size=n_bnd)
            mi = nbr[si, rng.randint(kk, size=n_bnd)]
            P, Q = X_c[si], X_maj[mi]
            t_flip = _flip_points(clf, P, Q, d)
            # cephenin hemen ici: ceviri noktasinin bir izgara adimi berisi
            t = rng.uniform(lo, hi, size=n_bnd) * np.maximum(t_flip - 1.0/CF_GRID, 0.05)
            syn.extend(P + t[:, None] * (Q - P))
        Xa.append(np.array(syn)); ya.append(np.full(len(syn), c))
    return np.vstack(Xa), np.concatenate(ya)


# ── CC-ens: 3-vekil medyan-p sertifikasi ────────────────────────────────────

def augment_ccens(X, y, n_sur=3, frac=0.5, lo=0.25, hi=0.75, seed=42):
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    Xa, ya = [X], [y]
    for c, diff in _minority_iter(X, y):
        X_c, X_maj = X[y == c], X[y != c]
        n_bnd = int(diff * frac); n_core = diff - n_bnd
        gmm = B.fit_gmm(X_c, seed=seed)
        comps = _kde_components(X_c, gmm.predict(X_c), gmm.n_components, d)
        syn = _core_samples(rng, gmm, comps, n_core)
        if n_bnd > 0 and len(X_maj):
            kk, nbr = _boundary_setup(X, y, c, X_c, X_maj)
            clfs = [_ovr_model(X, y, c, seed + 101 * j) for j in range(n_sur)]
            med_p = lambda Z: np.median(
                np.stack([cl.predict_proba(Z)[:, 1] for cl in clfs]), axis=0)
            out, need = [], n_bnd
            for _ in range(6):
                if need <= 0:
                    break
                si = rng.randint(len(X_c), size=need)
                mi = nbr[si, rng.randint(kk, size=need)]
                P, Q = X_c[si], X_maj[mi]
                t = rng.uniform(lo, hi, size=need)
                cand = P + t[:, None] * (Q - P)
                ok = med_p(cand) >= P_THR
                out.extend(cand[ok]); need = n_bnd - len(out)
            if need > 0:
                si = rng.randint(len(X_c), size=need)
                mi = nbr[si, rng.randint(kk, size=need)]
                P, Q = X_c[si], X_maj[mi]
                t = (rng.uniform(lo, hi, size=need)
                     * _flip_points(clfs[0], P, Q, d))
                out.extend(P + t[:, None] * (Q - P))
            syn.extend(out)
        Xa.append(np.array(syn)); ya.append(np.full(len(syn), c))
    return np.vstack(Xa), np.concatenate(ya)


# ── CC-adapt: sertifika-istatistigiyle sinif-bazli sinir payi ───────────────

def augment_ccadapt(X, y, lo=0.25, hi=0.75, seed=42):
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    Xa, ya = [X], [y]
    for c, diff in _minority_iter(X, y):
        X_c, X_maj = X[y == c], X[y != c]
        gmm = B.fit_gmm(X_c, seed=seed)
        comps = _kde_components(X_c, gmm.predict(X_c), gmm.n_components, d)
        if len(X_maj) == 0:
            syn = _core_samples(rng, gmm, comps, diff)
            Xa.append(np.array(syn)); ya.append(np.full(diff, c))
            continue
        kk, nbr = _boundary_setup(X, y, c, X_c, X_maj)
        clf = _ovr_model(X, y, c, seed)
        # pilot parti ile kabul orani → sinir payi
        m = min(400, max(50, diff))
        si = rng.randint(len(X_c), size=m)
        mi = nbr[si, rng.randint(kk, size=m)]
        P, Q = X_c[si], X_maj[mi]
        t = rng.uniform(lo, hi, size=m)
        acc = float((clf.predict_proba(P + t[:, None] * (Q - P))[:, 1]
                     >= P_THR).mean())
        frac = float(np.clip(0.25 + 0.5 * acc, 0.25, 0.75))
        n_bnd = int(diff * frac); n_core = diff - n_bnd
        syn = _core_samples(rng, gmm, comps, n_core)
        out, need = [], n_bnd
        for _ in range(6):
            if need <= 0:
                break
            si = rng.randint(len(X_c), size=need)
            mi = nbr[si, rng.randint(kk, size=need)]
            P, Q = X_c[si], X_maj[mi]
            t = rng.uniform(lo, hi, size=need)
            cand = P + t[:, None] * (Q - P)
            ok = clf.predict_proba(cand)[:, 1] >= P_THR
            out.extend(cand[ok]); need = n_bnd - len(out)
        if need > 0:
            si = rng.randint(len(X_c), size=need)
            mi = nbr[si, rng.randint(kk, size=need)]
            P, Q = X_c[si], X_maj[mi]
            t = rng.uniform(lo, hi, size=need) * _flip_points(clf, P, Q, d)
            out.extend(P + t[:, None] * (Q - P))
        syn.extend(out)
        Xa.append(np.array(syn)); ya.append(np.full(len(syn), c))
    return np.vstack(Xa), np.concatenate(ya)


# ── CC-core: cekirdek orneklerine de sertifika ──────────────────────────────

def augment_cccore(X, y, frac=0.5, lo=0.25, hi=0.75, seed=42):
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    Xa, ya = [X], [y]
    for c, diff in _minority_iter(X, y):
        X_c, X_maj = X[y == c], X[y != c]
        n_bnd = int(diff * frac) if len(X_maj) else 0
        n_core = diff - n_bnd
        gmm = B.fit_gmm(X_c, seed=seed)
        comps = _kde_components(X_c, gmm.predict(X_c), gmm.n_components, d)
        clf = _ovr_model(X, y, c, seed)
        # sertifikali cekirdek
        syn, need = [], n_core
        for _ in range(6):
            if need <= 0:
                break
            cand = np.array(_core_samples(rng, gmm, comps, need))
            ok = clf.predict_proba(cand)[:, 1] >= P_THR
            syn.extend(cand[ok]); need = n_core - len(syn)
        if need > 0:
            syn.extend(_core_samples(rng, gmm, comps, need))
        # sertifikali sinir (CC-SG ile ayni)
        if n_bnd > 0:
            kk, nbr = _boundary_setup(X, y, c, X_c, X_maj)
            out, need = [], n_bnd
            for _ in range(6):
                if need <= 0:
                    break
                si = rng.randint(len(X_c), size=need)
                mi = nbr[si, rng.randint(kk, size=need)]
                P, Q = X_c[si], X_maj[mi]
                t = rng.uniform(lo, hi, size=need)
                cand = P + t[:, None] * (Q - P)
                ok = clf.predict_proba(cand)[:, 1] >= P_THR
                out.extend(cand[ok]); need = n_bnd - len(out)
            if need > 0:
                si = rng.randint(len(X_c), size=need)
                mi = nbr[si, rng.randint(kk, size=need)]
                P, Q = X_c[si], X_maj[mi]
                t = rng.uniform(lo, hi, size=need) * _flip_points(clf, P, Q, d)
                out.extend(P + t[:, None] * (Q - P))
            syn.extend(out)
        Xa.append(np.array(syn)); ya.append(np.full(len(syn), c))
    return np.vstack(Xa), np.concatenate(ya)
