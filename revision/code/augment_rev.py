#!/usr/bin/env python3
"""Revizyon deneyleri — mekanizma kontrolleri (REVIEW_REPORT.md madde 4).
CC-SG iskeleti sabit; degisen tek sey SERTIFIKA VERICISI:
  issuer="xgb" : XGB-30 (CC-SG'nin kendisi, repro)
  issuer="rf"  : RandomForest-30
  issuer="lr"  : LogisticRegression
  issuer="knn" : 3-NN cogunluk oyu (gercek egitim verisi) — ENN-tarzi,
                 ACIKLAMA-NESNESIZ kontrol (Seytanin Avukati testi)
Fallback her varyantta kendi vericisinin cevirme noktasiyla kesme yapar
(mekanizma simetrisi korunur)."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from xgboost import XGBClassifier

import benchmark_math_models as B
from augment_xai_v4 import _minority_iter, _kde_components, CF_GRID
from augment_xai_v6 import _core_samples

LO, HI, FRAC, TAU = 0.25, 0.75, 0.5, 0.5


def _issuer(kind, X, y, c, seed):
    yb = (y == c).astype(int)
    if kind == "xgb":
        clf = XGBClassifier(verbosity=0, random_state=seed, n_estimators=30)
    elif kind == "rf":
        clf = RandomForestClassifier(n_estimators=30, random_state=seed,
                                     n_jobs=-1)
    elif kind == "lr":
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    elif kind == "knn":
        clf = KNeighborsClassifier(n_neighbors=min(3, max(1, (yb == 1).sum())))
    elif kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300,
                            random_state=seed)
    elif kind == "xgbcal":
        from sklearn.calibration import CalibratedClassifierCV
        base = XGBClassifier(verbosity=0, random_state=seed, n_estimators=30)
        cv = min(3, max(2, int((yb == 1).sum())))
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
    else:
        raise ValueError(kind)
    clf.fit(X, yb)
    return clf


def _flip_points(clf, P, Q, d):
    n = len(P)
    ts = np.linspace(0.05, 1.0, CF_GRID)
    G = (P[:, None, :] + ts[None, :, None] * (Q - P)[:, None, :]).reshape(-1, d)
    pr = clf.predict_proba(G)[:, 1].reshape(n, CF_GRID)
    flip = np.where(pr < TAU, 1, 0).argmax(1)
    return np.where((pr >= TAU).all(1), 1.0, ts[flip])


def augment_ccsg_issuer(X, y, issuer="xgb", seed=42):
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    Xa, ya = [X], [y]
    for c, diff in _minority_iter(X, y):
        X_c, X_maj = X[y == c], X[y != c]
        n_bnd = int(diff * FRAC) if len(X_maj) else 0
        n_core = diff - n_bnd
        gmm = B.fit_gmm(X_c, seed=seed)
        comps = _kde_components(X_c, gmm.predict(X_c), gmm.n_components, d)
        syn = _core_samples(rng, gmm, comps, n_core)
        if n_bnd > 0:
            kk = min(5, len(X_maj))
            nn = NearestNeighbors(n_neighbors=kk).fit(X_maj)
            _, nbr = nn.kneighbors(X_c)
            clf = _issuer(issuer, X, y, c, seed)
            out, need = [], n_bnd
            for _ in range(6):
                if need <= 0:
                    break
                si = rng.randint(len(X_c), size=need)
                mi = nbr[si, rng.randint(kk, size=need)]
                P, Q = X_c[si], X_maj[mi]
                t = rng.uniform(LO, HI, size=need)
                cand = P + t[:, None] * (Q - P)
                ok = clf.predict_proba(cand)[:, 1] >= TAU
                out.extend(cand[ok]); need = n_bnd - len(out)
            if need > 0:
                si = rng.randint(len(X_c), size=need)
                mi = nbr[si, rng.randint(kk, size=need)]
                P, Q = X_c[si], X_maj[mi]
                t = rng.uniform(LO, HI, size=need) * _flip_points(clf, P, Q, d)
                out.extend(P + t[:, None] * (Q - P))
            syn.extend(out)
        Xa.append(np.array(syn)); ya.append(np.full(len(syn), c))
    return np.vstack(Xa), np.concatenate(ya)


def ISSUER(kind):
    return lambda X, y, seed=42: augment_ccsg_issuer(X, y, issuer=kind,
                                                     seed=seed)


# ── Strict-fallback varyanti + boru hatti enstrumantasyonu ──────────────────
# Her cagride sinif-bazli kayit: (n_bnd, n_cert, n_fb, n_fb_viol)
FB_STATS = []


def augment_ccsg_strict(X, y, strict=True, seed=42):
    """CC-SG; fallback adaylari son p>=tau testinden gecirilir.
    strict=True: ihlalciler cekirdek ornekleriyle degistirilir.
    strict=False: orijinal davranis (ihlal yalnizca SAYILIR)."""
    import benchmark_math_models as B
    from augment_xai_v4 import _kde_components
    from augment_xai_v6 import _core_samples
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    Xa, ya = [X], [y]
    for c, diff in _minority_iter(X, y):
        X_c, X_maj = X[y == c], X[y != c]
        n_bnd = int(diff * FRAC) if len(X_maj) else 0
        n_core = diff - n_bnd
        gmm = B.fit_gmm(X_c, seed=seed)
        comps = _kde_components(X_c, gmm.predict(X_c), gmm.n_components, d)
        syn = _core_samples(rng, gmm, comps, n_core)
        n_cert = n_fb = n_viol = 0
        if n_bnd > 0:
            kk = min(5, len(X_maj))
            nn = NearestNeighbors(n_neighbors=kk).fit(X_maj)
            _, nbr = nn.kneighbors(X_c)
            clf = _issuer("xgb", X, y, c, seed)
            out, need = [], n_bnd
            for _ in range(6):
                if need <= 0:
                    break
                si = rng.randint(len(X_c), size=need)
                mi = nbr[si, rng.randint(kk, size=need)]
                P, Q = X_c[si], X_maj[mi]
                t = rng.uniform(LO, HI, size=need)
                cand = P + t[:, None] * (Q - P)
                ok = clf.predict_proba(cand)[:, 1] >= TAU
                out.extend(cand[ok]); need = n_bnd - len(out)
            n_cert = len(out)
            if need > 0:
                si = rng.randint(len(X_c), size=need)
                mi = nbr[si, rng.randint(kk, size=need)]
                P, Q = X_c[si], X_maj[mi]
                t = rng.uniform(LO, HI, size=need) * _flip_points(clf, P, Q, d)
                fb = P + t[:, None] * (Q - P)
                viol = clf.predict_proba(fb)[:, 1] < TAU
                n_fb, n_viol = len(fb), int(viol.sum())
                if strict:
                    keep = fb[~viol]
                    out.extend(keep)
                    out.extend(_core_samples(rng, gmm, comps,
                                             n_bnd - len(out)))
                else:
                    out.extend(fb)
            syn.extend(out)
        FB_STATS.append(dict(cls=int(c), n_bnd=n_bnd, n_cert=n_cert,
                             n_fb=n_fb, n_viol=n_viol))
        Xa.append(np.array(syn)); ya.append(np.full(len(syn), c))
    return np.vstack(Xa), np.concatenate(ya)
