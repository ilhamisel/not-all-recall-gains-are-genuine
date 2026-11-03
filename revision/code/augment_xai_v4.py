#!/usr/bin/env python3
"""
v4 — XAI'nin bant genişliği DIŞINDAKİ entegrasyon noktaları (bkz. XAI_V4_DESIGN.md)
═══════════════════════════════════════════════════════════════════════════════════
Dört mekanizma + XAI-siz eşlenik kontrolleri. Ortak çekirdek: x-uzayı GMM
bileşenleri + Silverman-tipi bant × SHRINK (v3b kazananı 0.85; önem bilgisi yok).

  XComp  : bileşen keşfi SHAP-uzayında (kontrol: rastgele atama)
  XSemi  : bütçenin yarısı vekil-sınır yarı-olgusal örnekleri (kontrol: geometrik)
  XBud   : tohum bütçesi SHAP-atipikliğiyle (kontroller: 1−p sertliği, rastgele)
  XGate  : açıklama-tutarlılığı kabul kapısı (kontrol: x-uzayı merkez kapısı)
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import shap
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

import benchmark_math_models as B

SHRINK = 0.85          # v3b: en güvenli kalibre daraltma (önem bilgisi içermez)
GATE_PCTL = 10         # XGate: gerçek üyelerin benzerlik 10. yüzdeliği eşik
GATE_ROUNDS = 5        # XGate: aday yenileme turu
SEMI_FRAC = 0.5        # XSemi: sınır örneği bütçe payı
CF_GRID = 20           # XSemi: çevirme noktası arama ızgarası
W_FLOOR = 0.05         # XBud: tohum ağırlığı tabanı


# ── Ortak yardımcılar ────────────────────────────────────────────────────────

def _ovr_model(X, y, c, seed):
    clf = XGBClassifier(verbosity=0, random_state=seed, n_estimators=30)
    clf.fit(X, (y == c).astype(int))
    return clf


def _shap_matrix(clf, X):
    """İmzalı örnek-başına SHAP (n,d); TreeExplainer başarısızsa None."""
    try:
        sv = shap.TreeExplainer(clf).shap_values(X)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        return sv
    except Exception:
        return None


def _silverman(Xk, d):
    return (Xk.std(0) + 1e-6) * (len(Xk) ** (-1.0 / (d + 4)))


def _kde_components(X_c, resp, K, d):
    """Bileşen → (üye matrisi, bant) sözlüğü; boş bileşen tüm sınıfa düşer."""
    out = {}
    for k in range(K):
        Xk = X_c[resp == k] if (resp == k).sum() else X_c
        out[k] = (Xk, _silverman(Xk, d) * SHRINK)
    return out


def _minority_iter(X, y):
    counts = pd.Series(y).value_counts()
    target = counts.max()
    for c in counts.index:
        if counts[c] < target:
            yield c, int(target - counts[c])


# ── XComp — SHAP-uzayında bileşen keşfi ─────────────────────────────────────

def augment_xcomp(X, y, mode="shap", seed=42):
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    Xa, ya = [X], [y]
    for c, diff in _minority_iter(X, y):
        X_c = X[y == c]
        if mode == "shap":
            Phi = _shap_matrix(_ovr_model(X, y, c, seed), X_c)
            space = Phi if Phi is not None else X_c
            gmm = B.fit_gmm(space, seed=seed)
            resp = gmm.predict(space)
            K, wts = gmm.n_components, gmm.weights_
        else:                                   # rand: aynı K, rastgele atama
            K = min(3, max(1, len(X_c) // 2))
            resp = rng.randint(0, K, len(X_c))
            wts = np.bincount(resp, minlength=K).astype(float) + 1e-9
            wts /= wts.sum()
        comps = _kde_components(X_c, resp, K, d)
        syn = []
        for _ in range(diff):
            k = rng.choice(K, p=wts)
            Xk, h = comps[k]
            syn.append(Xk[rng.randint(len(Xk))] + rng.normal(0, h))
        Xa.append(np.array(syn)); ya.append(np.full(diff, c))
    return np.vstack(Xa), np.concatenate(ya)


# ── XSemi — yarı-olgusal sınır örneklemesi ──────────────────────────────────
# mode: "geo"    — saf geometrik derinlik (kontrol)
#       "cf"     — kesme: t = depth × t_flip (Round-1 varyantı)
#       "cfgate" — sertifikasyon: geo-derinlikli aday, vekil p<0.5 ise RED +
#                  yeniden örnekleme; kalan açık cf-kesme ile doldurulur.

def _flip_points(clf, P, Q, d):
    """Her (P_i→Q_i) doğrusu için vekilin ilk çevirme noktası t_flip ∈ (0,1]."""
    n = len(P)
    ts = np.linspace(0.05, 1.0, CF_GRID)
    G = (P[:, None, :] + ts[None, :, None] * (Q - P)[:, None, :]).reshape(-1, d)
    pr = clf.predict_proba(G)[:, 1].reshape(n, CF_GRID)
    flip = np.where(pr < 0.5, 1, 0).argmax(1)
    return np.where((pr >= 0.5).all(1), 1.0, ts[flip])


def augment_xsemi(X, y, mode="cf", frac=SEMI_FRAC, depth_lo=0.25,
                  depth_hi=0.75, p_thr=0.5, seed=42):
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    Xa, ya = [X], [y]
    for c, diff in _minority_iter(X, y):
        X_c, X_maj = X[y == c], X[y != c]
        n_bnd = int(diff * frac)
        n_core = diff - n_bnd
        syn = []
        # çekirdek-güvenli kısım: GS-0.85
        gmm = B.fit_gmm(X_c, seed=seed)
        resp = gmm.predict(X_c)
        comps = _kde_components(X_c, resp, gmm.n_components, d)
        for _ in range(n_core):
            k = rng.choice(gmm.n_components, p=gmm.weights_)
            Xk, h = comps[k]
            syn.append(Xk[rng.randint(len(Xk))] + rng.normal(0, h))
        # sınır kısmı
        if n_bnd > 0 and len(X_maj):
            kk = min(5, len(X_maj))
            nn = NearestNeighbors(n_neighbors=kk).fit(X_maj)
            _, nbr = nn.kneighbors(X_c)
            clf = (_ovr_model(X, y, c, seed)
                   if mode in ("cf", "cfgate") else None)
            if mode == "cfgate":
                out, need = [], n_bnd
                for _ in range(6):                # sertifikasyon turları
                    if need <= 0:
                        break
                    si = rng.randint(len(X_c), size=need)
                    mi = nbr[si, rng.randint(kk, size=need)]
                    P, Q = X_c[si], X_maj[mi]
                    t = rng.uniform(depth_lo, depth_hi, size=need)
                    cand = P + t[:, None] * (Q - P)
                    ok = clf.predict_proba(cand)[:, 1] >= p_thr
                    out.extend(cand[ok])
                    need = n_bnd - len(out)
                if need > 0:                       # kalan: cf-kesme (garanti)
                    si = rng.randint(len(X_c), size=need)
                    mi = nbr[si, rng.randint(kk, size=need)]
                    P, Q = X_c[si], X_maj[mi]
                    t = (rng.uniform(depth_lo, depth_hi, size=need)
                         * _flip_points(clf, P, Q, d))
                    out.extend(P + t[:, None] * (Q - P))
                syn.extend(out)
            else:
                si = rng.randint(len(X_c), size=n_bnd)
                mi = nbr[si, rng.randint(kk, size=n_bnd)]
                P, Q = X_c[si], X_maj[mi]
                depth = rng.uniform(depth_lo, depth_hi, size=n_bnd)
                if mode == "cf":
                    t_new = depth * _flip_points(clf, P, Q, d)
                else:                              # geo
                    t_new = depth
                syn.extend(P + t_new[:, None] * (Q - P))
        Xa.append(np.array(syn)); ya.append(np.full(len(syn), c))
    return np.vstack(Xa), np.concatenate(ya)


# ── XBud — açıklama-atipikliği ile bütçe tahsisi ────────────────────────────

def augment_xbud(X, y, mode="shap", seed=42):
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    Xa, ya = [X], [y]
    for c, diff in _minority_iter(X, y):
        X_c = X[y == c]
        if mode == "shap":
            Phi = _shap_matrix(_ovr_model(X, y, c, seed), X_c)
            w = (np.linalg.norm(Phi - Phi.mean(0), axis=1)
                 if Phi is not None else np.ones(len(X_c)))
        elif mode == "prob":
            clf = _ovr_model(X, y, c, seed)
            w = 1.0 - clf.predict_proba(X_c)[:, 1]
        else:                                    # rand
            w = rng.uniform(0, 1, len(X_c))
        w = W_FLOOR + w / (w.max() + 1e-12)
        w /= w.sum()
        gmm = B.fit_gmm(X_c, seed=seed)
        resp = gmm.predict(X_c)
        comps = _kde_components(X_c, resp, gmm.n_components, d)
        si = rng.choice(len(X_c), size=diff, p=w)
        syn = []
        for i in si:
            _, h = comps[resp[i]]
            syn.append(X_c[i] + rng.normal(0, h))
        Xa.append(np.array(syn)); ya.append(np.full(diff, c))
    return np.vstack(Xa), np.concatenate(ya)


# ── XGate — açıklama-tutarlılığı kabul kapısı ───────────────────────────────

def _cos_to(M, v):
    nv = np.linalg.norm(v) + 1e-12
    return (M @ v) / ((np.linalg.norm(M, axis=1) + 1e-12) * nv)


def augment_xgate(X, y, mode="shap", seed=42):
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    Xa, ya = [X], [y]
    for c, diff in _minority_iter(X, y):
        X_c = X[y == c]
        gmm = B.fit_gmm(X_c, seed=seed)
        resp = gmm.predict(X_c)
        K = gmm.n_components
        comps = _kde_components(X_c, resp, K, d)
        clf = _ovr_model(X, y, c, seed) if mode == "shap" else None
        Phi = _shap_matrix(clf, X_c) if mode == "shap" else None
        if mode == "shap" and Phi is None:
            mode = "geo"                          # sessiz geri düşüş
        # bileşen-bazlı referans desen + eşik (gerçek üyelerin 10. yüzdeliği)
        ref, thr = {}, {}
        for k in range(K):
            mk = resp == k
            if mode == "shap":
                Mk = Phi[mk] if mk.sum() >= 2 else Phi
                ref[k] = Mk.mean(0)
                thr[k] = np.percentile(_cos_to(Mk, ref[k]), GATE_PCTL)
            else:
                Xk = X_c[mk] if mk.sum() >= 2 else X_c
                ref[k] = Xk.mean(0)
                dist = np.linalg.norm(Xk - ref[k], axis=1)
                thr[k] = np.percentile(dist, 100 - GATE_PCTL)
        accepted = []
        need = diff
        for _ in range(GATE_ROUNDS):
            if need <= 0:
                break
            m = max(need * 2, 16)
            ks = rng.choice(K, size=m, p=gmm.weights_)
            cand = np.empty((m, d))
            for j, k in enumerate(ks):
                Xk, h = comps[k]
                cand[j] = Xk[rng.randint(len(Xk))] + rng.normal(0, h)
            if mode == "shap":
                Psi = _shap_matrix(clf, cand)
                if Psi is None:
                    ok = np.ones(m, bool)
                else:
                    ok = np.array([_cos_to(Psi[j:j+1], ref[ks[j]])[0] >= thr[ks[j]]
                                   for j in range(m)])
            else:
                ok = np.array([np.linalg.norm(cand[j] - ref[ks[j]]) <= thr[ks[j]]
                               for j in range(m)])
            accepted.extend(cand[ok][:need])
            need = diff - len(accepted)
        while len(accepted) < diff:               # kalanı koşulsuz doldur
            k = rng.choice(K, p=gmm.weights_)
            Xk, h = comps[k]
            accepted.append(Xk[rng.randint(len(Xk))] + rng.normal(0, h))
        Xa.append(np.array(accepted)); ya.append(np.full(diff, c))
    return np.vstack(Xa), np.concatenate(ya)
