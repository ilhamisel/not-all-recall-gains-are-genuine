#!/usr/bin/env python3
"""
Deney v3 — SHAP katkısı kontrollü karşılaştırma (hakem raporu R-yeni #1)
═══════════════════════════════════════════════════════════════════════
Soru: Önem-kılavuzlu bant genişliği, sade GMM-KDE'yi geçebilir mi?
Geçerse hangi önem kaynağı ve hangi formülle?

İki eksen:
  A) Önem kaynağı (sınıf başına, standardize eğitim fold'unda):
     locSHAP  : bileşen-lokal SHAP (mevcut çekirdek)
     ovrSHAP  : sınıf-vs-geri-kalan ikili XGB TreeSHAP (hakem önerisi)
     fisher   : Fisher skoru (sınıf-arası/sınıf-içi varyans; filtre, model yok)
     mi       : mutual information (sklearn, kNN tabanlı)
     rand     : U(0,1) rastgele ağırlık   (kontrol)
     perm     : ovrSHAP'ın permütasyonu   (kontrol)
     inv      : 1 − ovrSHAP               (kontrol)
  B) Bant genişliği formülü:
     sh (shrink)          : h_j × (1 − λ_c s_j + ε)   — mevcut, yalnız daraltır
     vp (variance-pres.)  : shrink faktörleri geometrik-ortalama-1'e
                            normalize edilir → önemli dar, önemsiz GENİŞ;
                            toplam kernel hacmi GMM-KDE ile aynı kalır.
                            (Kılavuz etkisini küresel daralmadan ayırır.)

Bütün varyantlar: GMM bileşen keşfi + Silverman-tipi taban bant + adaptif λ,
ekstrem-rejim modülü YOK (saf çekirdek kıyası). RF, 5-fold × 3 tekrar,
fold-içi StandardScaler. Çıktı: results/results_importance_v3.csv
"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from imblearn.datasets import fetch_datasets

import benchmark_math_models as B
import augment_filtered as F
import eval_framework as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

TOHUM = ["Wine_Red", "Wine_White"]
KEEL = ["ecoli", "us_crime", "oil", "car_eval_4", "yeast_me2",
        "ozone_level", "mammography", "abalone_19"]

EPS = 1e-3


# ── Önem kaynakları (sınıf c için [0,1]^d vektör) ────────────────────────────

def _minmax(v):
    v = np.asarray(v, float)
    rng = v.max() - v.min()
    if rng <= 0:
        return np.zeros_like(v)
    return (v - v.min()) / rng


def imp_ovr_shap(X, y, c, seed):
    yb = (y == c).astype(int)
    return B.compute_shap(X, yb, seed=seed)


def imp_fisher(X, y, c, seed):
    mask = y == c
    Xc, Xr = X[mask], X[~mask]
    num = (Xc.mean(0) - Xr.mean(0)) ** 2
    den = Xc.var(0) + Xr.var(0) + 1e-12
    return _minmax(num / den)


def imp_mi(X, y, c, seed):
    yb = (y == c).astype(int)
    mi = mutual_info_classif(X, yb, random_state=seed)
    return _minmax(mi)


def imp_random(X, y, c, seed):
    rng = np.random.RandomState(seed * 7919 + int(c))
    return rng.uniform(0, 1, X.shape[1])


def imp_perm(X, y, c, seed):
    s = imp_ovr_shap(X, y, c, seed)
    rng = np.random.RandomState(seed * 104729 + int(c))
    return rng.permutation(s)


def imp_inv(X, y, c, seed):
    return 1.0 - imp_ovr_shap(X, y, c, seed)


IMPORTANCE = {
    "ovrSHAP": imp_ovr_shap,
    "fisher":  imp_fisher,
    "mi":      imp_mi,
    "rand":    imp_random,
    "perm":    imp_perm,
    "inv":     imp_inv,
}


# ── Bant genişliği faktörleri ────────────────────────────────────────────────

def bw_factors(s, lam, mode):
    f = 1.0 - lam * np.clip(s, 0, 1) + EPS
    if mode == "vp":
        f = f / np.exp(np.mean(np.log(f)))   # geometrik ortalama → 1
    return f


# ── Genel çekirdek (GMM-KDE + takılabilir önem) ──────────────────────────────

def augment_v3(X, y, imp_name=None, bw_mode="sh", local=False, seed=42):
    """GMM-KDE çekirdeği. imp_name=None → saf GMM-KDE (faktör=1).
    local=True → bileşen-lokal SHAP (mevcut çekirdek; imp_name yok sayılır)."""
    rng = np.random.RandomState(seed)
    counts = pd.Series(y).value_counts()
    target = counts.max()
    Xa, ya = [X], [y]
    g_multi = B.compute_shap(X, y, seed=seed) if local else None

    for c in counts.index:
        if counts[c] >= target:
            continue
        idx_c = np.where(y == c)[0]
        X_c = X[idx_c]
        diff = int(target - counts[c])
        lam_c = F.adaptive_lambda(counts[c], target)

        gmm = B.fit_gmm(X_c, seed=seed)
        resp = gmm.predict(X_c)
        n_comp, weights = gmm.n_components, gmm.weights_

        # bileşen başına örnek havuzu + önem vektörü
        Xk_cache, s_cache = {}, {}
        if imp_name is not None and not local:
            s_class = IMPORTANCE[imp_name](X, y, c, seed)
        for kk in range(n_comp):
            X_k = X_c[resp == kk]
            if len(X_k) == 0:
                X_k = X_c
            Xk_cache[kk] = X_k
            if local:
                s_cache[kk] = B.local_shap(X_c, resp, kk, g_multi, seed)
            elif imp_name is None:
                s_cache[kk] = None
            else:
                s_cache[kk] = s_class

        d = X.shape[1]
        syn = []
        for _ in range(diff):
            kk = rng.choice(n_comp, p=weights)
            X_k = Xk_cache[kk]
            std_k = X_k.std(0) + 1e-6
            h = std_k * (len(X_k) ** (-1.0 / (d + 4)))
            if s_cache[kk] is not None:
                h = h * bw_factors(s_cache[kk], lam_c, bw_mode)
            center = X_k[rng.randint(len(X_k))]
            syn.append(center + rng.normal(0, h))

        syn = np.array(syn)
        Xa.append(syn)
        ya.append(np.full(len(syn), c))

    return np.vstack(Xa), np.concatenate(ya)


def V(**kw):
    def fn(X, y, seed=42):
        return augment_v3(X, y, **kw, seed=seed)
    return fn


METHODS = {
    "Base":        B.augment_base,
    "GMM-KDE":     V(imp_name=None),
    # mevcut çekirdek (bileşen-lokal SHAP, shrink) — süreklilik referansı
    "locSHAP-sh":  V(local=True, bw_mode="sh"),
    "locSHAP-vp":  V(local=True, bw_mode="vp"),
    # sınıf-düzeyi önem kaynakları, shrink formülü
    "ovrSHAP-sh":  V(imp_name="ovrSHAP", bw_mode="sh"),
    "fisher-sh":   V(imp_name="fisher", bw_mode="sh"),
    "mi-sh":       V(imp_name="mi", bw_mode="sh"),
    # kontroller, shrink
    "rand-sh":     V(imp_name="rand", bw_mode="sh"),
    "perm-sh":     V(imp_name="perm", bw_mode="sh"),
    "inv-sh":      V(imp_name="inv", bw_mode="sh"),
    # varyans-koruyan formül
    "ovrSHAP-vp":  V(imp_name="ovrSHAP", bw_mode="vp"),
    "fisher-vp":   V(imp_name="fisher", bw_mode="vp"),
    "mi-vp":       V(imp_name="mi", bw_mode="vp"),
    # kontroller, vp
    "rand-vp":     V(imp_name="rand", bw_mode="vp"),
    "perm-vp":     V(imp_name="perm", bw_mode="vp"),
    "inv-vp":      V(imp_name="inv", bw_mode="vp"),
}


def load_all():
    d = {}
    for n in TOHUM:
        d[n] = B.load_dataset(B.DATASETS[n])
    raw = fetch_datasets(filter_data=tuple(KEEL))
    for n in KEEL:
        d[n] = (np.asarray(raw[n].data, float),
                LabelEncoder().fit_transform(raw[n].target))
    return d


def main():
    t0 = time.time()
    datasets = load_all()
    df = E.run_full(datasets, METHODS, classifiers=["RF"],
                    n_splits=5, n_repeats=3, seed=42, verbose=True, scale=True)
    df.to_csv(os.path.join(OUT, "results_importance_v3.csv"), index=False)
    fails = pd.DataFrame(E.LAST_FAILURES)
    fails.to_csv(os.path.join(OUT, "failures_importance_v3.csv"), index=False)
    print(f"\n[OK] results_importance_v3.csv  Toplam: {(time.time()-t0)/60:.1f} dk")


if __name__ == "__main__":
    main()
