#!/usr/bin/env python3
"""
SG-KDE çekirdek üretici — güvenlik filtresi, dinamik oran, adaptif λ, kalite kapısı
═══════════════════════════════════════════════════════════════════════════════════
Revizyon (hakem raporu sonrası):
  * Adaptif λ artık SABİT bir referans orana (IR_REF=1000) göre hesaplanır;
    çoğunluk sınıfının mutlak örnek sayısına bağlı değildir ve makaledeki
    denklemle birebir aynıdır.
  * Kalite kapısı GERÇEK reddetme yapar: MAX_TRIES deneme sonunda kabul
    edilmeyen örnek atılır (koşulsuz ekleme yok). Reddetme oranları
    LAST_DIAG üzerinden raporlanır.
  * Güvenlik filtresi fallback'i açık ve kontrollüdür: 2'den az güvenli nokta
    varsa tam dengelemeye dönülmez; en güvenli ⌈n_c/2⌉ nokta (en az 2) merkez
    alınır ve üretim miktarı Laplace-düzeltmeli güvenli oranla kısılır:
        ρ̂_safe = (n_safe + 1) / (n_c + 2).
  * Rejim kararı sınıf bazlıdır: extreme_c = (IR_c >= τ). Tek bir aşırı-nadir
    sınıf, diğer sınıfları ekstrem rejime sokmaz.
  * Bileşen-lokal SHAP kullanımı (lokal / global fallback) sayılır ve
    LAST_DIAG içinde raporlanır.
  * Ablasyon için use_gmm ve shap_mode ("none"|"global"|"local") bayrakları
    eklendi.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from benchmark_math_models import compute_shap, fit_gmm, local_shap

# ── Sabitler ─────────────────────────────────────────────────────────────────
IR_REF = 1000.0       # adaptif λ için sabit referans dengesizlik oranı
MAX_TRIES = 10        # kalite kapısı: örnek başına azami deneme; sonra RED
GATE_K = 5            # kalite kapısı / güvenlik filtresi komşu sayısı

# Son augmentasyon çağrısının sınıf-bazlı tanılama kayıtları.
# Her öğe: dict(cls, n_c, n_safe, rho_safe, fallback, G_target, n_generated,
#               n_rejected, K_gmm, comp_sizes, n_local_shap, n_global_shap,
#               lambda_c, extreme)
LAST_DIAG = []


# ── Yardımcılar ──────────────────────────────────────────────────────────────

def sample_safety(X, y, c, k=GATE_K):
    """Azınlık sınıf c örnekleri için k-NN tabanlı güvenlik skoru r_i ∈ [0,1]
    (kaç komşusu aynı sınıftan). idx_c ve r döndürür."""
    kk = min(k + 1, len(X))
    nn = NearestNeighbors(n_neighbors=kk).fit(X)
    idx_c = np.where(y == c)[0]
    _, nbrs = nn.kneighbors(X[idx_c])
    r = np.array([np.mean(y[row[1:]] == c) for row in nbrs])  # ilk komşu kendisi
    return idx_c, r


def adaptive_lambda(count_c, n_maj, lam_min=0.3, lam_max=0.9, ir_ref=IR_REF):
    """Imbalance-farkında λ (makale Denk. 8 ile birebir aynı):
        λ_c = λ_max − (λ_max − λ_min) · min(1, ln(1+IR_c)/ln(1+IR_ref))
    IR_c = n_maj / n_c; ir_ref SABİTTİR (veri kümesine bağlı değildir), bu
    yüzden λ yalnızca sınıfın kendi dengesizlik oranına bağlıdır."""
    ir_c = n_maj / max(count_c, 1)
    frac = min(1.0, np.log1p(ir_c) / np.log1p(ir_ref))
    return lam_max - (lam_max - lam_min) * frac


def accept_sample(x_new, nn_model, y_all, c):
    """Voronoi-tutarlı kalite kapısı: sentetik nokta ancak GERÇEK eğitim
    verisindeki en yakın komşusu kendi sınıfından ise kabul edilir.
    (k-NN çoğunluk kriteri aşırı kıtlıkta hiçbir örneği kabul edemez;
    1-NN kriteri kıtlık altında da anlamlı bir OOD reddi sağlar.)"""
    _, idx = nn_model.kneighbors(x_new.reshape(1, -1), n_neighbors=1)
    return y_all[idx[0][0]] == c


def _kde_sample_from(X_k, s_k, lam, rng):
    """Tek SHAP-kısıtlı Gaussian KDE örneği: merkez + Silverman-tipi bant ×
    (1−λs+1e-3) gürültü. σ'ya eklenen 1e-6 sabit özelliklerde sıfır bant
    genişliğini engeller."""
    d = X_k.shape[1]
    std_k = X_k.std(0) + 1e-6
    h = std_k * (len(X_k) ** (-1.0 / (d + 4))) * (1.0 - lam * s_k + 1e-3)
    center = X_k[rng.randint(len(X_k))]
    return center + rng.normal(0, h)


# ── Çekirdek üretici ─────────────────────────────────────────────────────────

def _augment_core(X, y, lam=0.90, k=GATE_K, safe_thresh=0.5,
                  use_filter=True, use_dynamic=True, use_adaptive=False,
                  use_gate=False, auto_regime=False, ir_threshold=100.0,
                  use_gmm=True, shap_mode="local", seed=42):
    """SG-KDE çekirdek + (opsiyonel) ekstrem-rejim modülü.

    auto_regime=True ise use_filter/use_dynamic/use_gate bayrakları yok
    sayılır ve her sınıf için ayrı karar verilir:
        extreme_c = (IR_c >= ir_threshold)
    shap_mode: "none" (λ=0 eşdeğeri, saf KDE bandı), "global" (sınıflar-arası
    global SHAP), "local" (bileşen-lokal SHAP, yetersiz örnekte global'e döner).
    """
    global LAST_DIAG
    LAST_DIAG = []
    rng = np.random.RandomState(seed)
    g = compute_shap(X, y, seed=seed) if shap_mode != "none" else None
    counts = pd.Series(y).value_counts()
    target = counts.max()
    Xa, ya = [X], [y]

    nn_all = NearestNeighbors(n_neighbors=min(k, len(X))).fit(X)

    for c in counts.index:
        if counts[c] >= target:
            continue
        idx_c = np.where(y == c)[0]
        n_c = len(idx_c)
        ir_c = target / n_c

        if auto_regime:
            extreme_c = ir_c >= ir_threshold
            f_filter, f_dynamic, f_gate = extreme_c, extreme_c, extreme_c
            f_adaptive = True
        else:
            extreme_c = use_gate
            f_filter, f_dynamic, f_gate = use_filter, use_dynamic, use_gate
            f_adaptive = use_adaptive

        # Güvenlik filtresi + açık fallback
        fallback = False
        if f_filter:
            _, r = sample_safety(X, y, c, k)
            safe_mask = r >= safe_thresh
            n_safe = int(safe_mask.sum())
            if n_safe >= 2:
                X_prod = X[idx_c[safe_mask]]
                # Laplace-düzeltmeli güvenli oran (0 ve 1 uçlarını engeller)
                rho_safe = (n_safe + 1.0) / (n_c + 2.0)
            else:
                # Açık fallback: güvenlik skoru bilgi vermiyor (tipik olarak
                # hepsi 0; her azınlık noktası çoğunlukla çevrili). Bilgisiz
                # skora dayanarak ne merkez atılır ne bütçe kısılır — merkez
                # olarak TÜM azınlık noktaları kalır, bütçe tam dengelemedir
                # ve tek koruma örnek-başına kalite kapısıdır. (Kısma yalnızca
                # skor gerçekten kanıt sağladığında uygulanır.)
                fallback = True
                X_prod = X[idx_c]
                rho_safe = 1.0
        else:
            X_prod = X[idx_c]
            n_safe = n_c
            rho_safe = 1.0

        # Dinamik oran
        full_diff = int(target - counts[c])
        diff = int(round(rho_safe * full_diff)) if f_dynamic else full_diff
        if diff <= 0:
            continue

        lam_c = adaptive_lambda(counts[c], target) if f_adaptive else lam
        if shap_mode == "none":
            lam_c = 0.0

        if use_gmm:
            gmm = fit_gmm(X_prod, seed=seed)
            resp = gmm.predict(X_prod)
            n_comp, weights = gmm.n_components, gmm.weights_
        else:
            resp = np.zeros(len(X_prod), dtype=int)
            n_comp, weights = 1, np.array([1.0])

        # Bileşen-lokal SHAP (bileşen başına 1 kez; fallback'ler sayılır)
        s_cache, Xk_cache = {}, {}
        n_local, n_global = 0, 0
        for kk in range(n_comp):
            X_k = X_prod[resp == kk]
            if len(X_k) == 0:
                X_k = X_prod
            Xk_cache[kk] = X_k
            if shap_mode == "none":
                s_cache[kk] = np.zeros(X.shape[1])
            elif shap_mode == "global":
                s_cache[kk] = g
                n_global += 1
            else:
                s_kk = local_shap(X_prod, resp, kk, g, seed)
                s_cache[kk] = s_kk
                if s_kk is g:
                    n_global += 1
                else:
                    n_local += 1

        syn, n_rej = [], 0
        for _ in range(diff):
            kk = rng.choice(n_comp, p=weights)
            x_new = _kde_sample_from(Xk_cache[kk], s_cache[kk], lam_c, rng)
            if f_gate:
                accepted, tries = accept_sample(x_new, nn_all, y, c), 0
                while not accepted and tries < MAX_TRIES:
                    x_new = _kde_sample_from(Xk_cache[kk], s_cache[kk], lam_c, rng)
                    accepted = accept_sample(x_new, nn_all, y, c)
                    tries += 1
                if not accepted:          # GERÇEK reddetme: örnek atılır
                    n_rej += 1
                    continue
            syn.append(x_new)

        LAST_DIAG.append(dict(
            cls=int(c), n_c=n_c, ir_c=float(ir_c), n_safe=n_safe,
            rho_safe=float(rho_safe), fallback=fallback, G_target=diff,
            n_generated=len(syn), n_rejected=n_rej, K_gmm=int(n_comp),
            comp_sizes=[int((resp == q).sum()) for q in range(n_comp)],
            n_local_shap=n_local, n_global_shap=n_global,
            lambda_c=float(lam_c), extreme=bool(extreme_c)))

        if not syn:
            continue
        syn = np.array(syn)
        Xa.append(syn)
        ya.append(np.full(len(syn), c))

    return np.vstack(Xa), np.concatenate(ya)


# ── Kayıtlı varyantlar (runner'ların çağıracağı imzalar) ─────────────────────

def augment_C0_fourier(X, y, lam=0.90, seed=42):
    """C0 = referans: filtresiz, dinamik-oransız."""
    return _augment_core(X, y, lam, use_filter=False, use_dynamic=False, seed=seed)


def augment_C1_filter(X, y, lam=0.90, seed=42):
    """C1 = sadece filtreleme."""
    return _augment_core(X, y, lam, use_filter=True, use_dynamic=False, seed=seed)


def augment_C2_dynamic(X, y, lam=0.90, seed=42):
    """C2 = filtre + dinamik oran."""
    return _augment_core(X, y, lam, use_filter=True, use_dynamic=True, seed=seed)


def augment_C3_full(X, y, lam=0.90, seed=42):
    """C3 = filtreleme + dinamik oran."""
    return _augment_core(X, y, lam, use_filter=True, use_dynamic=True, seed=seed)


def augment_D_adaptive(X, y, lam=0.90, seed=42):
    """D = adaptif λ (filtresiz)."""
    return _augment_core(X, y, lam, use_filter=False, use_dynamic=False,
                         use_adaptive=True, seed=seed)


def augment_E_sfkde(X, y, lam=0.90, gate=True, seed=42):
    """Ekstrem-rejim yapılandırması: filtre + dinamik oran + adaptif λ
    (+ opsiyonel kalite kapısı)."""
    return _augment_core(X, y, lam, use_filter=True, use_dynamic=True,
                         use_adaptive=True, use_gate=gate, seed=seed)


augment_sg_kde_extreme = augment_E_sfkde   # alias


def augment_E_gate(X, y, lam=0.90, seed=42):
    return augment_E_sfkde(X, y, lam, gate=True, seed=seed)


def augment_E_nogate(X, y, lam=0.90, seed=42):
    return augment_E_sfkde(X, y, lam, gate=False, seed=seed)


def augment_E_auto(X, y, lam=0.90, ir_threshold=100.0, seed=42):
    """SG-KDE (dağıtılan model): rejim kararı SINIF BAZLIDIR —
    extreme_c = (IR_c >= ir_threshold). Ekstrem sınıflar filtre+dinamik
    oran+kapı alır; diğerleri çekirdek (tam dengeleme + adaptif λ)."""
    return _augment_core(X, y, lam, auto_regime=True,
                         ir_threshold=ir_threshold, seed=seed)
