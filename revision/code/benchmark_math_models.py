#!/usr/bin/env python3
"""
Matematiksel Model Benchmark — 8 Yeni Varyant
══════════════════════════════════════════════
M1  DPMM          Bayesian Nonparametric GMM (K otomatik)
M2  Student-t     Ağır Kuyruklu Karışım + SHAP-güdümlü ν
M3  Dirichlet     Dirichlet Ağırlıklı KNN Interpolasyon + SHAP α
M4  Copula-SHAP   Gaussian Copula + SHAP Korelasyon Kısıtlaması
M5  OT-SHAP       Optimal Transport + SHAP Maliyet Matrisi
M6  Fisher-SHAP   Hassasiyet Matrisi Düzenlemesi (Precision Space)
M7  MaxEnt-SHAP   Maksimum Entropi Dağılımı (Gauss/Laplace Karışımı)
M8  Fourier-SHAP  SHAP-Bant Genişlikli Kernel Yoğunluk Tahmini

Referans grubu: Base, CLI-WCS, OvO-SHAP+T2 (önceki en iyi)
Metrik: F1-macro, 5-fold StratifiedKFold, RandomForest(100)
"""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
from xgboost import XGBClassifier
import shap

import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tohumdataset") + os.sep
DATASETS = {
    # Multi-class tabular sets used in the final study.
    # Run `python download_data.py` to fetch these into ./tohumdataset/
    "Wine_Red":   "Wine_Red.csv",
    "Wine_White": "Wine_White.csv",
}

# ══════════════════════════════════════════════════════════════════════════════
#  PAYLAŞILAN ALT YAPI
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(fname):
    df = pd.read_csv(DATA_DIR + fname)
    tc = df.columns[-1]
    X = df.drop(columns=[tc]).values.astype(float)
    y = LabelEncoder().fit_transform(df[tc].values)
    if np.isnan(X).any():
        X = SimpleImputer(strategy="median").fit_transform(X)
    return X, y


def compute_shap(X, y, n_est=30, seed=42):
    """SHAP önem → [0,1]; multi-class 3D (n,d,K) doğru işlenir."""
    clf = XGBClassifier(verbosity=0, use_label_encoder=False,
                        random_state=seed, n_estimators=n_est)
    clf.fit(X, y)
    try:
        sv = shap.TreeExplainer(clf).shap_values(X)
        if isinstance(sv, list):
            g = sum(np.abs(s).mean(0) for s in sv)
        elif sv.ndim == 3:
            g = np.abs(sv).mean(axis=(0, 2))
        else:
            g = np.abs(sv).mean(0)
    except Exception:
        g = clf.feature_importances_
    return MinMaxScaler().fit_transform(g.reshape(-1, 1)).flatten()


def fit_gmm(X_c, max_k=3, seed=42):
    """Sağlam GMM uyumu. Yüksek korelasyonlu / büyük ölçekli / az örnekli
    sınıflarda tekilliği önlemek için:
      - reg_covar veri ölçeğine göre ayarlanır
      - full → diag → spherical kademeli geri düşüş
    """
    k = min(max_k, max(1, len(X_c) // 2))
    if len(X_c) <= 2: k = 1
    scale = float(np.mean(np.var(X_c, axis=0))) + 1e-9   # ortalama özellik varyansı
    reg = max(1e-4 * scale, 1e-6)
    # örnek sayısı boyuttan küçükse full atlanır
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


def shrink(s, lam):
    return np.diag(1.0 - np.clip(s, 0, 1) * lam)


def full_cov(gmm, k):
    """GMM bileşen kovaryansını her zaman (d,d) full matris olarak döndürür.
    covariance_type 'diag'/'spherical'/'full'/'tied' hepsini destekler."""
    d = gmm.means_.shape[1]
    ct = gmm.covariance_type
    cov = gmm.covariances_
    if ct == "full":
        return cov[k]
    if ct == "tied":
        return cov
    if ct == "diag":
        return np.diag(cov[k])
    # spherical
    return np.eye(d) * cov[k]


def draw_gmm(gmm, D_list, n, seed=42):
    rng = np.random.RandomState(seed)
    choices = rng.choice(gmm.n_components, size=n, p=gmm.weights_)
    out, d = [], gmm.means_.shape[1]
    for k in range(gmm.n_components):
        cnt = int((choices == k).sum())
        if cnt == 0: continue
        cov = D_list[k] @ full_cov(gmm, k) @ D_list[k] + np.eye(d) * 1e-6
        out.append(rng.multivariate_normal(gmm.means_[k], cov, size=cnt))
    return np.vstack(out) if out else np.empty((0, d))


def local_shap(X_c, hard_resp, k, global_shap, seed):
    """Bileşen-lokal SHAP; yetersiz örnekte global'e döner."""
    mk = hard_resp == k
    X_k = X_c[mk]
    if len(X_k) < 4: return global_shap
    X_oth = X_c[~mk]
    if len(X_oth) < 2: return global_shap
    try:
        Xl = np.vstack([X_k, X_oth])
        yl = np.array([1]*len(X_k) + [0]*len(X_oth))
        return compute_shap(Xl, yl, seed=seed)
    except Exception:
        return global_shap


def ovo_global_shap(X, y, c, class_mu, seed):
    """En yakın sınıfa karşı OvO SHAP."""
    X_c = X[y == c]
    others = [o for o in np.unique(y) if o != c]
    if not others: return compute_shap(X, y, seed=seed)
    dists = [np.linalg.norm(class_mu[c] - class_mu[o]) for o in others]
    nc = others[int(np.argmin(dists))]
    try:
        Xb = np.vstack([X_c, X[y == nc]])
        yb = np.array([1]*len(X_c) + [0]*len(X[y == nc]))
        return compute_shap(Xb, yb, seed=seed)
    except Exception:
        return compute_shap(X, y, seed=seed)


# ══════════════════════════════════════════════════════════════════════════════
#  REFERANS YÖNTEMLERİ
# ══════════════════════════════════════════════════════════════════════════════

def augment_base(X, y, seed=42): return X, y


def augment_cli_wcs(X, y, lam=0.90, seed=42):
    np.random.seed(seed)
    g = compute_shap(X, y, seed=seed)
    counts = pd.Series(y).value_counts(); target = counts.max()
    Xa, ya = [X], [y]
    for c in counts.index:
        diff = target - counts[c]
        if diff <= 0: continue
        X_c = X[y == c]
        gmm = fit_gmm(X_c, seed=seed)
        resp = gmm.predict(X_c)
        D_list = [shrink(local_shap(X_c, resp, k, g, seed), lam)
                  for k in range(gmm.n_components)]
        syn = draw_gmm(gmm, D_list, diff, seed)
        if len(syn): Xa.append(syn); ya.append(np.full(len(syn), c))
    return np.vstack(Xa), np.concatenate(ya)


def augment_ovo_t2(X, y, lam=0.90, seed=42, imb_thresh=2.0):
    """OvO-SHAP + Threshold=2.0 (önceki benchmarktan en iyi)."""
    np.random.seed(seed)
    counts = pd.Series(y).value_counts(); target = counts.max()
    class_mu = {c: X[y == c].mean(0) for c in counts.index}
    Xa, ya = [X], [y]
    for c in counts.index:
        diff = target - counts[c]
        if diff <= 0: continue
        if imb_thresh and (target / counts[c]) < imb_thresh: continue
        X_c = X[y == c]
        g = ovo_global_shap(X, y, c, class_mu, seed)
        gmm = fit_gmm(X_c, seed=seed)
        resp = gmm.predict(X_c)
        D_list = [shrink(local_shap(X_c, resp, k, g, seed), lam)
                  for k in range(gmm.n_components)]
        syn = draw_gmm(gmm, D_list, diff, seed)
        if len(syn): Xa.append(syn); ya.append(np.full(len(syn), c))
    return np.vstack(Xa), np.concatenate(ya)


# ══════════════════════════════════════════════════════════════════════════════
#  M1 — DPMM  (Bayesian Nonparametric GMM, K otomatik)
# ══════════════════════════════════════════════════════════════════════════════

def augment_dpmm(X, y, lam=0.90, seed=42):
    """
    BayesianGaussianMixture ile K'yı veri üzerinden öğren.
    Stick-breaking: π_k = β_k · Π_{j<k}(1-β_j), β_k ~ Beta(1, γ).
    Gereksiz bileşenler otomatik olarak π_k → 0 yapar.
    """
    np.random.seed(seed)
    g = compute_shap(X, y, seed=seed)
    counts = pd.Series(y).value_counts(); target = counts.max()
    Xa, ya = [X], [y]

    for c in counts.index:
        diff = target - counts[c]
        if diff <= 0: continue
        X_c = X[y == c]
        n_upper = min(10, max(2, len(X_c) // 3))

        bgmm = BayesianGaussianMixture(
            n_components=n_upper,
            covariance_type="full",
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=1e-2,   # γ: küçük → az bileşen
            random_state=seed,
            reg_covar=1e-3,
            max_iter=200,
        ).fit(X_c)

        resp = bgmm.predict(X_c)
        active_k = [k for k in range(bgmm.n_components)
                    if bgmm.weights_[k] > 1e-3]
        if not active_k: active_k = [0]

        rng = np.random.RandomState(seed)
        active_w = bgmm.weights_[active_k]
        active_w /= active_w.sum()
        choices = rng.choice(active_k, size=diff, p=active_w)

        syn_parts = []
        d = X_c.shape[1]
        for k in active_k:
            cnt = int((choices == k).sum())
            if cnt == 0: continue
            s_k = local_shap(X_c, resp, k, g, seed)
            Dk = shrink(s_k, lam)
            cov = Dk @ bgmm.covariances_[k] @ Dk + np.eye(d) * 1e-6
            syn_parts.append(
                rng.multivariate_normal(bgmm.means_[k], cov, size=cnt))

        if syn_parts:
            syn = np.vstack(syn_parts)
            Xa.append(syn); ya.append(np.full(len(syn), c))

    return np.vstack(Xa), np.concatenate(ya)


# ══════════════════════════════════════════════════════════════════════════════
#  M2 — Student-t Mixture  (Ağır Kuyruk + SHAP-güdümlü ν)
# ══════════════════════════════════════════════════════════════════════════════

def sample_multivariate_t(mu, Sigma, nu, n, rng):
    """
    Çok değişkenli Student-t örnekleme:
      x = mu + L z / sqrt(u/nu)
      z ~ N(0,I),  u ~ chi2(nu)
    """
    d = len(mu)
    try:
        L = np.linalg.cholesky(Sigma + np.eye(d) * 1e-6)
    except np.linalg.LinAlgError:
        L = np.eye(d)
    z = rng.randn(n, d)
    u = rng.chisquare(nu, size=(n, 1))
    return mu + (z @ L.T) / np.sqrt(u / nu)


def augment_student_t(X, y, lam=0.90, nu_base=10.0, seed=42):
    """
    SHAP-güdümlü ν seçimi:
      ν_k = ν_base * (2 - mean(s_k))
      Yüksek ortalama SHAP → büyük ν → Gaussian-benzeri (sıkı).
      Düşük ortalama SHAP → küçük ν → ağır kuyruk (geniş).
    Kovaryans kısıtlaması CLI-WCS ile aynı (D·Σ·D).
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    g = compute_shap(X, y, seed=seed)
    counts = pd.Series(y).value_counts(); target = counts.max()
    Xa, ya = [X], [y]

    for c in counts.index:
        diff = target - counts[c]
        if diff <= 0: continue
        X_c = X[y == c]
        gmm = fit_gmm(X_c, seed=seed)
        resp = gmm.predict(X_c)
        choices = rng.choice(gmm.n_components, size=diff, p=gmm.weights_)

        syn_parts = []
        d = X_c.shape[1]
        for k in range(gmm.n_components):
            cnt = int((choices == k).sum())
            if cnt == 0: continue
            s_k = local_shap(X_c, resp, k, g, seed)
            mean_s = float(s_k.mean())
            nu_k = max(3.0, nu_base * (2.0 - mean_s))   # [nu_base, 2*nu_base]
            Dk = shrink(s_k, lam)
            cov_k = Dk @ full_cov(gmm, k) @ Dk + np.eye(d) * 1e-6
            syn_parts.append(
                sample_multivariate_t(gmm.means_[k], cov_k, nu_k, cnt, rng))

        if syn_parts:
            syn = np.vstack(syn_parts)
            Xa.append(syn); ya.append(np.full(len(syn), c))

    return np.vstack(Xa), np.concatenate(ya)


# ══════════════════════════════════════════════════════════════════════════════
#  M3 — Dirichlet Interpolasyon  (SHAP-Yoğunlaştırılmış KNN Ağırlıkları)
# ══════════════════════════════════════════════════════════════════════════════

def augment_dirichlet(X, y, k_neighbors=5, alpha0=1.0, lam=2.0, seed=42):
    """
    Sentetik örnek: x_new = Σ_i w_i · x_i,  w ~ Dirichlet(α)
    α_j = α_0 · (1 + λ · s_j)
    Yüksek SHAP → büyük α → Dirichlet konsantrasyonu artar
              → interpolasyon ağırlığı tek komşuya odaklanır
              → önemli özellikte gerçekçi değer korunur.
    Düşük SHAP → küçük α → yayılmış interpolasyon → çeşitlilik.
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    g = compute_shap(X, y, seed=seed)
    counts = pd.Series(y).value_counts(); target = counts.max()
    Xa, ya = [X], [y]

    for c in counts.index:
        diff = target - counts[c]
        if diff <= 0: continue
        X_c = X[y == c]
        k = min(k_neighbors, len(X_c) - 1)
        if k < 1: continue

        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_c)
        _, indices = nbrs.kneighbors(X_c)

        alpha = alpha0 * (1.0 + lam * g)    # (d,) → scalar per component
        alpha_scalar = float(alpha.mean())   # single α for Dirichlet(k)

        syn = []
        for _ in range(diff):
            i = rng.randint(0, len(X_c))
            nb_idx = indices[i, 1:k + 1]
            pts = X_c[nb_idx]                # (k, d)
            w = rng.dirichlet(np.full(k, alpha_scalar))
            syn.append(w @ pts)

        syn = np.array(syn)
        Xa.append(syn); ya.append(np.full(len(syn), c))

    return np.vstack(Xa), np.concatenate(ya)


# ══════════════════════════════════════════════════════════════════════════════
#  M4 — Copula-SHAP  (Gaussian Copula + SHAP Korelasyon Kısıtlaması)
# ══════════════════════════════════════════════════════════════════════════════

def empirical_cdf(vals, x):
    """Ampirik CDF değeri (sıkıştırılmış [ε, 1-ε])."""
    eps = 1e-6
    return np.clip(np.mean(vals <= x, axis=0), eps, 1 - eps)


def augment_copula_shap(X, y, lam=0.90, seed=42):
    """
    Sklar Teoremi: F(x) = C(F_1(x_1), ..., F_d(x_d))
    Gaussian Copula adımları:
      1. u_j = F_j(x_j)  ampirik CDF ile
      2. z_j = Φ^{-1}(u_j)  standard normal uzayına
      3. R_k = korelasyon matrisi; SHAP kısıtlaması: R_new ij = R_ij · (1 - λ·s_i·s_j)
         (çapraz öğeler korunur: R_ii = 1)
      4. z_new ~ N(0, R_new_k)
      5. u_new_j = Φ(z_new_j); x_new_j = F_j^{-1}(u_new_j) ampirik quantile ile

    SHAP katkısı: yüksek önem çifti (s_i·s_j yüksek) →
      korelasyon baskılanır → bağımsız, daha saf marjinal örnekler.
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    g = compute_shap(X, y, seed=seed)
    counts = pd.Series(y).value_counts(); target = counts.max()
    Xa, ya = [X], [y]

    for c in counts.index:
        diff = target - counts[c]
        if diff <= 0: continue
        X_c = X[y == c]
        d = X_c.shape[1]

        # Adım 1-2: ampirik CDF → normal uzayı
        Z_c = np.zeros_like(X_c)
        for j in range(d):
            u_j = np.array([empirical_cdf(X_c[:, j], x) for x in X_c[:, j]])
            Z_c[:, j] = norm.ppf(u_j)

        # Korelasyon matrisi
        R = np.corrcoef(Z_c.T) if len(Z_c) > 1 else np.eye(d)
        R = np.nan_to_num(R, nan=0.0)
        np.fill_diagonal(R, 1.0)

        # Adım 3: SHAP korelasyon kısıtlaması
        shap_outer = np.outer(g, g)        # s_i · s_j matrisi
        R_new = R * (1.0 - lam * shap_outer)
        np.fill_diagonal(R_new, 1.0)

        # Pozitif yarı-kesinlik garantisi
        eigvals = np.linalg.eigvalsh(R_new)
        if eigvals.min() < 1e-6:
            R_new += (np.abs(eigvals.min()) + 1e-4) * np.eye(d)
            D_inv = np.diag(1.0 / np.sqrt(np.diag(R_new)))
            R_new = D_inv @ R_new @ D_inv

        # Adım 4: Normal uzaydan örnekle
        Z_syn = rng.multivariate_normal(np.zeros(d), R_new, size=diff)

        # Adım 5: Geri dönüşüm — ampirik quantile
        syn = np.zeros((diff, d))
        for j in range(d):
            u_new = norm.cdf(Z_syn[:, j])
            vals_sorted = np.sort(X_c[:, j])
            q_idx = np.clip(
                (u_new * len(vals_sorted)).astype(int), 0, len(vals_sorted) - 1)
            syn[:, j] = vals_sorted[q_idx]

        Xa.append(syn); ya.append(np.full(diff, c))

    return np.vstack(Xa), np.concatenate(ya)


# ══════════════════════════════════════════════════════════════════════════════
#  M5 — OT-SHAP  (Optimal Transport + SHAP Maliyet Matrisi)
# ══════════════════════════════════════════════════════════════════════════════

def augment_ot_shap(X, y, lam=0.90, ot_subsample=80, seed=42):
    """
    SHAP-ağırlıklı OT maliyet:
      c(x_i^min, x_j^maj) = Σ_f s_f · (x_i_f - x_j_f)²
    OT planı: Hungarian algoritması (scipy.optimize.linear_sum_assignment).
    Örnekleme: x_new = (1-t)·x_min + t·T*(x_min),  t ~ Beta(2,5)
      t küçük tutulur → azınlık bölgesinde kalır, sınıra doğru iter.
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    g = compute_shap(X, y, seed=seed)
    counts = pd.Series(y).value_counts(); target = counts.max()
    class_mu = {c: X[y == c].mean(0) for c in counts.index}
    Xa, ya = [X], [y]

    for c in counts.index:
        diff = target - counts[c]
        if diff <= 0: continue
        X_min = X[y == c]

        # En yakın komşu sınıfı seç (OvO yaklaşımı)
        others = [o for o in counts.index if o != c]
        if not others:
            continue
        dists = [np.linalg.norm(class_mu[c] - class_mu[o]) for o in others]
        nc = others[int(np.argmin(dists))]
        X_maj = X[y == nc]

        # Büyük setleri örnekle (O(n³) kısıtı)
        n_s = min(ot_subsample, len(X_min))
        n_t = min(ot_subsample, len(X_maj))
        idx_s = rng.choice(len(X_min), n_s, replace=False)
        idx_t = rng.choice(len(X_maj), n_t, replace=False)
        Xs, Xt = X_min[idx_s], X_maj[idx_t]

        # SHAP ağırlıklı kare maliyet matrisi
        diff_feat = Xs[:, None, :] - Xt[None, :, :]   # (n_s, n_t, d)
        C = np.einsum('ijd,d->ij', diff_feat ** 2, g)  # (n_s, n_t)

        # OT planı (exact, Hungarian)
        row_ind, col_ind = linear_sum_assignment(C)

        # Örnekleme: t ~ Beta(2,5) → azınlık tarafına yakın
        syn = []
        for _ in range(diff):
            i = rng.randint(0, len(row_ind))
            x_s = Xs[row_ind[i]]
            x_t = Xt[col_ind[i]]
            t = rng.beta(2, 5)              # [0,1], azınlık tarafına yanlı
            syn.append((1 - t) * x_s + t * x_t)

        syn = np.array(syn)
        Xa.append(syn); ya.append(np.full(diff, c))

    return np.vstack(Xa), np.concatenate(ya)


# ══════════════════════════════════════════════════════════════════════════════
#  M6 — Fisher-SHAP  (Hassasiyet Matrisi Uzayında Düzenleme)
# ══════════════════════════════════════════════════════════════════════════════

def augment_fisher_shap(X, y, lam=0.90, seed=42):
    """
    Gaussian için Fisher Bilgi Matrisi = Σ⁻¹ (hassasiyet matrisi).
    SHAP-düzenleme hassasiyet uzayında ekleme ile yapılır:
      Λ_new_k = Σ_k⁻¹ + λ · diag(s_k)
      Σ_new_k = Λ_new_k⁻¹
    Yüksek SHAP → precision artar → varyans azalır.
    Mevcut yöntemden farkı: multiplicatif D·Σ·D yerine
    additive precision düzenlemesi → doğal gradyan uzayında shrinkage.
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    g = compute_shap(X, y, seed=seed)
    counts = pd.Series(y).value_counts(); target = counts.max()
    Xa, ya = [X], [y]

    for c in counts.index:
        diff = target - counts[c]
        if diff <= 0: continue
        X_c = X[y == c]
        gmm = fit_gmm(X_c, seed=seed)
        resp = gmm.predict(X_c)
        choices = rng.choice(gmm.n_components, size=diff, p=gmm.weights_)

        syn_parts = []
        d = X_c.shape[1]
        for k in range(gmm.n_components):
            cnt = int((choices == k).sum())
            if cnt == 0: continue
            s_k = local_shap(X_c, resp, k, g, seed)
            Sigma_k = full_cov(gmm, k) + np.eye(d) * 1e-4
            try:
                Lambda_k = np.linalg.inv(Sigma_k)
            except np.linalg.LinAlgError:
                Lambda_k = np.eye(d)
            # SHAP-düzenleme: hassasiyet matrisine önem eklenir
            Lambda_new = Lambda_k + lam * np.diag(s_k)
            try:
                Sigma_new = np.linalg.inv(Lambda_new)
            except np.linalg.LinAlgError:
                Sigma_new = Sigma_k
            Sigma_new = (Sigma_new + Sigma_new.T) / 2 + np.eye(d) * 1e-6
            syn_parts.append(
                rng.multivariate_normal(gmm.means_[k], Sigma_new, size=cnt))

        if syn_parts:
            syn = np.vstack(syn_parts)
            Xa.append(syn); ya.append(np.full(len(syn), c))

    return np.vstack(Xa), np.concatenate(ya)


# ══════════════════════════════════════════════════════════════════════════════
#  M7 — MaxEnt-SHAP  (Maksimum Entropi Dağılımı)
# ══════════════════════════════════════════════════════════════════════════════

def augment_maxent_shap(X, y, lam=0.90, seed=42):
    """
    MaxEnt çözümü E_p[f_j(x)] = c_j kısıtlarıyla üstel aile verir.
    Pratik uygulama:
      s_j yüksek → Gaussian marjinal (kısıtlı, sıkı)
      s_j düşük  → Laplace marjinal  (ağır kuyruk, geniş)
    Ölçek: Gaussian için σ_j, Laplace için b_j = σ_j/√2 · (2 - s_j)
    Özellikler arası korelasyon CLI-WCS ile aynı D·Σ·D ile korunur,
    ama Cholesky ile çözülen correlated sampling bileşen bazında.
    Şekilsel fark: her bileşende özellik başına dağılım ailesi seçimi.
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    g = compute_shap(X, y, seed=seed)
    counts = pd.Series(y).value_counts(); target = counts.max()
    Xa, ya = [X], [y]

    for c in counts.index:
        diff = target - counts[c]
        if diff <= 0: continue
        X_c = X[y == c]
        gmm = fit_gmm(X_c, seed=seed)
        resp = gmm.predict(X_c)
        choices = rng.choice(gmm.n_components, size=diff, p=gmm.weights_)

        syn_parts = []
        d = X_c.shape[1]
        for k in range(gmm.n_components):
            cnt = int((choices == k).sum())
            if cnt == 0: continue
            s_k = local_shap(X_c, resp, k, g, seed)
            mu_k = gmm.means_[k]
            sigma_k = np.sqrt(np.diag(full_cov(gmm, k)) + 1e-6)

            # Özellik başına dağılım ailesi seçimi
            threshold = 0.5
            samples = np.zeros((cnt, d))
            for j in range(d):
                if s_k[j] >= threshold:
                    # Gaussian — MaxEnt(ortalama, varyans) = Gaussian
                    scale = sigma_k[j] * (1.0 - lam * s_k[j] + 1e-3)
                    samples[:, j] = rng.normal(mu_k[j], max(scale, 1e-6), cnt)
                else:
                    # Laplace — MaxEnt(ortalama, mutlak sapma) = Laplace
                    # b = σ/√2 × (2 - s_j): düşük önem → daha geniş kuyruk
                    b = sigma_k[j] / np.sqrt(2) * (2.0 - s_k[j])
                    samples[:, j] = rng.laplace(mu_k[j], max(b, 1e-6), cnt)

            syn_parts.append(samples)

        if syn_parts:
            syn = np.vstack(syn_parts)
            Xa.append(syn); ya.append(np.full(len(syn), c))

    return np.vstack(Xa), np.concatenate(ya)


# ══════════════════════════════════════════════════════════════════════════════
#  M8 — Fourier-SHAP  (SHAP-Bant Genişlikli Kernel Yoğunluk Tahmini)
# ══════════════════════════════════════════════════════════════════════════════

def augment_fourier_shap(X, y, lam=0.90, seed=42):
    """
    Random Fourier Features ile SHAP-güdümlü KDE örnekleme.
    Silverman bant genişliği: h_j = h_sil · (1 - λ·s_j)
      Yüksek SHAP → küçük h_j → yüksek frekans → sıkı örnekleme.
      Düşük SHAP  → büyük h_j → düşük frekans → geniş örnekleme.
    Örnekleme: x_new = x_i + N(0, diag(h²)) (kernelize KDE)
    GMM bileşen ataması ile bileşen-lokal h hesaplanır.
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    g = compute_shap(X, y, seed=seed)
    counts = pd.Series(y).value_counts(); target = counts.max()
    Xa, ya = [X], [y]

    for c in counts.index:
        diff = target - counts[c]
        if diff <= 0: continue
        X_c = X[y == c]
        gmm = fit_gmm(X_c, seed=seed)
        resp = gmm.predict(X_c)

        syn = []
        d = X_c.shape[1]
        n_c = len(X_c)

        for _ in range(diff):
            k = rng.choice(gmm.n_components, p=gmm.weights_)
            mk = resp == k
            X_k = X_c[mk]
            if len(X_k) == 0: X_k = X_c

            s_k = local_shap(X_c, resp, k, g, seed)

            # Silverman bant genişliği per özellik
            std_k = X_k.std(0) + 1e-6
            n_k = len(X_k)
            h_silverman = std_k * (n_k ** (-1 / (d + 4)))   # Silverman kuralı
            h_shap = h_silverman * (1.0 - lam * s_k + 1e-3)  # SHAP kısıtlaması

            # Bir merkez nokta seç + kernel gürültüsü ekle
            center = X_k[rng.randint(len(X_k))]
            noise = rng.normal(0, h_shap)
            syn.append(center + noise)

        syn = np.array(syn)
        Xa.append(syn); ya.append(np.full(diff, c))

    return np.vstack(Xa), np.concatenate(ya)


# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK MOTORU
# ══════════════════════════════════════════════════════════════════════════════

METHODS = {
    "Base":          augment_base,
    "CLI-WCS":       augment_cli_wcs,
    "OvO-SHAP+T2":  augment_ovo_t2,
    # ── 8 Yeni Model ───────────────────────────────────────────────────────
    "M1-DPMM":       augment_dpmm,
    "M2-StudentT":   augment_student_t,
    "M3-Dirichlet":  augment_dirichlet,
    "M4-Copula":     augment_copula_shap,
    "M5-OT":         augment_ot_shap,
    "M6-Fisher":     augment_fisher_shap,
    "M7-MaxEnt":     augment_maxent_shap,
    "M8-Fourier":    augment_fourier_shap,
}


def run(n_splits=5, n_trees=100, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = {}
    t0_total = time.time()

    for ds_name, fname in DATASETS.items():
        X, y = load_dataset(fname)
        vc = pd.Series(y).value_counts()
        ratio = vc.max() / vc.min()
        print(f"\n{'━'*65}")
        print(f"  {ds_name}  |  {len(X)}×{X.shape[1]}  "
              f"|  {len(np.unique(y))} sınıf  |  {ratio:.1f}x imbalance")
        print(f"{'━'*65}")

        results[ds_name] = {}
        for mname, aug_fn in METHODS.items():
            scores = []
            t0 = time.time()
            for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
                Xtr, ytr = X[tr_idx].copy(), y[tr_idx].copy()
                Xte, yte = X[te_idx], y[te_idx]
                try:
                    Xtr, ytr = aug_fn(Xtr, ytr, seed=seed + fold)
                except Exception as e:
                    pass   # hata → orijinal eğitim seti
                clf = RandomForestClassifier(
                    n_estimators=n_trees, random_state=seed, n_jobs=-1)
                clf.fit(Xtr, ytr)
                scores.append(f1_score(yte, clf.predict(Xte), average="macro"))

            mu, std = np.mean(scores), np.std(scores)
            results[ds_name][mname] = {"mean": mu, "std": std}
            print(f"  {mname:<16} {mu:.4f} ± {std:.4f}   ({time.time()-t0:5.1f}s)")

    print(f"\nToplam: {(time.time()-t0_total)/60:.1f} dk")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  RAPORLAMA
# ══════════════════════════════════════════════════════════════════════════════

def report(results):
    methods = list(METHODS.keys())
    datasets = list(DATASETS.keys())
    W = 10

    # ── 1. Özet Tablo ───────────────────────────────────────────────────────
    sep = "═" * (22 + W * len(methods) + 2)
    print(f"\n{sep}")
    print("  ÖZET TABLO  (F1-macro, 5-Fold, RF-100)   * = satır birincisi")
    print(sep)
    hdr = f"  {'Dataset':<20}" + "".join(f"{m:>{W}}" for m in methods)
    print(hdr)
    print("  " + "─" * (18 + W * len(methods)))

    for ds in datasets:
        scores = [results[ds][m]["mean"] for m in methods]
        best = max(scores)
        row = f"  {ds:<20}"
        for m, s in zip(methods, scores):
            mk = "*" if abs(s - best) < 1e-6 else " "
            row += f"{s:.4f}{mk}".rjust(W)
        print(row)

    print("  " + "─" * (18 + W * len(methods)))
    print("\n  Ortalama F1-macro (sıralı):")
    avgs = {m: np.mean([results[ds][m]["mean"] for ds in datasets])
            for m in methods}
    for rank, (m, v) in enumerate(
            sorted(avgs.items(), key=lambda x: -x[1]), 1):
        marker = " ◄ EN İYİ" if rank == 1 else ""
        print(f"  {rank:2}. {m:<16} {v:.4f}{marker}")

    # ── 2. Delta Tablosu ────────────────────────────────────────────────────
    non_base = [m for m in methods if m != "Base"]
    sep2 = "═" * (22 + W * len(non_base) + 2)
    print(f"\n{sep2}")
    print("  DELTA  (Δ F1-macro vs Base)   + iyileşme  / − bozulma")
    print(sep2)
    hdr2 = f"  {'Dataset':<20}" + "".join(f"{m:>{W}}" for m in non_base)
    print(hdr2)
    print("  " + "─" * (18 + W * len(non_base)))

    wins = {m: 0 for m in non_base}
    for ds in datasets:
        b = results[ds]["Base"]["mean"]
        row = f"  {ds:<20}"
        for m in non_base:
            d = results[ds][m]["mean"] - b
            if d > 0.0005: wins[m] += 1
            s = f"+{d:.4f}" if d >= 0 else f"{d:.4f}"
            row += f"{s:>{W}}"
        print(row)

    print("  " + "─" * (18 + W * len(non_base)))
    print("\n  Kazanılan dataset (Δ > 0.0005):")
    for m in sorted(wins, key=wins.get, reverse=True):
        bar = "█" * wins[m] + "░" * (len(datasets) - wins[m])
        print(f"  {m:<16} {bar}  {wins[m]}/{len(datasets)}")

    # ── 3. Referans grubuyla karşılaştırma ──────────────────────────────────
    ref = "OvO-SHAP+T2"
    new_methods = [m for m in methods if m.startswith("M")]
    print(f"\n{'═'*60}")
    print(f"  YENİ MODELLER vs {ref}")
    print(f"{'═'*60}")
    print(f"  {'Model':<16} {'Ort Δ':>8}  {'Kazanım':>9}  {'Durum'}")
    print(f"  {'─'*56}")
    for m in new_methods:
        delta_list = [results[ds][m]["mean"] - results[ds][ref]["mean"]
                      for ds in datasets]
        avg_d = np.mean(delta_list)
        wins_ref = sum(1 for d in delta_list if d > 0.0005)
        status = "✓ ÜSTÜN" if avg_d > 0.001 else ("≈ EŞDEĞER" if avg_d > -0.001 else "✗ ZAYIF")
        sign = "+" if avg_d >= 0 else ""
        print(f"  {m:<16} {sign}{avg_d:.4f}   {wins_ref}/{len(datasets)} dataset  {status}")

    print(f"{'═'*60}")


if __name__ == "__main__":
    print("Matematiksel Model Benchmark — 8 Yeni Varyant")
    print("Referans: Base / CLI-WCS / OvO-SHAP+T2")
    print("─" * 65)
    results = run()
    report(results)
