#!/usr/bin/env python3
"""
Deney v3b — Round 2: kalibrasyon mu, kılavuz mu?
═══════════════════════════════════════════════════
Round 1 bulgusu: random/permuted/inverse ağırlıklar gerçek önem kaynaklarını
ve GMM-KDE'yi (küçük, anlamsız farklarla) geçiyor → şüphe: etki özellik-yönlü
kılavuzdan değil, yalnızca KÜRESEL bant genişliği kalibrasyonundan geliyor.

Test 1 (kalibrasyon kontrolleri, önem bilgisi YOK):
  GS-0.85 / GS-0.70 / GS-0.55 : uniform h × c (daraltma)
  GW-1.15 / GW-1.30           : uniform h × c (genişletme)
  Bunlar Round-1 kazananlarının aralığını kapsıyorsa, özellik-yönlü kılavuzun
  katkısı yoktur (kesin negatif sonuç).

Test 2 (son farklı XAI mekanizması): önem-ölçekli uzayda bileşen keşfi.
  wGMM-fisher / wGMM-ovrSHAP : z = x·√w uzayında GMM + Silverman KDE,
                               geri dönüşüm x = z/√w
  wGMM-rand                  : aynı mekanizma, rastgele w (kontrol)

RF, 10 veri kümesi, 5-fold × 3, fold-içi ölçekleme.
Çıktı: results/results_importance_v3b.csv
"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.datasets import fetch_datasets

import benchmark_math_models as B
import eval_framework as E
from run_importance_v3 import imp_ovr_shap, imp_fisher, imp_random, load_all

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

W_FLOOR = 0.2   # wGMM: özelliği tamamen yok etmemek için önem tabanı


def augment_gs(X, y, factor=1.0, seed=42):
    """GMM-KDE, uniform bant çarpanı `factor` (önem bilgisi yok)."""
    rng = np.random.RandomState(seed)
    counts = pd.Series(y).value_counts()
    target = counts.max()
    Xa, ya = [X], [y]
    for c in counts.index:
        if counts[c] >= target:
            continue
        X_c = X[y == c]
        diff = int(target - counts[c])
        gmm = B.fit_gmm(X_c, seed=seed)
        resp = gmm.predict(X_c)
        d = X.shape[1]
        Xk = {k: (X_c[resp == k] if (resp == k).sum() else X_c)
              for k in range(gmm.n_components)}
        syn = []
        for _ in range(diff):
            k = rng.choice(gmm.n_components, p=gmm.weights_)
            Xkk = Xk[k]
            h = (Xkk.std(0) + 1e-6) * (len(Xkk) ** (-1.0 / (d + 4))) * factor
            syn.append(Xkk[rng.randint(len(Xkk))] + rng.normal(0, h))
        Xa.append(np.array(syn))
        ya.append(np.full(diff, c))
    return np.vstack(Xa), np.concatenate(ya)


def augment_wgmm(X, y, imp_fn=None, seed=42):
    """Önem-ölçekli uzayda bileşen keşfi + KDE:
    z = x·√w  (w ∈ [W_FLOOR,1]) → GMM + uniform Silverman KDE → x = z/√w."""
    rng = np.random.RandomState(seed)
    counts = pd.Series(y).value_counts()
    target = counts.max()
    Xa, ya = [X], [y]
    for c in counts.index:
        if counts[c] >= target:
            continue
        w = imp_fn(X, y, c, seed)
        w = W_FLOOR + (1.0 - W_FLOOR) * np.clip(w, 0, 1)
        sw = np.sqrt(w)
        Z = X[y == c] * sw
        diff = int(target - counts[c])
        gmm = B.fit_gmm(Z, seed=seed)
        resp = gmm.predict(Z)
        d = X.shape[1]
        Zk = {k: (Z[resp == k] if (resp == k).sum() else Z)
              for k in range(gmm.n_components)}
        syn = []
        for _ in range(diff):
            k = rng.choice(gmm.n_components, p=gmm.weights_)
            Zkk = Zk[k]
            h = (Zkk.std(0) + 1e-6) * (len(Zkk) ** (-1.0 / (d + 4)))
            z_new = Zkk[rng.randint(len(Zkk))] + rng.normal(0, h)
            syn.append(z_new / sw)
        Xa.append(np.array(syn))
        ya.append(np.full(diff, c))
    return np.vstack(Xa), np.concatenate(ya)


def GS(f):
    return lambda X, y, seed=42: augment_gs(X, y, factor=f, seed=seed)


def WG(fn):
    return lambda X, y, seed=42: augment_wgmm(X, y, imp_fn=fn, seed=seed)


METHODS = {
    "Base":         B.augment_base,
    "GMM-KDE":      GS(1.0),
    "GS-0.85":      GS(0.85),
    "GS-0.70":      GS(0.70),
    "GS-0.55":      GS(0.55),
    "GW-1.15":      GS(1.15),
    "GW-1.30":      GS(1.30),
    "wGMM-fisher":  WG(imp_fisher),
    "wGMM-ovrSHAP": WG(imp_ovr_shap),
    "wGMM-rand":    WG(imp_random),
}


def main():
    t0 = time.time()
    datasets = load_all()
    df = E.run_full(datasets, METHODS, classifiers=["RF"],
                    n_splits=5, n_repeats=3, seed=42, verbose=True, scale=True)
    df.to_csv(os.path.join(OUT, "results_importance_v3b.csv"), index=False)
    fails = pd.DataFrame(E.LAST_FAILURES)
    fails.to_csv(os.path.join(OUT, "failures_importance_v3b.csv"), index=False)
    print(f"\n[OK] results_importance_v3b.csv  Toplam: {(time.time()-t0)/60:.1f} dk")


if __name__ == "__main__":
    main()
