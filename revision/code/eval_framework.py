#!/usr/bin/env python3
"""
Deney A — Sağlam Değerlendirme Altyapısı (hakem revizyonu sonrası)
═══════════════════════════════════════════════════════════════════
- Çok metrik : F1-macro, G-mean (ε-düzeltmeli + katı), MCC,
               azınlık-recall (en küçük %25 sınıf ortalaması),
               rare-recall (tek en nadir sınıf)
- Çok sınıflandırıcı : RandomForest, XGBoost, LogisticRegression
- Tekrarlı CV : n_splits × n_repeats fold
- Ölçekleme : her fold'da StandardScaler YALNIZCA eğitim verisine fit edilir;
              augmentasyon ölçeklenmiş uzayda yapılır, test aynı scaler ile
              dönüştürülür (tüm yöntemler için ortak → adil kıyas).
- İstatistik : fold-bazlı Wilcoxon YALNIZCA tanımlayıcı olarak kaydedilir;
              yayın istatistiği run_stats.py'deki veri-kümesi-bazlı
              Friedman + Holm analizidir.
- Hatalar   : sessizce yutulmaz; tür/mesaj/fold LAST_FAILURES'a kaydedilir
              ve yöntem o fold'da Base'e döner (n_fail sayacı artar).

Augmentasyon YALNIZCA eğitim fold'una uygulanır (leakage yok).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (f1_score, matthews_corrcoef, confusion_matrix)
from xgboost import XGBClassifier
from scipy.stats import wilcoxon

# Son run_full çağrısındaki augmentasyon hataları:
# dict(dataset, method, rep, fold, exc_type, exc_msg)
LAST_FAILURES = []


# ── METRİKLER ────────────────────────────────────────────────────────────────

def _per_class_recall(y_true, y_pred):
    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rec = np.diag(cm) / np.clip(cm.sum(axis=1), 1, None)
    return labels, rec


def gmean_macro(y_true, y_pred, eps=1e-6):
    """ε-düzeltmeli makro G-mean: recall'lar alttan eps ile kırpılır.
    (Sıfır-recall fold'larını karşılaştırılabilir tutar; katı sürüm için
    gmean_strict.)"""
    _, recalls = _per_class_recall(y_true, y_pred)
    recalls = np.clip(recalls, eps, None)
    return float(np.exp(np.mean(np.log(recalls))))


def gmean_strict(y_true, y_pred):
    """Katı makro G-mean: herhangi bir sınıf recall'u 0 ise sonuç 0."""
    _, recalls = _per_class_recall(y_true, y_pred)
    if np.any(recalls <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(recalls))))


def minority_recall(y_true, y_pred, frac=0.25):
    """En küçük `frac` oranındaki sınıfların ORTALAMA recall'u.
    (Çok sınıflı setlerde tek en nadir sınıf DEĞİLDİR → rare_recall'a bak.)"""
    vc = pd.Series(y_true).value_counts()
    n_min = max(1, int(round(len(vc) * frac)))
    small = list(vc.tail(n_min).index)
    labels, rec = _per_class_recall(y_true, y_pred)
    rmap = dict(zip(labels, rec))
    return float(np.mean([rmap[s] for s in small]))


def rare_recall(y_true, y_pred):
    """Tek en nadir sınıfın recall'u."""
    vc = pd.Series(y_true).value_counts()
    rarest = vc.index[-1]
    labels, rec = _per_class_recall(y_true, y_pred)
    rmap = dict(zip(labels, rec))
    return float(rmap[rarest])


METRICS = {
    "f1_macro":     lambda yt, yp: f1_score(yt, yp, average="macro"),
    "gmean":        gmean_macro,
    "gmean_strict": gmean_strict,
    "mcc":          matthews_corrcoef,
    "min_recall":   minority_recall,
    "rare_recall":  rare_recall,
}


class _WeightedXGB(XGBClassifier):
    """XGB'nin class_weight'i yok; fit'te balanced sample_weight uygular."""
    def fit(self, X, y, **kw):
        from sklearn.utils.class_weight import compute_sample_weight
        sw = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sw, **kw)


def make_classifier(name, seed=42):
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    if name == "XGB":
        return XGBClassifier(n_estimators=100, verbosity=0, random_state=seed,
                             n_jobs=-1)
    if name == "LR":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed))
    # maliyet-duyarlı (ağırlıklı) taban çizgileri — hakem R3-1
    if name == "RF-W":
        return RandomForestClassifier(n_estimators=100, random_state=seed,
                                      n_jobs=-1, class_weight="balanced")
    if name == "XGB-W":
        return _WeightedXGB(n_estimators=100, verbosity=0, random_state=seed,
                            n_jobs=-1)
    if name == "LR-W":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed,
                               class_weight="balanced"))
    if name == "MLP":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=300,
                             random_state=seed)
    if name == "BRF":
        from imblearn.ensemble import BalancedRandomForestClassifier
        return BalancedRandomForestClassifier(n_estimators=100,
                                              random_state=seed, n_jobs=-1)
    if name == "RF-THR":
        return _ThresholdMoveRF(seed)
    raise ValueError(name)


class _ThresholdMoveRF:
    """Threshold-moving baseline: RF'yi orijinal (dengesiz) veride egitir;
    ic bolme uzerinde her sinif icin OvR olasilik esigini secer ve tahminde
    p_c / thr_c en yuksek olan sinifi atar (cok-sinifli maliyet-esik kurali).
    Augmentasyonsuz calisir → oversampling kazancinin ne kadari yalniz esik
    kaydirmadan geliyor sorusunu yanitlar (dis hakem #6)."""
    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X, y):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score
        self.classes_ = np.unique(y)
        self.clf = RandomForestClassifier(n_estimators=100,
                                          random_state=self.seed, n_jobs=-1)
        # ic bolme ile esik secimi (oof olasiliklar)
        oof = np.zeros((len(X), len(self.classes_)))
        try:
            skf = StratifiedKFold(n_splits=3, shuffle=True,
                                  random_state=self.seed)
            for tr, te in skf.split(X, y):
                c = RandomForestClassifier(n_estimators=100,
                                           random_state=self.seed, n_jobs=-1)
                c.fit(X[tr], y[tr])
                cols = np.searchsorted(self.classes_, c.classes_)
                oof[te[:, None], cols] = c.predict_proba(X[te])
            thr = np.full(len(self.classes_), 0.5)
            for j, cl in enumerate(self.classes_):
                yb = (y == cl).astype(int)
                best_f1, best_t = -1, 0.5
                for t in np.linspace(0.05, 0.9, 18):
                    f1 = f1_score(yb, (oof[:, j] >= t).astype(int),
                                  zero_division=0)
                    if f1 > best_f1:
                        best_f1, best_t = f1, t
                thr[j] = max(best_t, 1e-3)
            self.thr_ = thr
        except Exception:
            self.thr_ = np.full(len(self.classes_), 0.5)
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        P = self.clf.predict_proba(X)
        cols = np.searchsorted(self.classes_, self.clf.classes_)
        Pf = np.zeros((len(X), len(self.classes_)))
        Pf[:, cols] = P
        score = Pf / self.thr_[None, :]
        return self.classes_[np.argmax(score, axis=1)]


CLASSIFIERS = ["RF", "XGB", "LR"]


# ── DEĞERLENDİRME ────────────────────────────────────────────────────────────

def wilcoxon_vs_base(method_scores, base_scores):
    """Fold-bazlı Wilcoxon signed-rank; p ve medyan delta.
    UYARI: CV fold'ları bağımsız değildir; bu p-değeri yalnızca tanımlayıcıdır.
    Yayın istatistiği için run_stats.py (veri-kümesi-bazlı Friedman/Holm)."""
    d = np.asarray(method_scores) - np.asarray(base_scores)
    if np.allclose(d, 0):
        return 1.0, 0.0
    try:
        _, p = wilcoxon(method_scores, base_scores)
    except Exception:
        p = 1.0
    return float(p), float(np.median(d))


# ── RUNNER ───────────────────────────────────────────────────────────────────

def run_full(datasets, methods, classifiers=CLASSIFIERS,
             n_splits=5, n_repeats=3, seed=42, skip_large=None, verbose=True,
             scale=True):
    """datasets: {name: (X, y)}, methods: {name: fn}
    Döndürür: uzun-format DataFrame (dataset, method, clf, metric, mean, std,
              delta, wilcoxon_p, n_fail).
    Hatalar LAST_FAILURES'a kaydedilir. scale=True ise her fold'da
    StandardScaler eğitim verisine fit edilip hem eğitim hem test verisi
    dönüştürülür; augmentasyon ölçeklenmiş uzayda yapılır.
    """
    global LAST_FAILURES
    LAST_FAILURES = []
    rows = []
    for ds_name, (X, y) in datasets.items():
        nrep = 1 if (skip_large and ds_name in skip_large) else n_repeats
        vc = pd.Series(y).value_counts()
        if verbose:
            print(f"\n{'='*70}\n{ds_name}  n={len(X)} d={X.shape[1]} "
                  f"K={len(np.unique(y))} IR={vc.max()/vc.min():.1f}  "
                  f"(repeats={nrep})\n{'='*70}", flush=True)

        scores = {}
        nfail = {m: 0 for m in methods}

        for rep in range(nrep):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=seed + rep)
            for fold, (tr, te) in enumerate(skf.split(X, y)):
                Xtr0, ytr0 = X[tr], y[tr]
                Xte, yte = X[te], y[te]
                if scale:
                    sc = StandardScaler().fit(Xtr0)
                    Xtr0 = sc.transform(Xtr0)
                    Xte = sc.transform(Xte)
                for mname, fn in methods.items():
                    try:
                        Xtr, ytr = fn(Xtr0.copy(), ytr0.copy(),
                                      seed=seed + rep * 10 + fold)
                    except Exception as e:
                        Xtr, ytr = Xtr0, ytr0
                        nfail[mname] += 1
                        LAST_FAILURES.append(dict(
                            dataset=ds_name, method=mname, rep=rep, fold=fold,
                            exc_type=type(e).__name__, exc_msg=str(e)[:200]))
                    for clf_name in classifiers:
                        clf = make_classifier(clf_name, seed)
                        clf.fit(Xtr, ytr)
                        yp = clf.predict(Xte)
                        for m, mfn in METRICS.items():
                            scores.setdefault((mname, clf_name, m), []).append(
                                mfn(yte, yp))

        # Toplama + tanımlayıcı istatistik
        for clf_name in classifiers:
            for mname in methods:
                for m in METRICS:
                    s = np.array(scores[(mname, clf_name, m)])
                    b = np.array(scores[("Base", clf_name, m)])
                    p, dlt = wilcoxon_vs_base(s, b)
                    rows.append(dict(
                        dataset=ds_name, method=mname, clf=clf_name, metric=m,
                        mean=float(s.mean()), std=float(s.std()),
                        delta=dlt, wilcoxon_p=p, n_fail=nfail[mname]))
            if verbose:
                for mname in methods:
                    f1 = np.mean(scores[(mname, clf_name, "f1_macro")])
                    gm = np.mean(scores[(mname, clf_name, "gmean")])
                    mr = np.mean(scores[(mname, clf_name, "min_recall")])
                    print(f"  [{clf_name:3s}] {mname:14s} "
                          f"F1={f1:.4f} G={gm:.4f} minRec={mr:.4f} "
                          f"fail={nfail[mname]}", flush=True)
    return pd.DataFrame(rows)
