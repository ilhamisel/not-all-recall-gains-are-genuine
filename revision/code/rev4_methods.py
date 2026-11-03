#!/usr/bin/env python3
"""Rev.4 (Reviewer 2) — instrumented CC-strict, matched generate-filter-refill
control (CC-batch), post-hoc filter without refill (GeoF), and a CV runner
that is a line-for-line replica of eval_framework.run_full but records which
dataset/rep/fold each augmentation call belongs to (for quota statistics).

Nothing in existing modules is modified; everything is imported.

Boundary proposal law (identical for all variants):
  seed  x_i ~ Uniform(minority class c)
  q     ~ Uniform(5 nearest majority (= not-c) neighbours of x_i)
  t     ~ U(0.25, 0.75);  cand = x_i + t (q - x_i)
Surrogate: OvR XGB-30 fitted on the (scaled) training fold, admission p>=0.5.

Variants
  policy="orig"   : CC-SG behaviour (unfilled quota -> flip-truncated fallback,
                    fallback delivered without re-test)   == augment_xsemi cfgate
  policy="strict" : CC-strict: fallback re-tested, failures replaced by core
                    (GS-0.85) samples                     == augment_ccsg_strict
  gen="inloop"    : rejection rounds draw exactly the missing quota (CC-SG)
  gen="batch"     : CC-batch: each round draws a FULL quota-sized batch
                    (n_bnd proposals), filters it, keeps the first admitted
                    proposals up to the missing quota; same max rounds (6).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import benchmark_math_models as B
import eval_framework as E
from augment_xai_v4 import (_minority_iter, _kde_components, _ovr_model,
                            _flip_points)
from augment_xai_v6 import _core_samples

LO, HI, FRAC, TAU, MAX_ROUNDS = 0.25, 0.75, 0.5, 0.5, 6

CTX = {}          # dataset / rep / fold of the current call (set by run_cv)
STATS = []        # one row per (call, minority class)


def _propose(rng, X_c, X_maj, nbr, kk, n):
    si = rng.randint(len(X_c), size=n)
    mi = nbr[si, rng.randint(kk, size=n)]
    P, Q = X_c[si], X_maj[mi]
    return P, Q


def ccsg_family(X, y, seed=42, policy="strict", gen="inloop", method=""):
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
        st = dict(method=method, cls=int(c), n_bnd=n_bnd, n_inloop=0,
                  n_fb=0, n_fb_pass=0, n_fb_viol_delivered=0, n_core_repl=0,
                  n_core_repl_viol=0, n_proposals=0, rounds=0,
                  n_bnd_delivered=0, n_bnd_delivered_viol=0)
        if n_bnd > 0:
            kk = min(5, len(X_maj))
            nn = NearestNeighbors(n_neighbors=kk).fit(X_maj)
            _, nbr = nn.kneighbors(X_c)
            clf = _ovr_model(X, y, c, seed)
            out, need = [], n_bnd
            for _ in range(MAX_ROUNDS):
                if need <= 0:
                    break
                m = need if gen == "inloop" else n_bnd
                P, Q = _propose(rng, X_c, X_maj, nbr, kk, m)
                t = rng.uniform(LO, HI, size=m)
                cand = P + t[:, None] * (Q - P)
                ok = clf.predict_proba(cand)[:, 1] >= TAU
                st["rounds"] += 1
                st["n_proposals"] += m
                out.extend(cand[ok][:need])
                need = n_bnd - len(out)
            st["n_inloop"] = len(out)
            bnd = list(out)
            repl = []
            if need > 0:
                P, Q = _propose(rng, X_c, X_maj, nbr, kk, need)
                t = rng.uniform(LO, HI, size=need) * _flip_points(clf, P, Q, d)
                fb = P + t[:, None] * (Q - P)
                viol = clf.predict_proba(fb)[:, 1] < TAU
                st["n_fb"] = len(fb)
                st["n_fb_pass"] = int((~viol).sum())
                if policy == "strict":
                    bnd.extend(fb[~viol])
                    repl = _core_samples(rng, gmm, comps, n_bnd - len(bnd))
                else:
                    bnd.extend(fb)
            # final audit of the delivered boundary set (surrogate re-test)
            if len(bnd):
                pv = clf.predict_proba(np.asarray(bnd))[:, 1] < TAU
                st["n_bnd_delivered"] = len(bnd)
                st["n_bnd_delivered_viol"] = int(pv.sum())
                st["n_fb_viol_delivered"] = int(pv[st["n_inloop"]:].sum())
            if len(repl):
                st["n_core_repl"] = len(repl)
                st["n_core_repl_viol"] = int(
                    (clf.predict_proba(np.asarray(repl))[:, 1] < TAU).sum())
            syn.extend(bnd)
            syn.extend(repl)
        st.update(CTX)
        STATS.append(st)
        Xa.append(np.array(syn).reshape(-1, d)); ya.append(np.full(len(syn), c))
    return np.vstack(Xa), np.concatenate(ya)


def geo_filter(X, y, seed=42):
    """GeoF: one GeoB batch, post-hoc surrogate filter, NO refill (quota
    shrinks). Core part identical to CC-SG. Generate-then-filter strawman."""
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    Xa, ya = [X], [y]
    for c, diff in _minority_iter(X, y):
        X_c, X_maj = X[y == c], X[y != c]
        n_bnd = int(diff * FRAC) if len(X_maj) else 0
        gmm = B.fit_gmm(X_c, seed=seed)
        comps = _kde_components(X_c, gmm.predict(X_c), gmm.n_components, d)
        syn = _core_samples(rng, gmm, comps, diff - n_bnd)
        n_keep = 0
        if n_bnd > 0:
            kk = min(5, len(X_maj))
            nn = NearestNeighbors(n_neighbors=kk).fit(X_maj)
            _, nbr = nn.kneighbors(X_c)
            clf = _ovr_model(X, y, c, seed)
            P, Q = _propose(rng, X_c, X_maj, nbr, kk, n_bnd)
            t = rng.uniform(LO, HI, size=n_bnd)
            cand = P + t[:, None] * (Q - P)
            ok = clf.predict_proba(cand)[:, 1] >= TAU
            syn.extend(cand[ok]); n_keep = int(ok.sum())
        st = dict(method="GeoF", cls=int(c), n_bnd=n_bnd, n_inloop=n_keep,
                  n_proposals=n_bnd, rounds=1, n_bnd_delivered=n_keep,
                  n_bnd_delivered_viol=0)
        st.update(CTX); STATS.append(st)
        Xa.append(np.array(syn).reshape(-1, d)); ya.append(np.full(len(syn), c))
    return np.vstack(Xa), np.concatenate(ya)


def CCSG_I(X, y, seed=42):
    return ccsg_family(X, y, seed, policy="orig", gen="inloop", method="CC-SG")


def CCSTRICT(X, y, seed=42):
    return ccsg_family(X, y, seed, policy="strict", gen="inloop",
                       method="CC-strict")


def CCBATCH(X, y, seed=42):
    return ccsg_family(X, y, seed, policy="strict", gen="batch",
                       method="CC-batch")


def SGEO(X, y, seed=42):
    import augment_xai_v4 as X4
    return X4.augment_xsemi(X, y, mode="geo", frac=0.5, depth_lo=0.25,
                            depth_hi=0.75, seed=seed)


def CCSG_REF(X, y, seed=42):
    """Published reference implementation (for reproduction check)."""
    import augment_xai_v4 as X4
    return X4.augment_xsemi(X, y, mode="cfgate", frac=0.5, depth_lo=0.25,
                            depth_hi=0.75, p_thr=0.5, seed=seed)


# ── CV runner: replica of eval_framework.run_full + context recording ──────
FAILURES = []
SCALER = [None]   # fold scaler of the current call (oracle experiments)


def run_cv(datasets, methods, classifiers=("RF",), n_splits=5, n_repeats=3,
           seed=42, verbose=True, scale=True, keep_folds=None):
    """Same folds/seeds/scaling/metrics as eval_framework.run_full.
    keep_folds: optional list to which per-fold score rows are appended."""
    rows = []
    for ds_name, (X, y) in datasets.items():
        vc = pd.Series(y).value_counts()
        if verbose:
            print(f"\n{'='*70}\n{ds_name}  n={len(X)} d={X.shape[1]} "
                  f"K={len(np.unique(y))} IR={vc.max()/vc.min():.1f}\n"
                  f"{'='*70}", flush=True)
        scores, nfail = {}, {m: 0 for m in methods}
        for rep in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=seed + rep)
            for fold, (tr, te) in enumerate(skf.split(X, y)):
                Xtr0, ytr0, Xte, yte = X[tr], y[tr], X[te], y[te]
                if scale:
                    sc = StandardScaler().fit(Xtr0)
                    Xtr0, Xte = sc.transform(Xtr0), sc.transform(Xte)
                    SCALER[0] = sc
                for mname, fn in methods.items():
                    CTX.clear(); CTX.update(dataset=ds_name, rep=rep, fold=fold)
                    try:
                        Xtr, ytr = fn(Xtr0.copy(), ytr0.copy(),
                                      seed=seed + rep * 10 + fold)
                    except Exception as e:
                        Xtr, ytr = Xtr0, ytr0
                        nfail[mname] += 1
                        FAILURES.append(dict(dataset=ds_name, method=mname,
                                             rep=rep, fold=fold,
                                             exc_type=type(e).__name__,
                                             exc_msg=str(e)[:200]))
                    for clf_name in classifiers:
                        clf = E.make_classifier(clf_name, seed)
                        clf.fit(Xtr, ytr)
                        yp = clf.predict(Xte)
                        for m, mfn in E.METRICS.items():
                            v = mfn(yte, yp)
                            scores.setdefault((mname, clf_name, m), []).append(v)
                            if keep_folds is not None:
                                keep_folds.append(dict(
                                    dataset=ds_name, method=mname,
                                    clf=clf_name, rep=rep, fold=fold,
                                    metric=m, value=v))
        for clf_name in classifiers:
            for mname in methods:
                for m in E.METRICS:
                    s = np.array(scores[(mname, clf_name, m)])
                    b = np.array(scores[("Base", clf_name, m)])
                    p, dlt = E.wilcoxon_vs_base(s, b)
                    rows.append(dict(dataset=ds_name, method=mname,
                                     clf=clf_name, metric=m,
                                     mean=float(s.mean()), std=float(s.std()),
                                     delta=dlt, wilcoxon_p=p,
                                     n_fail=nfail[mname]))
            if verbose:
                for mname in methods:
                    f1 = np.mean(scores[(mname, clf_name, "f1_macro")])
                    gm = np.mean(scores[(mname, clf_name, "gmean")])
                    mr = np.mean(scores[(mname, clf_name, "min_recall")])
                    mc = np.mean(scores[(mname, clf_name, "mcc")])
                    print(f"  [{clf_name:3s}] {mname:10s} F1={f1:.4f} "
                          f"G={gm:.4f} minRec={mr:.4f} MCC={mc:.4f} "
                          f"fail={nfail[mname]}", flush=True)
    return pd.DataFrame(rows)
