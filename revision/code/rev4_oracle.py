#!/usr/bin/env python3
"""Rev.4 Task 3 (R2 1.1) — known-posterior oracle.
Gaussian-mixture problems with analytically computable P(c|x).
  binary : d in {5,10}, IR in {100,200,400}, overlap in {low,med,high}
  multi  : d=5, 1 majority + 2 minority classes (IR 100 and 300),
           overlap in {low,med,high}  (OvR, posterior from the full mixture)
Two reference posteriors:
  Bayes   : true class priors (= generating proportions) -> P(c|x) >= 0.5 is
            the Bayes-optimal (0-1 loss) region of class c
  balanced: equal priors (likelihood ratio f_c / f_rest >= 1) -> the region a
            prior-corrected / balanced classifier would assign to c
(A) Proposal audit, every training fold of the 5x3 CV (fold-scaled space):
    GeoB proposals (one quota-sized batch, GeoB law) -> surrogate p, true
    posteriors (evaluated at inverse-scaled points); CC-strict delivered
    boundary set audited the same way.
(B) Downstream RF: Base, GeoB, CC-SG, CC-strict, CC-batch, Oracle gates.
Outputs: results/rev4_oracle_proposals.csv (per fold x class aggregates),
         results/rev4_oracle_summary.csv, results/rev4_oracle_downstream.csv,
         results/rev4_oracle_perprop_sample.csv (10% random sample of proposals)"""
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import benchmark_math_models as B
import rev4_methods as R
from augment_xai_v4 import _minority_iter, _kde_components, _ovr_model
from augment_xai_v6 import _core_samples

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
SEP = {"low": 3.0, "med": 2.0, "high": 1.25}   # minority-mean distance (overlap)
N_MAJ = 4000


class GMProblem:
    """Each class k = equal-weight mixture of Gaussians (means M[k], cov s2 I)."""
    def __init__(self, name, means, s2, counts, seed):
        self.name, self.means, self.s2, self.counts = name, means, s2, counts
        self.K = len(counts)
        self.logpi = np.log(np.asarray(counts, float) / sum(counts))
        rng = np.random.RandomState(seed)
        Xs, ys = [], []
        for k in range(self.K):
            M = means[k]
            comp = rng.randint(len(M), size=counts[k])
            Xs.append(M[comp] + rng.normal(0, np.sqrt(s2[k]), (counts[k], M.shape[1])))
            ys.append(np.full(counts[k], k))
        self.X = np.vstack(Xs); self.y = np.concatenate(ys)

    def loglik(self, X):
        L = np.zeros((len(X), self.K))
        for k in range(self.K):
            d = self.means[k].shape[1]
            X = np.atleast_2d(X)
            lc = np.stack([np.atleast_1d(mvn(m, self.s2[k] * np.eye(d)).logpdf(X))
                           for m in self.means[k]], 1)
            L[:, k] = logsumexp(lc, 1) - np.log(len(self.means[k]))
        return L

    def posterior(self, X, c, balanced=False):
        L = self.loglik(X)
        if not balanced:
            L = L + self.logpi[None]
        return np.exp(L[:, c] - logsumexp(L, 1))


def make_problems(seed=42):
    out = []
    for d in (5, 10):
        for ir in (100, 200, 400):
            for ov, sep in SEP.items():
                rng = np.random.RandomState(seed + d * 1000 + ir + int(sep * 10))
                maj = np.zeros((2, d)); maj[0, 0], maj[1, 0] = -1.5, 1.5
                u = rng.normal(size=(2, d)); u /= np.linalg.norm(u, axis=1)[:, None]
                mn = maj[[0, 1]] + sep * u          # minority comps near maj comps
                n_min = max(10, N_MAJ // ir)
                out.append(GMProblem(f"bin_d{d}_ir{ir}_{ov}", [maj, mn],
                                     [1.0, 0.5], [N_MAJ, n_min], seed + ir + d))
    d = 5
    for ov, sep in SEP.items():
        rng = np.random.RandomState(seed + 77 + int(sep * 10))
        maj = np.zeros((2, d)); maj[0, 0], maj[1, 0] = -1.5, 1.5
        u = rng.normal(size=(2, d)); u /= np.linalg.norm(u, axis=1)[:, None]
        m1 = maj[[0]] + sep * u[[0]]; m2 = maj[[1]] + sep * u[[1]]
        out.append(GMProblem(f"mc3_d5_ir100+300_{ov}", [maj, m1, m2],
                             [1.0, 0.5, 0.5], [N_MAJ, 40, 13], seed + 5))
    return out


# ── (A) proposal audit ───────────────────────────────────────────────────────

def audit_fold(prob, Xs, y, sc, seed, rng_sample):
    """Xs: scaled training fold. Replicates GeoB boundary law exactly
    (core draws first, then proposals) to obtain GeoB's delivered points."""
    rows, samp = [], []
    rng = np.random.RandomState(seed)
    d = Xs.shape[1]
    for c, diff in _minority_iter(Xs, y):
        X_c, X_maj = Xs[y == c], Xs[y != c]
        n_bnd = int(diff * R.FRAC); n_core = diff - n_bnd
        gmm = B.fit_gmm(X_c, seed=seed)
        comps = _kde_components(X_c, gmm.predict(X_c), gmm.n_components, d)
        _core_samples(rng, gmm, comps, n_core)
        kk = min(5, len(X_maj))
        _, nbr = NearestNeighbors(n_neighbors=kk).fit(X_maj).kneighbors(X_c)
        si = rng.randint(len(X_c), size=n_bnd)
        mi = nbr[si, rng.randint(kk, size=n_bnd)]
        P, Q = X_c[si], X_maj[mi]
        cand = P + rng.uniform(R.LO, R.HI, size=n_bnd)[:, None] * (Q - P)
        clf = _ovr_model(Xs, y, c, seed)
        ps = clf.predict_proba(cand)[:, 1]
        Xo = sc.inverse_transform(cand)
        pb = prob.posterior(Xo, c); pbal = prob.posterior(Xo, c, balanced=True)
        adm = ps >= 0.5
        r = dict(cls=int(c), n_prop=n_bnd, admit_rate=adm.mean())
        for ref, pt in (("bayes", pb), ("bal", pbal)):
            inmaj = pt < 0.5                       # "true majority region"
            r[f"{ref}_frac_true_maj"] = inmaj.mean()
            r[f"{ref}_TP_rej"] = int((~adm & inmaj).sum())   # correct rejection
            r[f"{ref}_FP_rej"] = int((~adm & ~inmaj).sum())  # wrongly rejected
            r[f"{ref}_FN_rej"] = int((adm & inmaj).sum())    # wrongly admitted
            r[f"{ref}_TN_rej"] = int((adm & ~inmaj).sum())
            r[f"{ref}_post_adm_mean"] = pt[adm].mean() if adm.any() else np.nan
            r[f"{ref}_post_rej_mean"] = pt[~adm].mean() if (~adm).any() else np.nan
            r[f"{ref}_post_adm_med"] = np.median(pt[adm]) if adm.any() else np.nan
            r[f"{ref}_post_rej_med"] = np.median(pt[~adm]) if (~adm).any() else np.nan
        # also the real data: what fraction of TRAINING minority points are
        # in their own true-majority region (irreducible overlap)
        Xc_o = sc.inverse_transform(X_c)
        r["bayes_frac_minority_seeds_in_maj"] = (prob.posterior(Xc_o, c) < .5).mean()
        r["bal_frac_minority_seeds_in_maj"] = (prob.posterior(Xc_o, c, True) < .5).mean()
        rows.append(r)
        k = rng_sample.rand(n_bnd) < 0.10
        samp.append(pd.DataFrame(dict(cls=int(c), p_sur=ps[k], post_bayes=pb[k],
                                      post_bal=pbal[k])))
    return rows, samp


def audit_delivered(prob, Xs, y, sc, seed, method):
    """Run CC-strict / CC-SG and audit their delivered boundary sets under the
    true posterior (boundary points = last n_bnd synthetic points per class
    minus core replacements; we recompute by re-running with the instrumented
    generator and capturing the boundary block)."""
    R.STATS.clear(); R.CTX.clear()
    fn = {"CC-strict": R.CCSTRICT, "CC-SG": R.CCSG_I, "CC-batch": R.CCBATCH}[method]
    Xa, ya = fn(Xs.copy(), y.copy(), seed=seed)
    out = []
    syn_X, syn_y = Xa[len(Xs):], ya[len(Xs):]
    for st in R.STATS:
        c = st["cls"]; Xc_syn = syn_X[syn_y == c]
        n_core = len(Xc_syn) - st["n_bnd"]
        bnd = Xc_syn[n_core:n_core + st["n_bnd_delivered"]]
        repl = Xc_syn[n_core + st["n_bnd_delivered"]:]
        r = dict(method=method, cls=c, n_bnd=st["n_bnd"],
                 n_delivered=len(bnd), n_repl=len(repl))
        for ref, bal in (("bayes", False), ("bal", True)):
            r[f"{ref}_viol_delivered"] = ((prob.posterior(sc.inverse_transform(bnd), c, bal) < .5).mean()
                                          if len(bnd) else np.nan)
            r[f"{ref}_viol_repl"] = ((prob.posterior(sc.inverse_transform(repl), c, bal) < .5).mean()
                                     if len(repl) else np.nan)
        out.append(r)
    return out


# ── (B) oracle-gated generator (gate = true posterior; strict replacement) ──

def oracle_gate(balanced):
    def fn(X, y, seed=42):
        prob, sc = ORACLE["prob"], R.SCALER[0]
        rng = np.random.RandomState(seed); d = X.shape[1]
        Xa, ya = [X], [y]
        for c, diff in _minority_iter(X, y):
            X_c, X_maj = X[y == c], X[y != c]
            n_bnd = int(diff * R.FRAC); n_core = diff - n_bnd
            gmm = B.fit_gmm(X_c, seed=seed)
            comps = _kde_components(X_c, gmm.predict(X_c), gmm.n_components, d)
            syn = _core_samples(rng, gmm, comps, n_core)
            kk = min(5, len(X_maj))
            _, nbr = NearestNeighbors(n_neighbors=kk).fit(X_maj).kneighbors(X_c)
            out, need = [], n_bnd
            for _ in range(R.MAX_ROUNDS):
                if need <= 0:
                    break
                si = rng.randint(len(X_c), size=need)
                mi = nbr[si, rng.randint(kk, size=need)]
                P, Q = X_c[si], X_maj[mi]
                cand = P + rng.uniform(R.LO, R.HI, size=need)[:, None] * (Q - P)
                ok = prob.posterior(sc.inverse_transform(cand), c, balanced) >= .5
                out.extend(cand[ok]); need = n_bnd - len(out)
            out.extend(_core_samples(rng, gmm, comps, n_bnd - len(out)))
            syn.extend(out)
            Xa.append(np.array(syn).reshape(-1, d)); ya.append(np.full(len(syn), c))
        return np.vstack(Xa), np.concatenate(ya)
    return fn


ORACLE = {}


def main():
    t0 = time.time()
    probs = make_problems()
    prop_rows, deliv_rows, samples, down = [], [], [], []
    for prob in probs:
        X, y = prob.X, prob.y
        vc = pd.Series(y).value_counts()
        print(f"{prob.name}: n={len(X)} IR={vc.max()/vc.min():.0f}", flush=True)
        rs = np.random.RandomState(0)
        for rep in range(3):
            skf = StratifiedKFold(5, shuffle=True, random_state=42 + rep)
            for fold, (tr, te) in enumerate(skf.split(X, y)):
                sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X[tr])
                seed = 42 + rep * 10 + fold
                rows, samp = audit_fold(prob, Xs, y[tr], sc, seed, rs)
                for r in rows:
                    r.update(problem=prob.name, rep=rep, fold=fold)
                prop_rows += rows
                for s_ in samp:
                    s_["problem"] = prob.name
                samples += samp
                for m in ("CC-SG", "CC-strict", "CC-batch"):
                    for r in audit_delivered(prob, Xs, y[tr], sc, seed, m):
                        r.update(problem=prob.name, rep=rep, fold=fold)
                        deliv_rows.append(r)
        # downstream
        ORACLE["prob"] = prob
        R.STATS.clear()
        meths = {"Base": B.augment_base, "Sgeo": R.SGEO, "CC-SG": R.CCSG_I,
                 "CC-strict": R.CCSTRICT, "CC-batch": R.CCBATCH,
                 "Oracle-bal": oracle_gate(True), "Oracle-Bayes": oracle_gate(False)}
        df = R.run_cv({prob.name: (X, y)}, meths, classifiers=["RF"])
        down.append(df)
    pp = pd.DataFrame(prop_rows); dv = pd.DataFrame(deliv_rows)
    pp.to_csv(os.path.join(OUT, "rev4_oracle_proposals.csv"), index=False)
    dv.to_csv(os.path.join(OUT, "rev4_oracle_delivered.csv"), index=False)
    pd.concat(samples).to_csv(os.path.join(OUT, "rev4_oracle_perprop_sample.csv"),
                              index=False)
    pd.concat(down).to_csv(os.path.join(OUT, "rev4_oracle_downstream.csv"), index=False)
    pd.DataFrame(R.FAILURES, columns=["dataset", "method", "rep", "fold", "exc_type",
                 "exc_msg"]).to_csv(os.path.join(OUT, "rev4_oracle_failures.csv"), index=False)
    print(f"[OK] oracle {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
