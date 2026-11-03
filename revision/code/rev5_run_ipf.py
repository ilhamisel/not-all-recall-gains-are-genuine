#!/usr/bin/env python3
"""Rev.5 Task 1: re-run the corrected SMOTE-IPF (rev5_ipf.smote_ipf) with the
identical protocol as rev4 / rev3 (5x3 stratified CV, StratifiedKFold
random_state = 42 + rep, augmenter seed = 42 + 10*rep + fold, fold-internal
StandardScaler, RF n_estimators=100 random_state=42) via rev4_methods.run_cv
(line-for-line replica of eval_framework.run_full).
SMOTE is re-run alongside as a reproduction check / paired reference.
Usage: python rev5_run_ipf.py {primary|real|synth|external} [consensus]
(consensus = sensitivity run with the consensus voting scheme; outputs rev5_ipfcons_*)
Outputs: results/rev5_ipf_<suite>.csv (run_full format),
         results/rev5_ipf_<suite>_folds.csv, results/rev5_ipf_<suite>_removal.csv,
         results/rev5_ipf_<suite>_failures.csv"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from imblearn.over_sampling import SMOTE
import benchmark_math_models as B
import rev4_methods as R
import rev5_ipf as I
from run_literature import wrap
from rev4_run_suite import load

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def ipf(X, y, seed=42):
    I.CTX.clear(); I.CTX.update(R.CTX)
    return I.smote_ipf(X, y, seed=seed)


def main(suite, cons=False):
    t0 = time.time()
    I.CONSENSUS[0] = cons
    tag = 'ipfcons' if cons else 'ipf'
    ds, _ = load(suite)
    meths = {"Base": B.augment_base, "SMOTE": wrap(SMOTE), "SMOTE-IPF": ipf}
    folds = []
    df = R.run_cv(ds, meths, classifiers=["RF"], keep_folds=folds)
    df.to_csv(os.path.join(OUT, f"rev5_{tag}_{suite}.csv"), index=False)
    pd.DataFrame(folds).to_csv(os.path.join(OUT, f"rev5_{tag}_{suite}_folds.csv"), index=False)
    pd.DataFrame(I.LOG).to_csv(os.path.join(OUT, f"rev5_{tag}_{suite}_removal.csv"), index=False)
    pd.DataFrame(R.FAILURES, columns=["dataset", "method", "rep", "fold", "exc_type",
                                      "exc_msg"]).to_csv(
        os.path.join(OUT, f"rev5_{tag}_{suite}_failures.csv"), index=False)
    print(f"[OK] rev5_{tag}_{suite}  {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == 'consensus')
