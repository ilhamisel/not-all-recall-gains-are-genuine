#!/usr/bin/env python3
"""Rev.4 Tasks 1+2: CC-strict (primary method), CC-batch (matched
generate-filter-refill control), GeoF (filter w/o refill), GeoB, CC-SG
re-run on a suite with the published protocol (5x3 strat. CV, seed=42,
fold-internal scaling). Usage: python rev4_run_suite.py {primary|primaryXL|external|real|synth}
Outputs: results/rev4_<suite>.csv (summary, run_full format),
         results/rev4_<suite>_folds.csv (per-fold scores),
         results/rev4_<suite>_quota.csv (per call x class quota/violation stats),
         results/rev4_<suite>_failures.csv"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import benchmark_math_models as B
import rev4_methods as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

METHODS = {"Base": B.augment_base, "Sgeo": R.SGEO, "CC-SG": R.CCSG_I,
           "CC-strict": R.CCSTRICT, "CC-batch": R.CCBATCH, "GeoF": R.geo_filter}


def load(suite):
    if suite == "primary":
        from run_importance_v3 import load_all
        return load_all(), ["RF"]
    if suite == "primaryXL":          # CC-strict under XGB / LR (Table 4)
        from run_importance_v3 import load_all
        return load_all(), ["XGB", "LR"]
    if suite == "external":
        from run_rev_external import load_external
        return load_external(), ["RF"]
    from datasets_rev3 import load_real_extreme, make_synth_extreme
    return (load_real_extreme() if suite == "real" else make_synth_extreme()), ["RF"]


def main(suite):
    t0 = time.time()
    ds, clfs = load(suite)
    meths = METHODS if suite != "primaryXL" else {
        k: METHODS[k] for k in ("Base", "CC-SG", "CC-strict")}
    folds = []
    df = R.run_cv(ds, meths, classifiers=clfs, keep_folds=folds)
    df.to_csv(os.path.join(OUT, f"rev4_{suite}.csv"), index=False)
    pd.DataFrame(folds).to_csv(os.path.join(OUT, f"rev4_{suite}_folds.csv"), index=False)
    pd.DataFrame(R.STATS).to_csv(os.path.join(OUT, f"rev4_{suite}_quota.csv"), index=False)
    pd.DataFrame(R.FAILURES, columns=["dataset", "method", "rep", "fold", "exc_type",
                                      "exc_msg"]).to_csv(
        os.path.join(OUT, f"rev4_{suite}_failures.csv"), index=False)
    print(f"[OK] rev4_{suite}  {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
