# Rev.4 documentation facts (Reviewer 2) — derived from code and existing results

Scripts: `rev4_doc_checks.py` (output `rev4_doc_checks.txt`, `rev4_failures_summary.csv`,
`rev4_lineage.csv`), `rev4_lineage_all.py` (output `rev4_lineage_all.csv`). No model was re-run for this file.

## 1. Wine_White rarest class: neighbour counts actually used

Wine_White class counts (label-encoded quality 3..9): {0: 20, 1: 163, 2: 1457, 3: 2198, 4: 880, 5: 175, 6: 5}.
With stratified 5-fold CV the rarest class (quality 9, n=5) has **4 training points in every one of the
15 training folds** and 1 test point per fold.

| Method (code) | Neighbour rule in our code | k on Wine_White |
|---|---|---|
| SMOTE (`run_literature.wrap`) | `k_neighbors = max(1, min(5, n_min_train - 1))`, where `n_min_train` is the size of the **rarest class in the training fold**, and that single k is applied to **all** minority classes | **k = 3** (all classes) |
| Borderline-SMOTE (`wrap(BorderlineSMOTE)`) | same k rule for `k_neighbors`; `m_neighbors` = imblearn default 10 (danger test over all classes) | k = 3, m = 10 |
| ADASYN (`run_literature._adasyn`) | `n_neighbors = max(1, min(5, n_min_train - 1))` | 3 |
| SMOTE-ENN / SMOTE-Tomek (`run_rev_baselines._safe_smote`) | inner SMOTE with the same k rule; ENN uses imblearn default `n_neighbors=3` | SMOTE k = 3 |
| SVM-SMOTE, KMeans-SMOTE (`results_master_v2`) | same `wrap` k rule | k = 3 |
| SMOTE-IPF (`augment_rev3.smote_ipf`) | inner SMOTE with the same k rule | 3 |
| MWMOTE (`augment_rev3.mwmote`) | informative-minority filter k1 = min(5, n−1) over all data; generation neighbours k2 = min(3, n_c − 1) **per class** | k1 = 5, k2 = 3 |
| CalKDE / GeoB / CC-SG / CC-strict | GMM components K = min(3, max(1, n_c // 2)) per class; covariance `full` only if n_c > d + 2, else `diag` (fallback `spherical`); boundary partner drawn from the min(5, n_maj) nearest **majority** neighbours of the seed | K = 2 diag components on 4 points; 5 majority neighbours |

So yes: the rule is k = min(5, n−1) with n = rarest training-class size (not k = min(5, n_c − 1)
per class). On every other primary dataset the rarest training class has ≥ 8 points, so k = 5 everywhere
except Wine_White (k = 3). (Per-dataset ranges in `rev4_doc_checks.txt`.)

## 2. Failed runs (augmenter raised an exception → that fold silently used Base data, counted in `n_fail`)

| File | Dataset | Method | Failed folds / 15 | Exception |
|---|---|---|---|---|
| results_xai_v5.csv (main tables) | Wine_Red | **ADASYN** | **15/15** | ValueError "No samples will be generated with the provided ratio settings" |
| results_master_v2.csv (extended table) | Wine_Red | ADASYN | 15/15 | same |
| results_master_v2.csv | Wine_Red | KMeans-SMOTE | 14/15 | RuntimeError "No clusters found with sufficient samples…" |
| results_master_v2.csv | Wine_White | KMeans-SMOTE | 15/15 | same |
| results_master_v2.csv | Wine_Red | SVM-SMOTE | 1/15 | ValueError "All support vectors are considered as noise…" |
| results_master_v2.csv | Wine_White | SVM-SMOTE | 6/15 | same |

Consequence for reporting: the ADASYN row on Wine_Red in the main tables is **identical to Base** (verified:
RF minRec 0.0528 = Base, MCC 0.5067 = Base, same for XGB/LR). This should be stated in a table footnote
(e.g. "ADASYN failed on Wine_Red in all folds; Base values substituted"). No failures occurred for
CC-SG, CC-strict, CC-batch, GeoB, CalKDE, SMOTE, Borderline, SMOTE-ENN/Tomek, SMOTE-IPF, MWMOTE, VAE,
ThreshMove, Weighted, or in any rev4 run (see `rev4_*_failures.csv`, empty unless noted in the final report).

## 3. Surrogate: inner split / calibration

* CC-SG / CC-strict / CC-batch surrogate = `augment_xai_v4._ovr_model`: `XGBClassifier(n_estimators=30,
  verbosity=0, random_state=seed)` with all other XGBoost defaults (max_depth 6, learning_rate 0.3),
  fitted as a binary one-vs-rest model (class c vs all others) on the **entire scaled training fold**
  (all original training points, no synthetic points).
* **There is no inner split, no hold-out, no cross-fitting and no probability calibration.** The admission
  rule is the raw `predict_proba(cand)[:, 1] >= 0.5`. The test fold is never seen (augmentation is
  fold-internal). Because the surrogate is evaluated only on *new* candidate points, in-sample overfit
  affects admission indirectly (sharp, over-confident boundaries); for Wine_White's rarest class the
  surrogate is trained with 4 positives vs ~3,900 negatives.
* A calibrated surrogate exists only as an ablation (`augment_rev._issuer('xgbcal')`:
  `CalibratedClassifierCV(XGB-30, method='sigmoid', cv=min(3, max(2, n_c)))`, internal CV inside the
  training fold) — results in `results_rev2_matrix.csv` (Table "issuer × final classifier").
* The flip-point fallback (`_flip_points`) uses the same surrogate on a 20-point grid t ∈ [0.05, 1].
* The only method with an inner split is the ThreshMove baseline (`eval_framework._ThresholdMoveRF`):
  3-fold OOF probabilities inside the training fold, per-class F1-optimal threshold on an 18-point grid.

## 4. Data lineage — external / extreme tasks vs primary tasks (exact feature-row matching)

Rows were matched on all features rounded to 6 decimals; every external (`run_rev_external`) and
real-extreme (`datasets_rev3.load_real_extreme`) dataset was compared with every primary dataset of equal
dimensionality (`rev4_lineage_all.csv`).

| Other task | Primary task | % of other task's rows found in primary | Label relationship |
|---|---|---|---|
| **wine_quality_x** (real-extreme suite) | **Wine_White** | **100 %** (4,746/4,746; imblearn `wine_quality` = the full 4,898-row white-wine file) | minority = quality ≤ 4; its 31 minority rows are primary classes 0 (quality 3, 4 rows) and 1 (quality 4, 27 rows) → *related* label concept (primary rare classes are quality 3 and 9) |
| **abalone_x** (real-extreme suite) | **abalone_19** | **100 %** (3,804/3,804; imblearn `abalone` and `abalone_19` share the identical 4,177-row matrix) | minority = imblearn `abalone` positive class (391 rows; imblearn definition: rings = 7 vs rest), not rings = 19; all 18 retained minority rows are primary class 0 → different label, same instances/features |
| **car_eval_34** (external suite, Table 9) | **car_eval_4** (primary) | **100 %** (1,728/1,728) | minority of car_eval_34 (134 rows) = all 65 car_eval_4 minority rows + 69 others → **nested / strongly related** |
| wine_quality_x | Wine_Red | 0.04 % (2 coincidental rows) | unrelated |
| all other external / extreme sets | — | 0 % | independent |

**Conclusion:** the external and extreme suites are *not* fully independent of the primary suite.
Three of the 19 real "unseen" tasks are re-labelled or sub-sampled views of primary parent data
(wine_quality_x ⊂ Wine_White rows, abalone_x ⊂ abalone_19 rows, car_eval_34 ≡ car_eval_4 rows).
The docstrings ("hicbir yerde kullanilmamis" / "orijinal suitle kesisimsiz") and any manuscript claim
of disjointness should be corrected, and a sensitivity analysis excluding these three tasks reported
(see the final rev4 report for that number).

## 5. Additional implementation fact found during Rev.4 (affects a rival)

**SMOTE-IPF is numerically (almost) identical to SMOTE.** In `results_rev3_real.csv` and
`results_rev3_synth.csv` every SMOTE-IPF cell equals SMOTE exactly (max |diff| = 0.0); on the primary suite
max |diff| = 0.003. Cause (`augment_rev3.smote_ipf`): each partition's RandomForest (25 trees, unpruned)
votes on *all* rows including its own training partition (5/6 of the data), so memorised synthetic points
are almost never voted "majority-misclassified" and the filter removes ~nothing. Classic IPF uses pruned
C4.5 trees, which do not memorise. Either re-implement (e.g. predict only on the held-out partition, or
use a pruned/shallow tree) or report SMOTE-IPF as "≈SMOTE in our implementation".

## 6. Scope of the "zero admission violation" guarantee (from `rev4_core_audit.csv`)

CC-strict's delivered *boundary* block has 0 % surrogate violations by construction, but the same
surrogate assigns p(c|x) < 0.5 to 58–78 % of the GS-0.85 *core* samples (half the budget) and to
57–80 % of the core replacements. Whole CC-strict synthetic set: 37–57 % surrogate-violating vs 63–83 %
for GeoB (suite means). Any manuscript wording must restrict the guarantee to the boundary block.
