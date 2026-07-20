# Results

`paper_*.csv` are the exact result tables behind the paper (long format:
`dataset, method, clf, metric, mean, std`). Re-running the experiment scripts
regenerates equivalent files (`primary.csv`, `external_*.csv`, …).

| File | Paper location | Content |
|---|---|---|
| `paper_primary_rf_xgb_lr.csv` | Tables 2–6, Fig. 4–6 | Main methods, RF/XGB/LR |
| `paper_primary_extra_rivals.csv` | Table 6 | SMOTE-IPF, MWMOTE, VAE, threshold-moving on the primary suite |
| `paper_primary_generate_then_clean_and_weighted.csv` | Tables 2–6 | SMOTE-ENN/Tomek + cost-sensitive baseline |
| `paper_primary_extended_rivals.csv` | Table 6 | ROS, SVM-SMOTE, KMeans-SMOTE, plain KDE, GMM sampling |
| `paper_primary_balanced_rf.csv` | Table 6 | Balanced Random Forest |
| `paper_primary_probability_metrics.csv` | Table 7 | macro-AP, balanced accuracy, Brier |
| `paper_issuer_classifier_matrix.csv` | Table 12 | Surrogate × downstream-classifier matrix |
| `paper_mechanism_issuer_ablation.csv` | Table 13 | Admission-issuer ablation + 3-NN non-model rule |
| `paper_external_moderate.csv` | Table 8 | External validation I (IR 9–20) |
| `paper_external_extreme_real.csv` | Table 9 | External validation II, real (IR 111–400) |
| `paper_external_synthetic.csv` | Table 10 | External validation II, synthetic (IR 100–444) |
| `paper_ablation_geob_trunc.csv` | Table 11, Fig. 6 | Cumulative ablation (GeoB, truncation) |
| `paper_tau_sweep.csv` | Fig. 7 | Admission-strictness (τ) sweep |
| `paper_soft_admission.csv` | Fig. 6 | Soft (probabilistic) admission variants |
| `paper_negative_result_round1.csv` | Section 5.2 | Importance-weighted bandwidth audit, round 1 |
| `paper_negative_result_round2.csv` | Section 5.2 | Importance-weighted bandwidth audit, round 2 |
| `paper_label_noise_diagnostic.csv` | Section 5.3, Fig. 8 | Model-independent diagnostic |
| `paper_fallback_stats.csv` | Section 3 | Fallback rate and admission-violation statistics |
