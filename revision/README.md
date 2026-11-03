# Revision analyses

Code and result files for the analyses added during the revision of the paper
(Reviewer 1 and Reviewer 2). Table, figure, and section numbers refer to the
revised manuscript. The scripts are the research code used for the paper; the
`ccsg/` package at the repository root is the clean reference implementation.

## Running

The scripts import each other as plain modules. Run them from `revision/code/`,
for example:

```bash
cd revision/code
python rev4_run_suite.py      # strict / CC-batch / GeoF on the four suites
python rev4_oracle.py         # known-posterior oracle (21 Gaussian mixtures)
python rev4_oracle_analyze.py
python rev4_core_audit.py     # admission audit of core and boundary samples
python rev4_lineage_all.py    # row-level lineage between suites
python rev4_doc_checks.py     # neighbour rules, failed runs, surrogate facts
python rev5_run_ipf.py        # corrected SMOTE-IPF on all suites
python rev4_stats.py          # Wilcoxon/Holm, bootstrap CIs, batch controls
python rev5_stats.py          # final tables (strict = CC-SG, corrected IPF)
```

Outputs are written to a `results/` folder next to the scripts; the archived
outputs used in the paper are in `revision/results/`. `rev4_stats.py` and
`rev5_stats.py` also read the first-submission result files
(`results_xai_v5.csv`, `results_rev3_*.csv`, `results_master_v2.csv`, ...),
which are included in `revision/results/` as well. Some comments in the research
code are in Turkish.

The medical image experiment is in `code/image/` (MedMNIST v2, frozen
ImageNet ResNet-50 embeddings, fold-internal PCA-32/64):

```bash
cd revision/code/image
python embed_img.py           # downloads MedMNIST, extracts embeddings (GPU optional)
python run_img.py p32 32 <task> [<task> ...]
python analyze_img.py p32
```

## Map to the revised manuscript

| Analysis | Script | Result files | Manuscript |
|---|---|---|---|
| Strict implementation, quota accounting | `rev4_methods.py`, `rev4_run_suite.py` | `rev4_{primary,primaryXL,external,real,synth}*.csv` | Sec. 3 (Phase 4), Tables 2–4, 9–11 |
| Matched screening controls (GeoB, CC-batch, GeoF) | `rev4_methods.py`, `rev4_stats.py` | `rev4_task2_batch.csv` | Sec. 5.3, Table 18 |
| Core / boundary admission audit | `rev4_core_audit.py` | `rev4_core_audit.csv` | Sec. 5.3 |
| Known-posterior oracle | `rev4_oracle.py`, `rev4_oracle_analyze.py` | `rev4_oracle_*.csv`, `rev4_oracle_report.txt` | Sec. 5.4, Table 19 |
| Corrected SMOTE-IPF | `rev5_ipf.py`, `rev5_run_ipf.py` | `rev5_ipf_*.csv` | Sec. 4.1, Tables 9–11, 13 |
| Data lineage | `rev4_lineage_all.py` | `rev4_lineage_all.csv` | Secs. 4.6–4.7 |
| Neighbour rules, failed runs, surrogate | `rev4_doc_checks.py` | `rev4_documentation.md`, `rev4_failures_summary.csv` | Sec. 4.1 |
| Wilcoxon/Holm, bootstrap CIs, Hodges–Lehmann | `rev5_stats.py` | `rev5_holm.csv`, `rev5_bootstrap.csv`, `rev5_means.csv` | Tables 6–7, 9–13 |
| Holm family of the ten main rivals, Friedman ranks | computed from `rev5_stats.assemble()` | `rev6_r2_holm_core10.csv`, `rev6_r2_numbers.txt` | Sec. 4.4, Tables 5–6 |
| Medical image experiment | `image/*.py` | `image/summary_p32.md`, `image/summary_p64.md` | Sec. 4.8, Table 12 |
