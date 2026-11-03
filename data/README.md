# Datasets

No dataset is redistributed in this repository. Everything is fetched from its
original public source by `imbalanced-learn` or by `python data/download.py`.

## Primary suite (10 datasets, imbalance ratio 9–440)

| Dataset | Type | Source |
|---|---|---|
| ecoli, us_crime, oil, car_eval_4, yeast_me2, ozone_level, mammography, abalone_19 | binary | KEEL / UCI, fetched via [`imbalanced-learn`](https://imbalanced-learn.org/stable/references/generated/imblearn.datasets.fetch_datasets.html) (Zenodo mirror) |
| Wine_Red, Wine_White | multi-class (ordinal wine quality) | [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) |

The two wine datasets are the UCI *Wine Quality* red/white tables with the
`quality` score (integer 3–9) used directly as an ordinal multi-class label,
which yields extreme imbalance (Wine_White has a five-sample rarest class,
IR ≈ 440). Fetch them with:

```bash
python data/download.py     # writes data/raw/Wine_Red.csv and Wine_White.csv
```

The eight binary benchmarks are downloaded automatically the first time you run
an experiment (internet required on first run; `imbalanced-learn` caches them).

## External validation I — moderate imbalance (10 datasets, IR 9–20)

`satimage, sick_euthyroid, spectrometer, car_eval_34, libras_move,
thyroid_sick, solar_flare_m0, yeast_ml8, coil_2000, arrhythmia` — all from the
same `imbalanced-learn` collection. `car_eval_34` is not independent of the primary suite (see
*Data lineage* below); the other nine share no rows with it.

## External validation II — extreme imbalance (9 real + 9 synthetic)

* **Real (IR 111–400):** public benchmarks
  (`webpage, letter_img, pen_digits, optical_digits, scene, isolet,
  wine_quality, abalone`) sub-sampled at their minority class to a target
  imbalance ratio, plus the natively extreme `protein_homo` (IR 111). The exact
  construction is in `ccsg/datasets.py::load_external_extreme` (fixed seed 42).
* **Synthetic (IR 100–444, noise-free labels):** generated with
  `sklearn.datasets.make_classification` (IR ∈ {100, 200, 400} × three
  separability levels, `flip_y=0`). See
  `ccsg/datasets.py::make_synthetic_extreme`.

All construction code is deterministic (seed 42) and fully contained in
`ccsg/datasets.py`, so every suite can be regenerated exactly.

**Data lineage.** The external suites are not fully independent of the primary suite (see `revision/results/lineage_observation_ids.csv` for the matched row indices):

* `car_eval_34` has the same 1,728 feature rows as `car_eval_4` with a different label definition; its 134 minority rows include all 65 minority rows of `car_eval_4`.
* `wine_quality` (real extreme suite) is a sub-sample whose 4,746 rows all occur in `Wine_White`, with a different minority definition (quality ≤ 4).
* `abalone` (real extreme suite) is a sub-sample whose 3,804 rows all occur in `abalone_19`, with a different label.
* Two `wine_quality` rows also match the feature values of three `Wine_Red` rows; the same feature vectors occur in `Wine_White`, so these matches are most likely coincidental and cannot be confirmed as the same observations (the source data carry no observation IDs).
* All other external tasks share no rows with the primary suite.
