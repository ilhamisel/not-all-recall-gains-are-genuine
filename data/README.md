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
same `imbalanced-learn` collection, disjoint from the primary suite.

## External validation II — extreme imbalance (9 real + 9 synthetic)

* **Real (IR 111–400):** unused public benchmarks
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
