# Not All Recall Gains Are Genuine

**In-loop admission control for boundary oversampling under extreme class imbalance**

Reference implementation and full reproduction package for the paper

> İlhami Sel. *Not All Recall Gains Are Genuine: In-Loop Admission Control for
> Boundary Oversampling under Extreme Class Imbalance.* (under review)

<!-- Add the paper DOI/link here once available. -->

---

## What is this?

Boundary-oriented oversampling places synthetic minority samples along the
k-nearest-neighbor graph, but the model that will ultimately classify the data
plays no role in where those samples land. This package shows that a large part
of the recall such methods report is a **label-noise artifact** — synthetic
points planted inside the true majority region — and provides **CC-SG**
(Consistency-Controlled Surrogate-Gated Gaussian oversampling), which removes
that artifact by moving admission control **inside** the generation loop:

* half of each class's synthetic budget comes from a calibrated component-local
  Gaussian KDE **core**;
* the other half are **boundary samples that must pass an in-loop admission
  test** — a one-vs-rest surrogate must still assign the candidate to the
  minority class, otherwise it is **rejected and resampled** rather than deleted
  after the fact.

On 10 primary and 28 external datasets (imbalance ratios up to 444), CC-SG keeps
most of the recall of boundary oversampling while attaining the best
precision-sensitive scores (macro-F1, MCC), and the advantage is largest exactly
where imbalance is extreme.

## Installation

```bash
git clone https://github.com/<your-username>/not-all-recall-gains-are-genuine.git
cd not-all-recall-gains-are-genuine
pip install -r requirements.txt
```

Python ≥ 3.10. PyTorch is only needed for the optional deep-generative baseline.

## Quick start

```python
from ccsg import CCSG

# X: (n_samples, n_features) float array; y: (n_samples,) integer labels
X_res, y_res = CCSG(random_state=42).fit_resample(X, y)
```

`CCSG` follows the `imbalanced-learn` `fit_resample` convention, so it drops
into any scikit-learn pipeline that expects a resampler. Useful options:

```python
CCSG(
    beta=0.5,          # fraction of the budget for the certified boundary sampler
    tau=0.5,           # admission threshold (surrogate prob. >= tau to admit)
    depth=(0.25, 0.75),# boundary depth interval
    shrink=0.85,       # KDE core bandwidth shrinkage factor
    strict=False,      # True -> every admitted sample passes the surrogate
    random_state=42,
)
```

## Adapting CC-SG to your own dataset

CC-SG needs only a numeric feature matrix `X` and integer labels `y`; it detects
the minority classes automatically and balances every one of them.

```python
import numpy as np
from ccsg import CCSG

# 1. Encode categorical features and labels to numbers, impute missing values,
#    and (recommended) standardize features on the TRAINING split only.
# 2. Oversample the training set:
X_train_res, y_train_res = CCSG(random_state=0).fit_resample(X_train, y_train)
# 3. Train any classifier on the resampled data and evaluate on the untouched
#    test set.
```

Guidelines:

* **Standardize inside the fold.** CC-SG samples in feature space; fit the
  scaler on the training split and apply it to the test split (never the other
  way round). The experiment scripts do this for you.
* **Continuous features only.** The Gaussian core is not meant for purely
  categorical columns; use a categorical-aware kernel or SMOTE-NC for those.
* **Very few minority samples.** CC-SG is designed for the extreme regime
  (as few as five minority samples); no special handling is required.
* **Want a fully certified set?** Pass `strict=True` so that every admitted
  sample — including the fallback — passes the surrogate. It performs within
  ±0.006 of the default on every metric in our experiments.
* **Tuning.** The defaults (`beta=0.5, tau=0.5, shrink=0.85`) were fixed once on
  the primary suite and transferred to all external suites unchanged; you should
  rarely need to touch them. `tau` trades recall for precision.

## Reproducing the paper

```bash
python data/download.py                        # fetch the two UCI wine datasets
python experiments/reproduce_primary.py        # Tables 2-6, Fig. 4-6
python experiments/reproduce_external.py       # Tables 7-10 (three unseen suites)
python experiments/label_noise_diagnostic.py   # Section 5.3, Fig. 8
```

Each script writes a long-format CSV to `results/`. The exact result tables
behind the paper are archived in `results/paper_*.csv` (see `results/README.md`
for the mapping), and the eight paper figures are in `figures/`.

## Repository layout

```
ccsg/            the method and its evaluation framework
  method.py        CC-SG (CCSG.fit_resample) + strict variant
  core.py          GMM fit, component-local KDE core, surrogate, helpers
  baselines.py     CalKDE, GeoB (matched control), SMOTE-IPF, MWMOTE, VAE
  evaluation.py    metrics, classifiers, repeated stratified CV
  datasets.py      the three evaluation suites (deterministic, seed 42)
data/            dataset source links + download script (data/README.md)
experiments/     reproduction entry points
results/         paper result tables (CSV) + regenerated outputs
figures/         the eight figures used in the paper
```

## Datasets

No data is redistributed; everything is fetched from the original public
sources. See [`data/README.md`](data/README.md) for links and the exact
construction of each suite.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`):

```bibtex
@article{sel_ccsg,
  title   = {Not All Recall Gains Are Genuine: In-Loop Admission Control for
             Boundary Oversampling under Extreme Class Imbalance},
  author  = {Sel, {\.I}lhami},
  year    = {2026},
  note    = {under review}
}
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
