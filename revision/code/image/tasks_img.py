#!/usr/bin/env python3
"""Rev.4 (Reviewer 1) — image-domain task definitions on MedMNIST v2 (64x64).

All official train/val/test splits of a dataset are POOLED, then each task is
built by stratified subsampling with a fixed seed (0). Evaluation is the
paper's protocol: 5x3 repeated stratified CV on the task pool (official test
splits would hold only 1-3 minority images at IR 100-200 -> unusable).

Scenario coverage
  native-moderate   : BreastMNIST, PneumoniaMNIST (IR ~2.7)
  native-extreme    : DermaMNIST 7-class (IR ~58)
  induced binary    : Derma mel-vs-nv, Blood ig-vs-neutrophil (IR 200),
                      Pneumonia, Breast (IR 50-100)
  induced multi-min : Blood 8-class (IR 50 & 100), Derma 7-class (IR 100 & 200),
                      OrganA 6-class (kidney-left IR 100; laterality-sensitive)

Flip policy (pixel augmentation baseline) is modality-aware: laterality-coded
labels (OrganA left/right organs) and chest X-ray anatomy forbid flips.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

FLIP_POLICY = {          # (hflip, vflip+rot90)
    "dermamnist":     (True, True),    # dermoscopy: orientation-free
    "bloodmnist":     (True, True),    # smear microscopy: orientation-free
    "breastmnist":    (True, False),   # ultrasound: depth axis is fixed
    "pneumoniamnist": (False, False),  # chest X-ray: cardiac laterality
    "organamnist":    (False, False),  # labels encode left/right organs
}

# task -> (dataset, {class: count or "all"/fraction}, description)
# counts are TOTAL images in the task pool (CV then puts 80% in training)
TASKS = {
    "Breast-native":   ("breastmnist", {0: "all", 1: "all"},
                        "malignant(0) vs normal/benign(1), native"),
    "Pneu-native":     ("pneumoniamnist", {0: 800, 1: 2200},
                        "normal vs pneumonia, native ratio (~2.7)"),
    "Derma-native":    ("dermamnist", "strat4000",
                        "7 classes, stratified 4000 of 10015, native IR"),
    "Derma-mel":       ("dermamnist", {5: 2000, 4: 20},
                        "nevus(5) 2000 vs melanoma(4) 20, IR 100"),
    "Derma-7ext":      ("dermamnist", {5: 2000, 4: 400, 2: 400, 1: 200,
                                       0: 130, 6: 20, 3: 10},
                        "7 classes, vasc IR 100, df IR 200"),
    "Blood-8cls":      ("bloodmnist", {1: 1000, 2: 1000, 3: 1000, 5: 1000,
                                       6: 1000, 7: 1000, 0: 20, 4: 10},
                        "8 classes, basophil IR 50, lymphocyte IR 100"),
    "Blood-ig":        ("bloodmnist", {6: 2000, 3: 10},
                        "neutrophil 2000 vs immature granulocyte 10, IR 200"),
    "Pneu-ext":        ("pneumoniamnist", {0: 1500, 1: 15},
                        "normal 1500 vs pneumonia 15, IR 100"),
    "Breast-ext":      ("breastmnist", {1: 550, 0: 11},
                        "normal/benign 550 vs malignant 11, IR 50"),
    "OrganA-6ext":     ("organamnist", {0: 800, 1: 800, 2: 800, 3: 800,
                                        5: 800, 4: 8},
                        "bladder/femurL/femurR/heart/kidneyR 800, "
                        "kidney-left 8, IR 100"),
}


def load_pool(ds):
    z = np.load(os.path.join(DATA, f"{ds}_64.npz"), allow_pickle=False)
    X = np.concatenate([z[f"{s}_images"] for s in ("train", "val", "test")])
    y = np.concatenate([z[f"{s}_labels"][:, 0] for s in ("train", "val",
                                                         "test")])
    split = np.concatenate([np.full(len(z[f"{s}_labels"]), i)
                            for i, s in enumerate(("train", "val", "test"))])
    return X, y.astype(int), split


def task_indices(name, y):
    ds, spec, _ = TASKS[name]
    rng = np.random.RandomState(0)
    if spec == "strat4000":
        idx = []
        for c in np.unique(y):
            ic = np.where(y == c)[0]
            k = int(round(len(ic) * 4000 / len(y)))
            idx.extend(rng.choice(ic, k, replace=False))
        return np.sort(np.array(idx))
    idx = []
    for c, k in spec.items():
        ic = np.where(y == c)[0]
        idx.extend(ic if k == "all" else rng.choice(ic, k, replace=False))
    return np.sort(np.array(idx))
