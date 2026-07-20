#!/usr/bin/env python3
"""
Download the datasets used in the paper.

We do NOT redistribute the data; this script fetches it from the original public
sources and writes it in the layout the code expects (last column = target label).

  * Multi-class tabular sets (Wine_Red, Wine_White)  -> UCI Wine Quality, saved to ./raw/
  * 8 binary KEEL/UCI imbalance benchmarks           -> fetched at runtime by imbalanced-learn
                                                        (ecoli, us_crime, oil, car_eval_4,
                                                         yeast_me2, ozone_level, mammography,
                                                         abalone_19) — no download needed here.

Usage:
    python download_data.py
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "raw")
os.makedirs(OUT, exist_ok=True)

UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality"
WINE = {
    "Wine_Red.csv":   f"{UCI}/winequality-red.csv",
    "Wine_White.csv": f"{UCI}/winequality-white.csv",
}


def main():
    for fname, url in WINE.items():
        dst = os.path.join(OUT, fname)
        print(f"Downloading {url} ...")
        # UCI wine files are ';'-separated with 'quality' as the last column.
        df = pd.read_csv(url, sep=";")
        df.to_csv(dst, index=False)   # comma-separated; last column stays 'quality' (target)
        print(f"  saved {dst}  ({df.shape[0]} rows x {df.shape[1]} cols)")

    print("\nDone. Multi-class Wine sets are in ./raw/.")
    print("The 8 binary KEEL/UCI benchmarks are fetched automatically by imbalanced-learn")
    print("when you run the experiment scripts (internet required on first run).")


if __name__ == "__main__":
    main()
