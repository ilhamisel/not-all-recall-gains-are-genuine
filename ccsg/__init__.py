"""CC-SG: consistency-controlled, surrogate-gated Gaussian oversampling.

Reference implementation for the paper
    "Not All Recall Gains Are Genuine: In-Loop Admission Control for Boundary
     Oversampling under Extreme Class Imbalance."

Quick start
-----------
>>> from ccsg import CCSG
>>> X_res, y_res = CCSG(random_state=42).fit_resample(X, y)
"""
from .method import CCSG, oversample
from .baselines import calkde, geob, smote_ipf, mwmote, vae_oversample

__all__ = ["CCSG", "oversample", "calkde", "geob", "smote_ipf", "mwmote",
           "vae_oversample"]
__version__ = "1.0.0"
