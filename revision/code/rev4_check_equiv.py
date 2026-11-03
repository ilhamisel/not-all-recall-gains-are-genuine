"""Rev.4: bit-equality check of rev4_methods vs published implementations."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from run_importance_v3 import load_all
import rev4_methods as R
import augment_rev as AR
ds = load_all()
for n in ["ecoli", "abalone_19", "Wine_White", "yeast_me2"]:
    X, y = ds[n]; X = StandardScaler().fit_transform(X)
    for s in [42, 53]:
        a = R.CCSG_I(X, y, seed=s); b = R.CCSG_REF(X, y, seed=s)
        c = R.CCSTRICT(X, y, seed=s); d = AR.augment_ccsg_strict(X, y, strict=True, seed=s)
        print(n, s, "CC-SG equal:", np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]),
              "| CC-strict equal:", np.array_equal(c[0], d[0]) and np.array_equal(c[1], d[1]))
