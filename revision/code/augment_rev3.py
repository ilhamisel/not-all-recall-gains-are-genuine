#!/usr/bin/env python3
"""Rev.3 ek rakipler (dis hakem): SMOTE-IPF, MWMOTE-tarzi, derin uretici (VAE).
Hepsi ortak (X, y, seed) imzasi. Cok-sinifli: her azinlik sinifi hedefe
tamamlanir."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE


def _kmin(y):
    return int(pd.Series(y).value_counts().min())


def _safe_smote(y, seed):
    return SMOTE(k_neighbors=max(1, min(5, _kmin(y) - 1)), random_state=seed)


# ── SMOTE-IPF (Saez ve ark. 2015): SMOTE + yinelemeli bolumleme filtresi ─────

def smote_ipf(X, y, seed=42, n_part=6, k_iter=3, p=0.01):
    """SMOTE ile dengele, sonra k tur boyunca: veriyi n_part parcaya ayir,
    her parcada bir C4.5-benzeri (RF) egit, TUM veriyi oyla; cogunlukla
    yanlis siniflanan orneklerini at. Degisim orani p'nin altina inince dur.
    Yalnizca SENTETIK ornekler silinebilir (gercek veriye dokunma)."""
    rng = np.random.RandomState(seed)
    Xr, yr = _safe_smote(y, seed).fit_resample(X, y)
    Xr, yr = np.asarray(Xr), np.asarray(yr)
    n_orig = len(X)
    synth_mask = np.arange(len(Xr)) >= n_orig
    keep = np.ones(len(Xr), bool)
    for _ in range(k_iter):
        idx = np.where(keep)[0]
        if len(idx) < n_part * 2:
            break
        votes = np.zeros(len(Xr))
        counts = np.zeros(len(Xr))
        skf = StratifiedKFold(n_splits=n_part, shuffle=True,
                              random_state=rng.randint(1 << 30))
        Xk, yk = Xr[idx], yr[idx]
        try:
            splits = list(skf.split(Xk, yk))
        except ValueError:
            break
        for tr, _te in splits:
            clf = RandomForestClassifier(n_estimators=25, random_state=seed,
                                         n_jobs=-1)
            clf.fit(Xk[tr], yk[tr])
            pred = clf.predict(Xr)
            votes += (pred == yr).astype(float)
            counts += 1.0
        agree = np.divide(votes, np.maximum(counts, 1))
        # cogunlukla yanlis (agree<0.5) VE sentetik olan ornekleri at
        drop = synth_mask & keep & (agree < 0.5)
        changed = drop.sum() / max(len(Xr), 1)
        keep &= ~drop
        if changed < p:
            break
    return Xr[keep], yr[keep]


# ── MWMOTE-tarzi (Barua ve ark. 2014): bilgilendirici azinligi agirlikla ────

def mwmote(X, y, seed=42, k1=5, k2=3, k3=None):
    """MWMOTE cekirdegi: (i) guvenli olmayan azinligi filtrele (en az bir
    azinlik komsusu olan), (ii) sinir-yakinlik + azinlik-yogunluk agirligi ata,
    (iii) agirlikli tohumdan azinlik-komsu yonunde uret. Basitlestirilmis ama
    sadik: kume-ici kisit yerine k-NN azinlik komsusu kullanilir."""
    rng = np.random.RandomState(seed)
    counts = pd.Series(y).value_counts()
    target = counts.max()
    Xa, ya = [X], [y]
    for c in counts.index:
        diff = int(target - counts[c])
        if diff <= 0:
            continue
        Xc = X[y == c]
        Xoth = X[y != c]
        if len(Xc) < 2 or len(Xoth) < 1:
            # yedek: duz cogaltma
            si = rng.randint(len(Xc), size=diff)
            Xa.append(Xc[si]); ya.append(np.full(diff, c)); continue
        # (i) bilgilendirici azinlik: en yakin komsulari arasinda cogunluk olanlar
        kk = min(k1, len(X) - 1)
        nn_all = NearestNeighbors(n_neighbors=kk + 1).fit(X)
        _, nb = nn_all.kneighbors(Xc)
        yflat = np.asarray(y)
        border = np.array([np.any(yflat[row[1:]] != c) for row in nb])
        Xinfo = Xc[border] if border.sum() >= 2 else Xc
        # (ii) agirlik: en yakin cogunluga yakinlik (sinir) ile orantili
        nn_oth = NearestNeighbors(n_neighbors=1).fit(Xoth)
        dmaj, _ = nn_oth.kneighbors(Xinfo)
        w = 1.0 / (dmaj[:, 0] + 1e-6)
        w = w / w.sum()
        # (iii) uretim: agirlikli tohum + azinlik-komsu yonunde interpolasyon
        kc = min(k2, max(1, len(Xc) - 1))
        nn_min = NearestNeighbors(n_neighbors=kc + 1).fit(Xc)
        _, nbmin = nn_min.kneighbors(Xinfo)
        syn = []
        for _ in range(diff):
            i = rng.choice(len(Xinfo), p=w)
            j = nbmin[i, rng.randint(1, nbmin.shape[1])]
            g = rng.uniform(0, 1)
            syn.append(Xinfo[i] + g * (Xc[j] - Xinfo[i]))
        Xa.append(np.array(syn)); ya.append(np.full(diff, c))
    return np.vstack(Xa), np.concatenate(ya)


# ── Derin uretici baseline: sinif-basi kucuk tablo VAE ──────────────────────

def vae_oversample(X, y, seed=42, epochs=200, latent=8, hidden=32):
    """Her azinlik sinifi icin kucuk bir VAE egit ve hedefe kadar orneklem.
    Asiri kitlikta (n_c<8) egitim kararsiz olabilir → duz KDE-benzeri yedek.
    Amac: modern derin-uretici ailesinin bu rejimdeki davranisini gostermek."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    counts = pd.Series(y).value_counts()
    target = counts.max()
    d = X.shape[1]
    Xa, ya = [X], [y]

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
            self.mu = nn.Linear(hidden, latent)
            self.lv = nn.Linear(hidden, latent)
            self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                     nn.Linear(hidden, d))

        def forward(self, x):
            h = self.enc(x); mu = self.mu(h); lv = self.lv(h)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            return self.dec(z), mu, lv

    for c in counts.index:
        diff = int(target - counts[c])
        if diff <= 0:
            continue
        Xc = X[y == c].astype(np.float32)
        if len(Xc) < 8:                       # asiri kitlik: KDE-benzeri yedek
            std = Xc.std(0) + 1e-6
            si = rng.randint(len(Xc), size=diff)
            syn = Xc[si] + rng.normal(0, std * 0.5, size=(diff, d))
            Xa.append(syn.astype(float)); ya.append(np.full(diff, c)); continue
        Xt = torch.tensor(Xc)
        vae = VAE()
        opt = torch.optim.Adam(vae.parameters(), lr=1e-2)
        for _ in range(epochs):
            opt.zero_grad()
            xh, mu, lv = vae(Xt)
            rec = ((xh - Xt) ** 2).mean()
            kld = -0.5 * torch.mean(1 + lv - mu ** 2 - torch.exp(lv))
            (rec + 0.01 * kld).backward(); opt.step()
        with torch.no_grad():
            z = torch.randn(diff, latent)
            syn = vae.dec(z).numpy()
        Xa.append(syn.astype(float)); ya.append(np.full(diff, c))
    return np.vstack(Xa), np.concatenate(ya)
