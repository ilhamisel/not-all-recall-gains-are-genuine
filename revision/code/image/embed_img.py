#!/usr/bin/env python3
"""Rev.4 — frozen ImageNet ResNet-50 embeddings for every task image, plus
embeddings of pixel-augmented views of every non-majority-class image
(image-native augmentation baseline).

Extractor: torchvision ResNet-50, IMAGENET1K_V2 weights, fc removed ->
2048-d global-average-pooled features. 64x64 MedMNIST images are resized to
224 (bilinear), grayscale replicated to 3 channels, ImageNet normalisation.
The extractor is label-free and frozen, so embedding all images once is not a
leakage source; everything label-dependent (PCA, scaling, oversampling,
classifier) happens inside each CV training fold.

Augmented views: V_c = min(200, ceil(n_max / n_c)) views per image of class c
(enough to balance the class from its own images). Each view = modality-aware
flips (tasks_img.FLIP_POLICY) + random affine (rot +-15 deg, zoom 1.0-1.15,
shift +-4%) + brightness/contrast +-10%.
Output: data/emb_<task>.npz  (E, y, src_idx, E_aug, aug_src)
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks_img import TASKS, DATA, FLIP_POLICY, load_pool, task_indices

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
BS = 256


def model():
    m = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Identity()
    return m.eval().to(DEV)


def to_tensor(imgs):
    t = torch.from_numpy(imgs).to(DEV).float() / 255.0
    if t.ndim == 3:
        t = t.unsqueeze(1).repeat(1, 3, 1, 1)
    else:
        t = t.permute(0, 3, 1, 2)
    return t


def augment(t, policy, gen):
    n = t.shape[0]
    hf, vf = policy
    if hf:
        m = torch.rand(n, generator=gen, device=DEV) < 0.5
        t[m] = t[m].flip(-1)
    if vf:
        m = torch.rand(n, generator=gen, device=DEV) < 0.5
        t[m] = t[m].flip(-2)
        k = torch.randint(0, 4, (n,), generator=gen, device=DEV)
        for kk in range(1, 4):
            mk = k == kk
            if mk.any():
                t[mk] = torch.rot90(t[mk], kk, dims=(-2, -1))
    ang = (torch.rand(n, generator=gen, device=DEV) * 30 - 15) * np.pi / 180
    sc = 1.0 / (1.0 + torch.rand(n, generator=gen, device=DEV) * 0.15)
    tx = torch.rand(n, 2, generator=gen, device=DEV) * 0.08 - 0.04
    A = torch.zeros(n, 2, 3, device=DEV)
    A[:, 0, 0] = sc * torch.cos(ang); A[:, 0, 1] = -sc * torch.sin(ang)
    A[:, 1, 0] = sc * torch.sin(ang); A[:, 1, 1] = sc * torch.cos(ang)
    A[:, :, 2] = tx
    grid = F.affine_grid(A, t.shape, align_corners=False)
    t = F.grid_sample(t, grid, padding_mode="reflection", align_corners=False)
    b = (torch.rand(n, 1, 1, 1, generator=gen, device=DEV) * 0.2 - 0.1)
    c = 1 + (torch.rand(n, 1, 1, 1, generator=gen, device=DEV) * 0.2 - 0.1)
    mu = t.mean(dim=(1, 2, 3), keepdim=True)
    return ((t - mu) * c + mu + b).clamp(0, 1)


@torch.no_grad()
def embed(net, imgs, policy=None, gen=None):
    out = []
    for i in range(0, len(imgs), BS):
        t = to_tensor(imgs[i:i + BS])
        if policy is not None:
            t = augment(t, policy, gen)
        t = F.interpolate(t, size=224, mode="bilinear", align_corners=False)
        t = (t - MEAN) / STD
        with torch.autocast(DEV, dtype=torch.float16):
            f = net(t)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def main(names):
    net = model()
    cache = {}
    for name in names:
        ds = TASKS[name][0]
        out = os.path.join(DATA, f"emb_{name}.npz")
        if os.path.exists(out):
            print("skip", name); continue
        t0 = time.time()
        if ds not in cache:
            cache = {ds: load_pool(ds)}
        Xi, yi, _ = cache[ds]
        idx = task_indices(name, yi)
        imgs, y = Xi[idx], yi[idx]
        E = embed(net, imgs)
        # augmented views of non-majority classes
        cnt = {c: (y == c).sum() for c in np.unique(y)}
        nmax = max(cnt.values())
        gen = torch.Generator(device=DEV); gen.manual_seed(0)
        src, Ea = [], []
        for c, nc in cnt.items():
            if nc >= nmax:
                continue
            V = int(min(200, np.ceil(nmax / nc)))
            ic = np.where(y == c)[0]
            rep = np.repeat(ic, V)
            Ea.append(embed(net, imgs[rep], FLIP_POLICY[ds], gen))
            src.append(rep)
        Ea = np.concatenate(Ea).astype(np.float32)
        src = np.concatenate(src)
        np.savez_compressed(out, E=E.astype(np.float32), y=y, src_idx=idx,
                            E_aug=Ea, aug_src=src)
        print(f"{name:14s} n={len(y)} counts={dict(sorted(cnt.items()))} "
              f"aug={len(src)}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(TASKS))
