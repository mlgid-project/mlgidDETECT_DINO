"""
Dataset / sampler / split for SimMIM pretraining on backbone_ssl_corpus.h5.

- reads stored UNPROCESSED moneta uint8 polar frames, applies the detector-matching
  transform (/255) + physics-respecting augmentation (vflip/gamma/noise)
- generates a SimMIM mask grid per frame, and a valid (non-no-data) pixel mask
- WHOLE-SCAN train/val split (deterministic by scan_id hash) — never random frames
- scan-balanced sampling so giant scans don't dominate a gradient step
"""
import os, sys, hashlib
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import backbone_transform as BT

H5_DEFAULT = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/backbone_ssl_corpus.h5"


class MaskGenerator:
    """SimMIM random mask grid at mask_patch resolution."""
    def __init__(self, img_h=512, img_w=1024, mask_patch=32, ratio=0.6):
        self.gh, self.gw = img_h // mask_patch, img_w // mask_patch
        self.count = self.gh * self.gw
        self.nmask = int(round(self.count * ratio))

    def __call__(self, rng):
        m = np.zeros(self.count, bool)
        m[rng.choice(self.count, self.nmask, replace=False)] = True
        return m.reshape(self.gh, self.gw)


def scan_val_split(scan_ids, val_frac=0.12, seed=0):
    """Deterministic whole-scan split: a scan goes to val if hash falls in the val band."""
    uniq = sorted(set(scan_ids))
    val = set()
    for s in uniq:
        h = int(hashlib.md5(f"{seed}:{s}".encode()).hexdigest(), 16) % 1000
        if h < int(val_frac * 1000):
            val.add(s)
    is_val = np.array([s in val for s in scan_ids])
    return is_val, val


class SimMIMDataset(Dataset):
    def __init__(self, h5_path=H5_DEFAULT, indices=None, mask_patch=32, mask_ratio=0.6,
                 augment=True, img_h=512, img_w=1024):
        self.h5_path = h5_path
        self.augment = augment
        self.maskgen = MaskGenerator(img_h, img_w, mask_patch, mask_ratio)
        self._h5 = None
        import h5py
        with h5py.File(h5_path, "r") as h:
            self.n_total = h["images"].shape[0]
        self.indices = np.arange(self.n_total) if indices is None else np.asarray(indices)

    def _h(self):
        if self._h5 is None:                                # open per-worker
            import h5py
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        u8 = self._h()["images"][idx]                       # (512,1024) uint8, 0=no-data
        rng = np.random.default_rng()
        x = BT.to_model_input(u8)                           # float32 [0,1], 0=no-data
        if self.augment == "v2":                            # round-2 aug (ssl/RECIPE_v2.md)
            x = BT.augment_v2(x, rng)
        elif self.augment:                                  # True or "v1" -> round-1 aug
            x = BT.augment(x, rng)
        valid = (x > 0).astype(np.float32)
        mask = self.maskgen(rng)
        return (torch.from_numpy(x)[None],                  # (1,H,W)
                torch.from_numpy(mask),                     # (gh,gw) bool
                torch.from_numpy(valid)[None])              # (1,H,W)


def build_loaders(h5_path=H5_DEFAULT, batch_size=8, num_workers=8, val_frac=0.12,
                  mask_patch=32, mask_ratio=0.6, seed=0, aug="v1"):
    import h5py
    with h5py.File(h5_path, "r") as h:
        scans = h["scan_id"][:].astype(str)
    is_val, val_scans = scan_val_split(scans, val_frac, seed)
    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]

    # scan-balanced weights for the training sampler (per-frame weight = 1/scan_size)
    from collections import Counter
    cnt = Counter(scans[train_idx].tolist())
    weights = np.array([1.0 / cnt[scans[i]] for i in train_idx])
    sampler = WeightedRandomSampler(torch.from_numpy(weights).double(),
                                    num_samples=len(train_idx), replacement=True)

    tr = SimMIMDataset(h5_path, train_idx, mask_patch, mask_ratio, augment=aug)
    va = SimMIMDataset(h5_path, val_idx, mask_patch, mask_ratio, augment=False)
    train_loader = torch.utils.data.DataLoader(
        tr, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=num_workers > 0)
    val_loader = torch.utils.data.DataLoader(
        va, batch_size=batch_size, shuffle=False, num_workers=max(2, num_workers // 2),
        pin_memory=True, drop_last=False, persistent_workers=num_workers > 0)
    print(f"split: {len(train_idx)} train frames / {len(val_idx)} val frames; "
          f"val holds out {len(val_scans)} whole scans")
    return train_loader, val_loader


if __name__ == "__main__":
    tl, vl = build_loaders(batch_size=2, num_workers=0)
    x, m, v = next(iter(tl))
    print(f"batch: x {tuple(x.shape)} {x.dtype} [{x.min():.2f},{x.max():.2f}]; "
          f"mask {tuple(m.shape)} {m.dtype} frac={m.float().mean():.2f}; valid {tuple(v.shape)}")
