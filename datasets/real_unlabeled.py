"""
Real unlabeled GIWAXS frames for semi-supervised (Semi-DETR-style) training.

Serves weak/strong augmented view pairs of the curated real polar corpus
(backbone_ssl_corpus.h5, 12991 frames, 512x1024 uint8, 0 = no-data) for
mean-teacher pseudo-labeling: the EMA teacher predicts on the WEAK view, the
student trains on the STRONG view against those pseudo-boxes.
Design: docs/SEMI_DETR_INTEGRATION.md (S4a).

This dataset is deliberately CPU/numpy (h5 read + numpy augment, moved to cuda
inside the training step) so it can use num_workers > 0 — unlike
SimulationDataset, which builds cuda tensors and forces num_workers=0.

Geometry note: weak and strong views MUST stay pixel-aligned, otherwise the
teacher's pseudo-boxes are wrong in the student frame. Any geometric op (only
the physics-legal vertical chi-flip) is therefore applied to BOTH views before
the photometric strong augmentation. backbone_transform.augment(_v2) are NOT
used directly because they roll their own internal chi-flip.
"""
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "backbone_curation"))
import backbone_transform as BT  # noqa: E402  (to_model_input; /255, 0 stays no-data)

H5_DEFAULT = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/backbone_ssl_corpus.h5"


def photometric_strong(x, rng, version="v2"):
    """Photometric-only strong augmentation for the student view.

    Mirrors backbone_transform.augment / augment_v2 MINUS their internal chi-flip
    (geometry is handled upstream, shared between views). No-data stays exactly 0.
    x: float32 [0,1] polar from to_model_input. Returns an augmented copy.
    """
    m = x > 0
    if version == "v2":
        # wider intensity gamma on valid pixels
        g = rng.uniform(0.7, 1.40)
        x = np.where(m, np.clip(x, 1e-6, 1.0) ** g, 0.0).astype(np.float32)
        # global exposure scale (detector dose / acquisition-time variation)
        if rng.random() < 0.7:
            x = np.where(m, np.clip(x * rng.uniform(0.8, 1.2), 0, 1), 0.0).astype(np.float32)
        # smooth q-direction intensity ramp (sample absorption / footprint along q)
        if rng.random() < 0.5:
            ramp = np.linspace(rng.uniform(0.85, 1.0), rng.uniform(0.85, 1.0), x.shape[1])[None, :]
            x = np.where(m, np.clip(x * ramp, 0, 1), 0.0).astype(np.float32)
        # additive noise on valid pixels
        if rng.random() < 0.7:
            x = np.where(m, np.clip(x + rng.normal(0, 0.03, x.shape), 0, 1), 0.0).astype(np.float32)
    else:  # "v1"
        g = rng.uniform(0.8, 1.25)
        x = np.where(m, np.clip(x, 1e-6, 1.0) ** g, 0.0).astype(np.float32)
        if rng.random() < 0.5:
            x = np.where(m, np.clip(x + rng.normal(0, 0.02, x.shape), 0, 1), 0.0).astype(np.float32)
    return x


class RealUnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path=H5_DEFAULT, strong="v2", geom_flip=False):
        self.h5_path = h5_path
        self.strong = strong
        self.geom_flip = geom_flip
        self._h5 = None
        import h5py
        with h5py.File(h5_path, "r") as h:
            self.n = h["images"].shape[0]

    def _h(self):
        if self._h5 is None:  # open per-worker
            import h5py
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        u8 = self._h()["images"][i]                     # (512,1024) uint8, 0=no-data
        rng = np.random.default_rng()
        x = BT.to_model_input(u8)                       # float32 [0,1], 0=no-data
        # shared chi-flip: applied to BOTH views, so pseudo-boxes from the weak view
        # are valid in the strong view without any coordinate transform
        if self.geom_flip and rng.random() < 0.5:
            x = x[::-1].copy()
        weak = x                                        # teacher view: identity
        strong = photometric_strong(x.copy(), rng, self.strong)
        return (torch.from_numpy(weak)[None],           # (1,H,W) each
                torch.from_numpy(strong)[None])


def collate_unlabeled(batch):
    weak = torch.stack([b[0] for b in batch])           # (B,1,H,W)
    strong = torch.stack([b[1] for b in batch])
    return weak, strong


if __name__ == "__main__":
    import h5py
    ds = RealUnlabeledDataset()
    w, s = ds[0]
    with h5py.File(ds.h5_path, "r") as h:
        nodata = torch.from_numpy(h["images"][0] == 0)
    # invariant: no-data pixels are exactly 0 in BOTH views (valid pixels may clip to 0
    # under additive noise — that is fine, same as the proven augment_v2 behavior)
    ok = bool((w[0][nodata] == 0).all() and (s[0][nodata] == 0).all())
    print(f"corpus: {len(ds)} frames; weak {tuple(w.shape)} [{w.min():.3f},{w.max():.3f}] "
          f"strong {tuple(s.shape)} [{s.min():.3f},{s.max():.3f}]; no-data stays zero: {ok}")
