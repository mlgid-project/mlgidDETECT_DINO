"""Synthetic-input appearance matching (docs/STYLE_TRANSFER_INVESTIGATION.md), Phase-0 MVP.

TRAINING-ONLY: a monotone per-image CDF/histogram match of SYNTHETIC training images onto
a fixed reference intensity distribution pooled offline from REAL preprocessed frames (the
SSL corpus `backbone_ssl_corpus.h5`, 12,991 frames -- disjoint from the eval sets, and put
into the model's [0,1] input space by backbone_transform.to_model_input, same space the
synthetic images live in). It closes the pixel-distribution SHAPE gap the diagnostics show
(synthetic spiky/bright-skewed vs real smooth) -- which is different from the reverted Path A
(mask geometry + quantization level count, MODIFICATIONS.md phase H).

Two guarantees, both by construction:
  * Rank-preserving: the remap is monotone in intensity, so a labeled peak that was a local
    maximum stays one -> faint peaks cannot be erased or reordered.
  * No-data preserved: 0 (no-data) pixels are excluded from the match and stay exactly 0.

It is applied ONLY inside SimulationDataset.__getitem__ (main.py); the real-image
preprocessing (standard_preprocessing), the eval path, and the exported ONNX graph never
touch it -> deployment is byte-identical.
"""
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO, "backbone_curation") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "backbone_curation"))


def build_reference(h5_path, out_path, n_frames=2000, K=1024, per_frame_cap=4000, seed=0):
    """Pool nonzero pixels from n_frames real corpus frames (model-input space) and save
    the reference quantile function ref_q[k] = quantile at percentile k/(K-1). Run once."""
    import h5py
    import backbone_transform as BT
    rng = np.random.default_rng(seed)
    pooled = []
    with h5py.File(h5_path, "r") as h:
        N = h["images"].shape[0]
        idx = np.sort(rng.choice(N, size=min(n_frames, N), replace=False))
        for i in idx:
            x = BT.to_model_input(h["images"][i])            # float32 [0,1], 0=no-data
            v = x[x > 0].ravel().astype(np.float32)
            if v.size > per_frame_cap:
                v = rng.choice(v, size=per_frame_cap, replace=False)
            pooled.append(v)
    pooled = np.concatenate(pooled)
    perc = np.linspace(0.0, 1.0, K, dtype=np.float64)
    ref_q = np.quantile(pooled, perc).astype(np.float32)
    torch.save({'ref_q': torch.from_numpy(ref_q), 'K': int(K),
                'n_frames': int(len(idx)), 'n_pixels': int(pooled.size)}, out_path)
    return ref_q


def load_reference(path, device='cpu'):
    d = torch.load(path, map_location=device)
    return d['ref_q'].to(device)


def cdf_match(img, ref_q):
    """Monotone rank-based histogram match of one image's nonzero pixels onto ref_q.
    img: float tensor in [0,1] (0 = no-data), any shape. ref_q: (K,) ascending reference
    quantiles. Returns same shape/device; zeros preserved, intensity ordering preserved."""
    K = ref_q.shape[0]
    flat = img.reshape(-1)
    nz = flat > 0
    n = int(nz.sum())
    if n == 0:
        return img
    vals = flat[nz]
    order = torch.argsort(vals)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(n, device=vals.device)
    perc = (ranks.to(ref_q.dtype) + 0.5) / n                 # (n,) in (0,1)
    pos = perc * (K - 1)                                     # index into ref quantile grid
    lo = pos.floor().long().clamp(0, K - 1)
    hi = (lo + 1).clamp(0, K - 1)
    w = pos - lo.to(ref_q.dtype)
    matched = ref_q[lo] * (1 - w) + ref_q[hi] * w
    out = flat.clone()
    out[nz] = matched.to(flat.dtype)
    return out.reshape_as(img)


if __name__ == "__main__":
    # Build the reference once from the SSL corpus.
    H5 = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/backbone_ssl_corpus.h5"
    OUT = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/style_ref.pt"
    rq = build_reference(H5, OUT)
    print(f"saved reference -> {OUT}  (K={len(rq)}, min={rq.min():.4f}, "
          f"med={np.median(rq):.4f}, max={rq.max():.4f}, ascending={bool(np.all(np.diff(rq) >= 0))})")
