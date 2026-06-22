"""
Detector-matching preprocessing for the SSL backbone dataloader.

The corpus (`backbone_ssl_corpus.h5`) stores UNPROCESSED moneta 8-bit polar frames
(512x1024 uint8, 0 = no-data) so preprocessing stays changeable. This module turns a
stored frame into exactly what the mlgidDETECT_DINO detector feeds its backbone, so the
SSL-pretrained weights transfer cleanly.

What the detector backbone actually sees (verified in util/exp_preprocess.py):
  contrast_correction = clip -> log10 -> normalize -> histogram-equalize -> /255  ==> float [0,1] polar, 0=no-data
  (no ImageNet mean/std on the input; simulation training images are likewise [0,1]).

The moneta 8-bit polar is ALREADY a log/hist-equalized polar, so the matching transform
is simply /255 (0 stays no-data). We deliberately do NOT re-run contrast_correction:
it expects LINEAR reciprocal input, and our frames are already contrast-baked, so
re-applying it would double-process. The histogram-equalization details differ slightly
between moneta and mlgidDETECT, but the representation/range/structure match — which is
what backbone transfer needs.

Orientation: the chi axis convention varies slightly between source families (raw_* vs
existing_*). For MAE-style SSL, enable a random vertical flip in augmentation; this both
regularizes and makes the detector's orientation one of the seen orientations.
"""
import numpy as np


def to_model_input(u8_polar):
    """Stored uint8 polar (H,W), 0=no-data  ->  float32 [0,1], 0=no-data.
    Matches the detector backbone input range/representation."""
    x = u8_polar.astype(np.float32) / 255.0
    x[u8_polar == 0] = 0.0                      # keep no-data exactly zero
    return x


def valid_mask(u8_polar):
    """Boolean mask of real (non-no-data) pixels."""
    return u8_polar != 0


# ---- physics-respecting SSL augmentations (use in the MAE/DINO dataloader) ----
def augment(x, rng):
    """x: float32 [0,1] polar from to_model_input. Returns augmented copy.
    Only transforms that respect polar (q-radial x chi) geometry — NO arbitrary rotation."""
    m = x > 0
    # random vertical (chi) flip — handles cross-family orientation, free regularizer
    if rng.random() < 0.5:
        x = x[::-1].copy(); m = m[::-1].copy()
    # intensity gamma on valid pixels
    g = rng.uniform(0.8, 1.25)
    x = np.where(m, np.clip(x, 1e-6, 1.0) ** g, 0.0).astype(np.float32)
    # mild additive noise on valid pixels
    if rng.random() < 0.5:
        x = np.where(m, np.clip(x + rng.normal(0, 0.02, x.shape), 0, 1), 0.0).astype(np.float32)
    return x


def augment_v2(x, rng):
    """Round-2 (simmim2) aug — see ssl/RECIPE_v2.md. Richer but still polar-legal.
    v1 `augment` above is kept byte-identical so simmim1 stays reproducible."""
    m = x > 0
    # random vertical (chi) flip
    if rng.random() < 0.5:
        x = x[::-1].copy(); m = m[::-1].copy()
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
    return x


if __name__ == "__main__":
    # smoke test against the built corpus
    import h5py
    p = "/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/backbone_ssl_corpus.h5"
    with h5py.File(p, "r") as h:
        img = h["images"][0]
        x = to_model_input(img)
        print(f"frame0: u8 range [{img.min()},{img.max()}] valid={float((img!=0).mean()):.3f}"
              f"  -> model input range [{x.min():.3f},{x.max():.3f}] dtype {x.dtype}")
