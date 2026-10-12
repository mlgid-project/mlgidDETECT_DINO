"""Are the simulated backgrounds diverse over a run, or the same few puzzled together? (colorbox1 / any machine with the realsim data; no GPU, ~2 GB RAM)

  python heatmap/bkg_diversity.py [--canvases 24] [--per_canvas 25] [--seed 0]
Builds only the background part of the real-background simulator (donor mosaic, PCA envelope, detector masks) with the settings of realsim/factory.py R6
(canvas 2048 x 4096, rebuilt every 25 frames, 50 % of the frames on a random-phase surrogate texture) and draws canvases x per_canvas backgrounds, exactly as
RealBkgSimulation._frame_background_v2 does. Then:
  1. TEXTURE REUSE: the fluctuation field (frame / its 16-px blur - 1, invalid pixels 0) of frames is cross-correlated with every other frame over ALL shifts
     (FFT). Independent noise gives a maximum of about 0.01 or less; a texture patch that appears in two frames gives a peak of about (shared area / frame area).
     Reported separately for frames from DIFFERENT canvases, frames from the SAME canvas, and surrogate-texture frames.
  2. ENVELOPE: pairwise correlation of the smooth large-scale shape (log of the 64-px blur, 16 x 32 samples) over all frames, and how many pairs are near-duplicates.
  3. MASKS: how often each detector geometry is drawn.
"""
import os, sys, types, argparse, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import cv2

H, W = 512, 1024


def fluct(b, m):
    """Fluctuation field: frame / its 16-px blur - 1, with the blur normalised over VALID pixels only and everything within 24 px of an invalid pixel set to 0.
    (A first version blurred the zeros of the masked area into the blur, which makes every detector-gap edge a strong feature shared by all frames with the
    same detector geometry; the random-phase surrogate frames, whose texture cannot be shared, showed the same 'correlation' - so that version measured masks.)"""
    mf = m.astype(np.float32)
    bl = cv2.GaussianBlur(b.astype(np.float32) * mf, (0, 0), 16.0) / np.maximum(cv2.GaussianBlur(mf, (0, 0), 16.0), 1e-6)
    r = np.where(m & (bl > 0), b / np.maximum(bl, 1e-6) - 1.0, 0.0)
    core = cv2.erode(m.astype(np.uint8), np.ones((49, 49), np.uint8)).astype(bool)
    return np.where(core, r, 0.0).astype(np.float32)


def envelope(b, m):
    mf = m.astype(np.float32)
    e = cv2.GaussianBlur(b.astype(np.float32) * mf, (0, 0), 64.0) / np.maximum(cv2.GaussianBlur(mf, (0, 0), 64.0), 1e-6)
    e = np.log(np.maximum(e, 1e-3))
    return cv2.resize(e, (32, 16), interpolation=cv2.INTER_AREA).ravel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--canvases', type=int, default=24); p.add_argument('--per_canvas', type=int, default=25)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    from realsim.realbkg_sim.mosaic_background import MosaicBackground
    from realsim.realbkg_sim.detector_masks import MaskBank
    from realsim.realbkg_sim.background_v2 import BackgroundV2
    from realsim.factory import R6
    rng = np.random.default_rng(a.seed); np.random.seed(a.seed)
    mb = MosaicBackground(seed=a.seed)
    masks = MaskBank(seed=a.seed, keep=R6['mask_keep'])
    bv = BackgroundV2(mb, canvas=R6['v2_canvas'], n_pc=R6['v2_n_pc'], seed=a.seed)
    fr = []                                            # (canvas id, is surrogate, fluctuation, envelope vector, geometry id)
    for c in range(a.canvases):
        bv.new_canvas()
        for _ in range(a.per_canvas):
            m, md = masks.draw()
            sur = rng.random() < R6['surrogate_frac']
            b, m = (bv.surrogate(mask=m) if sur else bv.frame(mask=m))
            fr.append((c, sur, fluct(b, m), envelope(b, m), md['id']))
        print(f'canvas {c + 1}/{a.canvases} done', flush=True)
    n = len(fr)
    print(f'{n} backgrounds from {a.canvases} canvases ({sum(f[1] for f in fr)} surrogate-texture frames)')

    # 1. texture reuse by FFT cross-correlation (max over all shifts), on a subset of frames
    def pick(pred, k):
        idx = [i for i in range(n) if pred(fr[i])]
        return [idx[j] for j in np.linspace(0, len(idx) - 1, min(k, len(idx))).astype(int)] if idx else []
    F = {}
    def ft(i):
        if i not in F:
            x = np.zeros((2 * H, 2 * W), np.float32); x[:H, :W] = fr[i][2]
            F[i] = (np.fft.rfft2(x), float(np.sqrt((fr[i][2] ** 2).sum())))
        return F[i]
    def peak(i, j):
        (A, na), (B, nb) = ft(i), ft(j)
        cc = np.fft.irfft2(A * np.conj(B), s=(2 * H, 2 * W))
        return float(cc.max() / max(na * nb, 1e-9))
    def report(name, pairs):
        v = np.array([peak(i, j) for i, j in pairs])
        print(f'  {name:46s} {len(v):4d} pairs | max-over-shifts correlation p50 {np.median(v):.3f}  p90 {np.percentile(v, 90):.3f}  max {v.max():.3f}  | pairs above 0.05: {(v > 0.05).sum()}')
    one_per_canvas = [next(i for i in range(n) if fr[i][0] == c and not fr[i][1]) for c in range(a.canvases) if any(fr[i][0] == c and not fr[i][1] for i in range(n))]
    report('real-tile texture, different canvases', [(i, j) for k, i in enumerate(one_per_canvas) for j in one_per_canvas[k + 1:]])
    same = [i for i in range(n) if fr[i][0] == 0 and not fr[i][1]][:12]
    report('real-tile texture, SAME canvas (expected overlap)', [(i, j) for k, i in enumerate(same) for j in same[k + 1:]])
    sur = pick(lambda f: f[1], 16)
    report('NULL: surrogate texture (cannot be shared), any canvases', [(i, j) for k, i in enumerate(sur) for j in sur[k + 1:]])
    report('real-tile vs surrogate (also a null)', [(i, j) for i in one_per_canvas[:8] for j in sur[:8]])

    # 2. envelope
    E = np.stack([f[3] for f in fr]); E = E - E.mean(0, keepdims=True)          # remove the shape every frame has (bright at low q)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    C = E @ E.T; iu = np.triu_indices(n, 1)
    print(f'ENVELOPE (population mean removed): pairwise correlation of the smooth shape over {len(iu[0])} pairs: median {np.median(C[iu]):.2f}, p90 {np.percentile(C[iu], 90):.2f}, '
          f'pairs above 0.98: {(C[iu] > 0.98).sum()} ({(C[iu] > 0.98).mean():.2%}), above 0.995: {(C[iu] > 0.995).sum()}')
    # 3. masks
    cnt = collections.Counter(f[4] for f in fr)
    print('MASKS (geometry: share of frames): ' + ', '.join(f'{k.split("_")[0]}_{k.split("_")[2] if len(k.split("_")) > 2 else ""}:{v / n:.0%}' for k, v in cnt.most_common()))
    print('canvases per 60-epoch x 1000-image run at refresh 25: %d' % (60000 // R6['v2_refresh']))


if __name__ == '__main__':
    main()
