"""What does each stage of background generation cost, and how much diversity does it buy?

The decay in [[realbkg-overfits-its-backgrounds]] is background reuse: 48 cached mosaics, each
seen ~21x per epoch. But the mosaic GENERATOR is combinatorially huge -- it is only cached because
assembling a canvas is slow. So the question is not "how do we invent diverse backgrounds", it is
"which stage can we afford to re-run per frame". This times each stage.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
import cv2
from realbkg_sim.mosaic_background import MosaicBackground
from realbkg_sim.detector_masks import MaskBank

mb = MosaicBackground(seed=0)
mk = MaskBank(seed=0, keep='default')
H, W = 512, 1024

def t(fn, n=5):
    fn(); ts = []
    for _ in range(n):
        a = time.perf_counter(); fn(); ts.append(time.perf_counter()-a)
    return float(np.median(ts))*1000

print(f"donor frames: {len(mb.frames)}   canvas {mb.canvas}  tile {mb.tile} overlap {mb.overlap}")
step = mb.tile - mb.overlap
gr = len(range(0, mb.canvas[0], step)); gc = len(range(0, mb.canvas[1], step))
print(f"tiles per canvas: {gr} x {gc} = {gr*gc}")
print(f"crop offsets per canvas: {mb.canvas[0]-H} x {mb.canvas[1]-W} = {(mb.canvas[0]-H)*(mb.canvas[1]-W)}")
print(f"exposure pool size for a random class: ", end="")
tg = float(mb.med[mb.rng.integers(len(mb.med))])
print(int(((mb.med >= tg/2) & (mb.med <= tg*2)).sum()), "donors\n")

cv = mb.canvas_image()
print("STAGE COSTS (ms, median of 5)")
print(f"  canvas_image()        assemble {gr*gc} tiles   {t(lambda: mb.canvas_image()):8.1f}")
print(f"  _envelope()           smooth shape            {t(lambda: mb._envelope()):8.1f}")
print(f"  MaskBank.draw()                               {t(lambda: mk.draw()):8.1f}")
m, _ = mk.draw()
def crop_only():
    r = int(mb.rng.integers(0, mb.canvas[0]-H)); c = int(mb.rng.integers(0, mb.canvas[1]-W))
    return cv[r:r+H, c:c+W]
print(f"  random crop 512x1024                          {t(crop_only):8.1f}")
b = crop_only().copy()
print(f"  _noise_map equivalent (GaussianBlur sig 16)   {t(lambda: cv2.GaussianBlur(b,(0,0),16.0)):8.1f}")
def full_entry():
    bb, mm = mb.background(mask=mk.draw()[0])
    nz = cv2.GaussianBlur(bb.astype(np.float64), (0,0), 16.0)
    return bb, nz
print(f"  FULL entry (canvas+env+crop+mask+noise)       {t(full_entry, 3):8.1f}")
print("\nFor reference: one simulated FRAME (peaks+render+compose) is ~300 ms.")
