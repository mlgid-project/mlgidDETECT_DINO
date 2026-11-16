"""Is the real-tile TEXTURE fresh per frame, or re-cropped from one canvas that lives 200 frames?

The earlier freshness check hashed the finished background, which already carries a freshly
sampled envelope and mask -- so it would report "all distinct" even if every frame shared one
texture. This hashes the CROP itself, and measures how much of the canvas each frame actually
re-uses, which is the quantity that decides whether the network can memorise it.
"""
import os, sys, time, hashlib
import numpy as np
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
from realbkg_sim.mosaic_background import MosaicBackground
from realbkg_sim.background_v2 import BackgroundV2, HEIGHT, WIDTH

mb = MosaicBackground(seed=3)
for CAN in [(768, 1536), (1536, 3072), (2048, 4096)]:
    bv = BackgroundV2(mb, canvas=CAN, n_pc=6, seed=3)
    t = time.perf_counter(); bv.new_canvas(); dt = time.perf_counter()-t
    H, W = CAN
    area = H*W
    frame = HEIGHT*WIDTH
    print(f'canvas {str(CAN):14} build {dt*1000:7.1f} ms   area {area/frame:4.1f} frames worth')
    # how distinct are consecutive crops from ONE canvas?
    hs, off = [], []
    for _ in range(40):
        p = bv.crop()
        hs.append(hashlib.md5(np.ascontiguousarray(p).tobytes()).hexdigest()[:12])
    print(f'   crops hashed distinct: {len(set(hs))}/40  (trivially true -- offsets differ)')
    # the real question: pixel re-use. Expected overlap between two random crops.
    rs = np.random.default_rng(0).integers(0, max(H-HEIGHT, 1), 4000)
    cs = np.random.default_rng(1).integers(0, max(W-WIDTH, 1), 4000)
    ov = []
    for i in range(0, 4000, 2):
        dr = abs(int(rs[i])-int(rs[i+1])); dc = abs(int(cs[i])-int(cs[i+1]))
        ov.append(max(0, HEIGHT-dr)*max(0, WIDTH-dc)/frame)
    print(f'   mean pixel overlap between two random crops: {np.mean(ov)*100:5.1f}%')
    for refresh in (200, 50, 25):
        per_frame = dt*1000/refresh
        reuse = refresh*frame/area
        print(f'   refresh {refresh:4d}: {per_frame:6.1f} ms/frame amortised, '
              f'each canvas pixel used ~{reuse:4.1f}x, {1000//refresh:3d} canvases/epoch')
    print()
print('v1 for reference: 48 FINISHED backgrounds, identical pixels, each seen ~21x per epoch.')
