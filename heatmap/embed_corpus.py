"""Latent map step 1: one feature vector per frame, for the corpus and the two labelled validation sets, from one frozen backbone.

  python heatmap/embed_corpus.py --out <file>.npz [--per_scan 5] [--bb dinov3_vith16plus --weights <pth>] [--max_scans N] [--device cuda]

What it does, per frame:
  1. the full 512 x 1024 polar image (chi down, q across) in the detector's own contrast (clip 5 / 99.5, log, HE), plus its no-data mask
  2. through the backbone; the stride-32 feature map (16 x 32 cells, C channels) is averaged over the cells that hold data -> one C-vector
  3. the same for the image flipped along chi, and the two vectors averaged: chi runs the other way in the corpus than in the validation files,
     so the vector must not depend on that direction
Corpus: every scan of corpus_v2 (scans + scans_other), --per_scan frames spread evenly over each scan (frames of one scan are nearly identical, so
more per scan only weights long scans). Hold-out scans (pretrain_exclude_v2.tsv) are kept and marked, they are where the validation frames come from.
Validation: all frames of organic and 41 (HM_DATA_DIR), in the order of heatmap/evaluation.iter_frames, so index i here is frame i in evaluate.py / dump_dets.py.
Output npz: feat [N, C] float16, set ('corpus' / 'organic' / '41'), path (h5 file), frame, beamtime, held_out (bool), plus bb and weights.
"""
import argparse, glob, os, sys, time, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
from types import SimpleNamespace
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from models.dinov3_backbone import DinoV3Backbone
from util.exp_preprocess import apply_contrast

DATA = '/mnt/DATA/mlgidDETECT_DINO_HEATMAP'
DEPLOYED = {'clip': (5.0, 99.5), 'log': True, 'gamma': None, 'he': True, 'clahe': None}   # = pretrain_dinov3.DEPLOYED = evaluation CONTRAST_CHANNELS[0]


def decode(v, log_max):
    """stored uint16 -> intensity, no-data mask (same as pretrain_dinov3.decode)"""
    ok = v > 0
    return np.where(ok, 10.0 ** ((v.astype(np.float32) - 1) / 65534 * log_max) - 1, 0).astype(np.float32), ok


def avail_gb():
    return int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) / 2 ** 20


@torch.no_grad()
def embed(bb, img, mask, dev):
    """img [H,W] float, mask [H,W] bool -> C-vector: masked mean of the stride-32 map, averaged over chi and its flip"""
    x = torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float32, device=dev)[None, None]
    m = torch.as_tensor(np.ascontiguousarray(mask), dtype=torch.float32, device=dev)[None, None]
    x = torch.cat([x, x.flip(2)]); m = torch.cat([m, m.flip(2)])
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
        f = bb(SimpleNamespace(tensors=x))[3].tensors.float()                  # [2, C, h, w]
    w = F.adaptive_avg_pool2d(m, f.shape[-2:])                                 # fraction of data in each cell
    w = (w > 0.5).float() if (w > 0.5).any() else w
    v = (f * w).sum((2, 3)) / w.sum((2, 3)).clamp_min(1e-6)                     # [2, C]
    return v.mean(0).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True); p.add_argument('--per_scan', type=int, default=5)
    p.add_argument('--bb', default='dinov3_vith16plus')
    p.add_argument('--weights', default=f'{DATA}/pretrain_runs/vith16plus_3k_sk2/backbone_teacher.pth')
    p.add_argument('--max_scans', type=int, default=0, help='test: only the first N scans and 3 frames per validation set')
    p.add_argument('--device', default='cuda')
    p.add_argument('--min_free_gb', type=float, default=8.0, help='pause between scans while less RAM than this is available (until 2 GB more)')
    a = p.parse_args()
    dev = a.device
    bb = DinoV3Backbone(a.bb)
    print('weights:', bb.net.load_state_dict(torch.load(a.weights, map_location='cpu', mmap=True), strict=True), flush=True)   # mmap: no second RAM copy
    bb = bb.to(dev).eval()
    import gc; gc.collect()
    rows = dict(feat=[], set=[], path=[], frame=[], beamtime=[], held_out=[])

    def add(v, s, f, i, bt, ho):
        for k, x in zip(rows, (v.astype(np.float16), s, f, i, bt, ho)):
            rows[k].append(x)

    # validation sets first (small; a failure here shows up early)
    from heatmap import evaluation as E
    for name in ('organic', '41'):
        n = 0
        for cfg, ic in E.iter_frames(E.DATASETS[name]):
            img = E.frame_inputs(ic, 'cpu')[0, 0].numpy(); mask = E.frame_mask(ic, 'cpu')[0].numpy()
            add(embed(bb, img, mask, dev), name, E.DATASETS[name], n, '', True); n += 1
            if a.max_scans and n >= 3:
                break
        print(f'{name}: {n} frames', flush=True)

    held = {x.split('\t')[1].strip() for x in open(f'{DATA}/moneta_sweep/holdout/pretrain_exclude_v2.tsv')}
    files = [f for d in (f'{DATA}/corpus_v2/scans', f'{DATA}/corpus_v2/scans_other') for f in sorted(glob.glob(os.path.join(d, '*', '*.h5')))]
    if a.max_scans:
        files = files[:a.max_scans]
    t0, skipped, nf, paused = time.time(), 0, 0, 0.0
    for k, f in enumerate(files):
        if avail_gb() < a.min_free_gb:                 # other jobs (GUI_project sweeps) have priority: wait until RAM is back
            print(f'pause at scan {k}: {avail_gb():.1f} GB available < {a.min_free_gb}', flush=True); tp = time.time()
            while avail_gb() < a.min_free_gb + 2:
                time.sleep(30)
            paused += time.time() - tp; print(f'resume after {(time.time() - tp) / 60:.0f} min', flush=True)
        with h5py.File(f, 'r') as h:
            n = h['images'].shape[0]; lm = float(h.attrs['log_max']); bt = str(h.attrs.get('beamtime', os.path.basename(os.path.dirname(f))))
            for i in sorted(set(np.linspace(0, n - 1, min(n, a.per_scan)).round().astype(int).tolist())):
                raw, ok = decode(h['images'][i], lm)
                if raw.shape != (512, 1024) or not ok.any():
                    skipped += 1; continue
                add(embed(bb, apply_contrast(raw, ok, DEPLOYED), ok, dev), 'corpus', f, i, bt, f in held); nf += 1
        if k % 200 == 0 or k == len(files) - 1:
            el = time.time() - t0
            print(f'scan {k + 1}/{len(files)}  frames {nf}  skipped {skipped}  {nf / max(el, 1e-6):.1f} frames/s  eta {el / (k + 1) * (len(files) - k - 1) / 60:.0f} min', flush=True)
    np.savez(a.out, feat=np.stack(rows['feat']), **{k: np.array(v) for k, v in rows.items() if k != 'feat'}, bb=a.bb, weights=a.weights)
    print('saved', a.out, len(rows['feat']), 'frames', flush=True)


if __name__ == '__main__':
    main()
