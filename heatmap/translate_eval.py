"""Sim-to-real translation, step 3: translate the held-out sim frames and check (a) that the boxes survive, (b) where the frames land (embed_arrays.py + latent_sim.py afterwards).

  PYTHONPATH=. python heatmap/translate_eval.py --data <latent_map/translate> --ckpt <run>/G_last.pt --out <dir> [--first 3600] [--n 400]

Frames first..first+n of sim.npy (never seen by the GAN, which trains on frames < --n_train_sim = 3600) are translated at full resolution with the same mask. Writes <out>/translated.npy (uint8, = input of embed_arrays.py), copies of the sim frames / masks, and box_report.txt:
box contrast = mean image value inside the GT box minus the mean in the ring between the box and the box grown to 2 x (valid pixels only), before and after translation. 'kept' = the contrast has the same sign and at least half the size
(boxes with fewer than 20 valid pixels inside or in the ring are skipped). A translation that deletes or moves peaks shows up as a low 'kept' fraction."""
import argparse, os, pickle, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, torch
from heatmap.translate_train import G


def contrast(img, m, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    w, h = x2 - x1, y2 - y1
    X1, Y1, X2, Y2 = max(0, x1 - w // 2), max(0, y1 - h // 2), min(1024, x2 + w // 2), min(512, y2 + h // 2)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(1024, x2), min(512, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    inner = np.zeros(m.shape, bool); inner[y1:y2, x1:x2] = True; outer = np.zeros(m.shape, bool); outer[Y1:Y2, X1:X2] = True
    ring = outer & ~inner & m; inner &= m
    if inner.sum() < 20 or ring.sum() < 20:
        return None
    return float(img[inner].mean() - img[ring].mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True); p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True); p.add_argument('--first', type=int, default=3600); p.add_argument('--n', type=int, default=400); p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True); dev = 'cuda'; torch.manual_seed(a.seed)
    sI, sM = np.load(f'{a.data}/sim.npy', mmap_mode='r'), np.load(f'{a.data}/sim_mask.npy', mmap_mode='r'); boxes = pickle.load(open(f'{a.data}/sim_boxes.pkl', 'rb'))
    g = G().to(dev); g.load_state_dict(torch.load(a.ckpt, map_location=dev)); g.eval()
    T, S, Mk = np.zeros((a.n, 512, 1024), np.uint8), np.zeros((a.n, 512, 1024), np.uint8), np.zeros((a.n, 512, 1024), np.uint8)
    rows = []
    with torch.no_grad():
        for k in range(a.n):
            j = a.first + k
            x = torch.from_numpy(np.asarray(sI[j])).to(dev).float()[None, None] / 255; m = torch.from_numpy(np.asarray(sM[j])).to(dev).float()[None, None]
            y = g(x, m)[0, 0].cpu().numpy(); T[k] = np.round(y * 255).astype(np.uint8); S[k] = sI[j]; Mk[k] = sM[j]
            xs, mb = x[0, 0].cpu().numpy(), sM[j].astype(bool)
            for b, r in zip(boxes[j]['boxes'], boxes[j]['is_ring']):
                c0, c1 = contrast(xs, mb, b), contrast(y, mb, b)
                if c0 is not None:
                    rows.append((bool(r), c0, c1))
    np.save(f'{a.out}/translated.npy', T); np.save(f'{a.out}/sim.npy', S); np.save(f'{a.out}/mask.npy', Mk)
    out = [f'{a.n} held-out sim frames, {len(rows)} boxes with enough valid pixels', f'mean |pixel change| (translated - sim, valid pixels): {np.abs(T.astype(np.float32) - S.astype(np.float32))[Mk > 0].mean() / 255:.4f}']
    for name, sel in (('all', lambda r: True), ('segments', lambda r: not r), ('rings', lambda r: r)):
        v = np.array([(c0, c1) for ring, c0, c1 in rows if sel(ring)])
        if len(v) == 0:
            continue
        pos = v[:, 0] > 0
        kept = ((v[:, 1] * v[:, 0] > 0) & (np.abs(v[:, 1]) >= 0.5 * np.abs(v[:, 0])))
        out.append(f'{name}: {len(v)} boxes, median contrast sim {np.median(v[:, 0]):+.3f} -> translated {np.median(v[:, 1]):+.3f}, kept {kept.mean():.0%}' + (f' (of the {pos.sum()} positive-contrast boxes: kept {kept[pos].mean():.0%})' if pos.any() else ''))
    open(f'{a.out}/box_report.txt', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))


if __name__ == '__main__':
    main()
