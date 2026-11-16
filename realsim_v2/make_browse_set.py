"""Render frames with the old realsim (realsim/realbkg_simulation.py, E9 settings) for browsing -- notebook: /mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim/browse_realsim.ipynb

    python realsim_v2/make_browse_set.py --bank /mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim/bank/bank_v2peaks_60k.npz --n 150 --seed 5 --out NAME.npz   # one structure per frame, peaks of the realsim_v2 shards
    python realsim_v2/make_browse_set.py --bank old --n 150 --seed 5 --out NAME.npz                                                                  # old organic bank, as E9 (needs ~7 GB RAM)

Output npz: img float16 [N, 512, 1024] (model input), mask packed bits, boxes float32 [M, 4] (xyxy, box_coef 3.0/3.0), is_ring [M], box_off [N+1], opts (json).
E9 below = the SIM list of heatmap/queue_realsim_v2.sh (run hm_simmim_frozen_ridge_realsimE9flip_...).
"""
import argparse, json, os, random, sys
import numpy as np

os.environ.setdefault('GIWAXS_WORK', '/mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_data')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

E9 = dict(amplitude_mode='fitted', intensity_decades=None, ring_gain=0.4, ring_contrast_min=0.3, ring_chi_texture=0.05, ring_chi_corr=(8.0, 64.0), seg_sc_scale=0.6,
          ring_sq_scale=1.0, ring_w_coef=2.2, ring_no_elongate=True, ring_cap=4.0, ring_chi_floor=0.5, ring_follow_bkg=True, width_jitter=(0.2, 0.832), bkg_noise_blur=(0.0, 1.2),
          bkg_noise_power=1.5, mask_keep=[0, 3, 6, 7], ring_x_share=(0.106, 0.179, 0.257, 0.238, 0.220), seg_wide_frac=0.0, p_ring=0.85, n_powder=(1, 2), rings_cap=(6, 30),
          n_oriented=(2, 3), spots_cap=(25, 120), sq_q_slope=0.6, mask_wedge_only_frac=0.2)
STRUCTURE = dict(structure_frames=True, spots_cap=(200, 200), rings_cap=(15, 15))     # one structure per frame, every stored peak (the tables are already a selection)
V2_WIDTHS = dict(v2_widths=True, ring_w_coef=None)    # widths and eta of the v2 shards; ring boxes at the same 3.0 sigma as the segments (E9: 2.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', required=True, help="path of a peak bank npz (realsim_v2/build_peak_bank.py) or 'old' for the organic bank of E9")
    ap.add_argument('--v2_widths', type=int, default=1, help='1 = peak widths / eta / ring box width as in realsim_v2 (needs a bank built with widths), 0 = fitted draws of the old realsim')
    ap.add_argument('--width_scale', default='1,1', help='sq,sc: multiplies the v2 peak widths (radial, angular), e.g. 2,2')
    ap.add_argument('--eta', default=None, help='lo,hi of the pseudo-Voigt mixing (v1 draws 0.6,1.0 = Lorentzian-heavy wings, v2 0.1,0.6)')
    ap.add_argument('--v1_width_scale', default='1,1', help='sq,sc multiplier of the fitted v1 widths (with --v2_widths 0)')
    ap.add_argument('--blur', default=None, help='lo,hi of bkg_noise_blur (E9: 0,1.2); sigma of the noise correlation in px')
    ap.add_argument('--extra_p', type=float, default=0.0, help='probability to add a second, dense structure of the old organic bank (the other kind: rings for spots-only frames and vice versa; same kind 10 %%)')
    ap.add_argument('--extra_spots', default=None, help='lo,hi cap of spots of the second structure (default 25,120)')
    ap.add_argument('--extra_low_q', type=float, default=None, help='scale (1/A) of an exp(-q/scale) weight that favours low-q spots of the second structure')
    ap.add_argument('--extra_same', type=float, default=0.1, help='probability that the second structure is of the same kind as the frame (2 oriented / 2 powder)')
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import torch
    from realsim.factory import make_realbkg_sim
    opts = dict(E9)
    bank = None if a.bank == 'old' else a.bank
    if bank:
        opts.update(STRUCTURE)
        if a.extra_p > 0:
            opts['extra_structure_p'] = a.extra_p
            opts['extra_same_kind'] = a.extra_same
            if a.extra_spots:
                opts['extra_spots_cap'] = tuple(int(v) for v in a.extra_spots.split(','))
            if a.extra_low_q:
                opts['extra_low_q'] = a.extra_low_q
        if a.blur:
            opts['bkg_noise_blur'] = tuple(float(v) for v in a.blur.split(','))
        if a.eta:
            opts['voigt_eta'] = tuple(float(v) for v in a.eta.split(','))
        opts['v1_width_scale'] = tuple(float(v) for v in a.v1_width_scale.split(','))
        if a.v2_widths:
            opts.update(V2_WIDTHS, v2_width_scale=tuple(float(v) for v in a.width_scale.split(',')))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    sim = make_realbkg_sim(device='cuda' if torch.cuda.is_available() else 'cpu', box_coef=(3.0, 3.0), overrides=opts, bank_path=bank, preset='r6')
    imgs, masks, boxes, rings, off = [], [], [], [], [0]
    while len(imgs) < a.n:
        r = sim.simulate_img()
        if r is None:
            continue
        img, bx, m, rg = (t.cpu().numpy() for t in r)
        imgs.append(img.astype(np.float16)); masks.append(np.packbits(m)); boxes.append(bx.astype(np.float32)); rings.append(rg.astype(bool)); off.append(off[-1] + len(bx))
        if len(imgs) % 25 == 0:
            print(len(imgs), 'frames', flush=True)
    np.savez(a.out, img=np.stack(imgs), mask=np.stack(masks), boxes=np.concatenate(boxes), is_ring=np.concatenate(rings), box_off=np.array(off),
             opts=json.dumps(dict(bank=a.bank, seed=a.seed, options={k: v for k, v in opts.items()}), default=str))
    print('saved', a.out, len(imgs), 'frames, mean boxes', np.mean(np.diff(off)), 'mean rings', np.mean([r.sum() for r in rings]))


if __name__ == '__main__':
    main()
