"""~25 simple frames under a candidate LABELLING CONVENTION, as PNGs, for visual judgement.

The convention this renders (all overridable):
  * RENDER IFF LABELLED -- one gate governs both, so no visible peak is left without a box and
    no box sits on nothing. This is the substantive change; the thresholds below are dials.
  * a peak is visible when its height is >= contrast_min x the LOCAL BACKGROUND NOISE and its
    matched-filter SNR is >= snr_min. Two criteria because a broad faint arc integrates to
    something a human would mark while a 4-pixel blip at the same height does not.
  * overlapping boxes are suppressed greedily by amplitude: seg_iou_max between segments,
    ring_iou_max between rings. Suppression removes the PEAK, not just its box.
  * at most max_peaks reflections are drawn per frame, before any gate.

Writes boxconv_frame_NN_<recipe>.png plus a one-line summary per frame to stdout.
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
HEIGHT, WIDTH = 512, 1024


def pw_iou(b):
    if len(b) < 2:
        return np.zeros(0)
    x1 = np.maximum(b[:, None, 0], b[None, :, 0]); y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2]); y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    it = np.clip(x2-x1, 0, None)*np.clip(y2-y1, 0, None)
    ar = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return (it/np.maximum(ar[:, None]+ar[None, :]-it, 1e-9))[np.triu_indices(len(b), k=1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=25)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--config', default='config/DINO/DINO_4scale_swin_realbkg.py')
    ap.add_argument('--bank', default=None)
    ap.add_argument('--contrast-min', type=float, default=2.0)
    ap.add_argument('--snr-min', type=float, default=6.0)
    ap.add_argument('--seg-iou-max', type=float, default=0.30)
    ap.add_argument('--ring-iou-max', type=float, default=0.10)
    ap.add_argument('--max-peaks', type=int, default=200)
    ap.add_argument('--cmap', default='magma', help='matplotlib colormap for the frame')
    ap.add_argument('--layout', default='quad', choices=['quad', 'side', 'stacked', 'boxed'],
                    help="'quad' = 2x2, raw and preprocessed, each without and with boxes; "
                         "'side' = preprocessed clean | boxed; 'stacked' = the same one above "
                         "the other; 'boxed' = the single boxed panel only")
    ap.add_argument('--out-dir', default='/mnt/lustre/work/schreiber/szb389/tmp_diag/sim2/images/08_box_convention')
    ap.add_argument('--donor-cache',
                    default='/mnt/lustre/work/schreiber/szb389/datasets/realbkg_donors_selected.h5')
    args = ap.parse_args()

    import random, torch
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    from util.slconfig import SLConfig
    from simulation import SimulationConfig
    import realbkg_simulation as RS
    from realbkg_simulation import RealBkgSimulation

    cfg = SLConfig.fromfile(os.path.join(_REPO, args.config))
    sc = SimulationConfig()
    coefs = getattr(cfg, 'box_coef_override', None) or (2.80, 1.30)
    sc.a_coef, sc.w_coef = float(coefs[0]), float(coefs[1])

    from diagnostics.cache_realbkg_donors import load_into
    _ld = RealBkgSimulation._load_donors
    if os.path.exists(args.donor_cache):
        RealBkgSimulation._load_donors = lambda self, *a, **k: load_into(self, args.donor_cache)

    bank = args.bank or cfg.physics_bank_path
    sim = RealBkgSimulation(
        bank_path=bank, donor_path=cfg.realbkg_donor_path,
        stats_path=cfg.realbkg_stats_path, sim_config=sc, device='cpu',
        n_oriented=tuple(getattr(cfg, 'realbkg_n_oriented', (1, 3))),
        p_ring=float(getattr(cfg, 'realbkg_p_ring', 0.15)),
        contrast_min=args.contrast_min, snr_min=args.snr_min,
        ring_iou_max=args.ring_iou_max, seg_iou_max=args.seg_iou_max,
        unified_labels=True, max_peaks=args.max_peaks,
        mosaic=bool(getattr(cfg, 'realbkg_mosaic', False)),
        mosaic_pool=int(getattr(cfg, 'realbkg_mosaic_pool', 48)),
        mosaic_refresh=int(getattr(cfg, 'realbkg_mosaic_refresh', 64)),
        mosaic_seed=getattr(cfg, 'realbkg_mosaic_seed', None),
        intensity_decades=getattr(cfg, 'realbkg_intensity_decades', None),
        amplitude_mode=getattr(cfg, 'realbkg_amplitude_mode', 'fitted'),
        mask_bank=bool(getattr(cfg, 'realbkg_mask_bank', True)),
        mask_keep=getattr(cfg, 'realbkg_mask_keep', 'default'))
    RealBkgSimulation._load_donors = _ld

    # simulate_img() returns the CONTRAST-CHAINED frame (clip p5/p99.5 -> log10 -> HE), which is
    # the model's input but has every physical intensity destroyed. Intercept apply_contrast to
    # keep `total` as well: the linear frame in real count units, donor + peaks + counting noise.
    snap = {}
    _ac = RS.apply_contrast
    def _ac_hook(total, mask, chain):
        snap['raw'] = np.asarray(total).copy(); snap['mask'] = np.asarray(mask).copy()
        return _ac(total, mask, chain)
    RS.apply_contrast = _ac_hook

    print(f'bank {bank}\nCONVENTION  render-iff-labelled | contrast >= {args.contrast_min} x local '
          f'noise | snr >= {args.snr_min} | IoU seg {args.seg_iou_max} ring {args.ring_iou_max} '
          f'| max {args.max_peaks} peaks drawn', flush=True)

    # (label, oriented entries, reflections per entry, powder entries)
    RECIPE = [
        ('segments-verysparse', (1, 1), (3,    10), (0, 0)),
        ('segments-sparse',     (1, 1), (10,   30), (0, 0)),
        ('segments-light',      (1, 2), (25,   60), (0, 0)),
        ('segments-medium',     (2, 2), (50,  100), (0, 0)),
        ('segments-dense',      (2, 3), (100, 200), (0, 0)),
        ('segments-max',        (3, 3), (150, 200), (0, 0)),
        ('rings-single',        (0, 0), (2,   200), (1, 1)),
        ('rings-few',           (0, 0), (2,   200), (2, 2)),
        ('rings-many',          (0, 0), (2,   200), (3, 3)),
        ('mixed-sparse',        (1, 1), (5,   15), (1, 1)),
        ('mixed-light',         (1, 2), (20,  50), (1, 1)),
        ('mixed-medium',        (2, 2), (40,  90), (1, 2)),
        ('mixed-dense',         (3, 3), (100, 200), (1, 2)),
        ('mixed-max',           (3, 3), (150, 200), (2, 3)),
        ('ringheavy-sparse',    (1, 1), (3,   12), (2, 3)),
        ('ringheavy-dense',     (2, 3), (80,  200), (3, 3)),
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    seg_all, ring_all, counts = [], [], []
    made = 0
    while made < args.frames:
        lab, no, sp, npw = RECIPE[made % len(RECIPE)]
        sim.n_oriented, sim.spots_cap, sim.n_powder = no, sp, npw
        sim.p_ring = 1.0 if npw[1] > 0 else 0.0
        r = sim.simulate_img()
        if r is None:
            continue
        img, bx, mask, rg = (np.asarray(v.cpu()) for v in r)
        seg, ring = bx[~rg.astype(bool)], bx[rg.astype(bool)]
        si, ri = pw_iou(seg), pw_iou(ring)
        seg_all.append(si); ring_all.append(ri)
        counts.append((len(bx), int(rg.sum())))

        # magma runs black -> purple -> orange -> pale yellow, so the box colours have to sit
        # off that ramp entirely or they vanish over bright peaks: cyan and spring green do.
        seg_c, ring_c, seg_n, ring_n = (('#00e5ff', '#76ff03', 'cyan', 'green')
                                        if args.cmap == 'magma' else
                                        ('#39ff14', '#ff9d00', 'green', 'orange'))

        def draw_boxes(ax):
            for b in seg:
                ax.add_patch(Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False,
                                       lw=1.0, ec=seg_c, alpha=0.9))
            for b in ring:
                ax.add_patch(Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False,
                                       lw=1.2, ec=ring_c, alpha=0.9))

        def show(ax, arr, norm=None):
            ax.imshow(arr, cmap=args.cmap, origin='lower', aspect='auto', norm=norm)
            ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT)
            ax.set_xlabel('q  (polar px)'); ax.set_ylabel('chi  (polar px)')

        if args.layout == 'quad':
            raw, m = snap['raw'], snap['mask'].astype(bool)
            # A LOG colour scale, not HE. HE is a rank transform: it equalises the histogram and
            # so shows faint structure at the cost of destroying every intensity ratio. Log keeps
            # the ratios, which is the point of looking at the raw frame at all. Floor at the
            # background median so the scale spans background -> brightest peak.
            lo = max(float(np.median(raw[m])), 1e-3)
            hi = max(float(np.percentile(raw[m], 99.99)), lo*1.01)
            norm = LogNorm(vmin=lo, vmax=hi, clip=True)
            fig, axs = plt.subplots(2, 2, figsize=(26, 13.6), sharex=True, sharey=True)
            show(axs[0, 0], raw, norm); show(axs[0, 1], raw, norm)
            show(axs[1, 0], img);      show(axs[1, 1], img)
            draw_boxes(axs[0, 1]); draw_boxes(axs[1, 1])
            axs[0, 0].set_title(f'RAW, linear counts on a log colour scale  '
                                f'[{lo:.3g} .. {hi:.3g}]   max/median '
                                f'{float(raw[m].max()/lo):.0f}', fontsize=11)
            axs[0, 1].set_title('RAW + boxes', fontsize=11)
            axs[1, 0].set_title('PREPROCESSED, what the model sees  '
                                '(clip p5/p99.5 -> log10 -> histogram equalisation)', fontsize=11)
            axs[1, 1].set_title(f'PREPROCESSED + boxes   {len(bx)} boxes  '
                                f'({len(seg)} segments {seg_n}, {len(ring)} rings {ring_n})',
                                fontsize=11)
            fig.suptitle(f'{made:02d}  {lab}', fontsize=14)
        else:
            if args.layout == 'side':
                fig, axs = plt.subplots(1, 2, figsize=(26, 7.0), sharex=True, sharey=True)
            elif args.layout == 'stacked':
                fig, axs = plt.subplots(2, 1, figsize=(15, 15.6), sharex=True, sharey=True)
            else:
                fig, ax1 = plt.subplots(figsize=(16, 8.4)); axs = [ax1]
            for ax in np.ravel(axs):
                show(ax, img)
            draw_boxes(np.ravel(axs)[-1])
            if args.layout != 'boxed':
                np.ravel(axs)[0].set_title('no boxes -- is every visible peak one we labelled?',
                                           fontsize=11)
                np.ravel(axs)[-1].set_title(f'{len(bx)} boxes  ({len(seg)} segments {seg_n}, '
                                            f'{len(ring)} rings {ring_n})', fontsize=11)
                fig.suptitle(f'{made:02d}  {lab}', fontsize=13)
            else:
                np.ravel(axs)[-1].set_title(
                    f'{made:02d}  {lab}   {len(bx)} boxes  '
                    f'({len(seg)} segments {seg_n}, {len(ring)} rings {ring_n})', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f'boxconv_frame_{made:02d}_{lab}.png'), dpi=100)
        plt.close(fig)
        print(f'  {made:02d} {lab:<20s} {len(bx):3d} boxes ({len(ring):2d} ring)  '
              f'seg-pairs>0.1 {int((si>0.1).sum()):3d} >0.3 {int((si>0.3).sum()):3d}',
              flush=True)
        made += 1

    si = np.concatenate(seg_all) if seg_all else np.zeros(0)
    ri = np.concatenate(ring_all) if ring_all else np.zeros(0)
    c = np.array([a for a, _ in counts]); nr = np.array([b for _, b in counts])
    print(f'\n{args.out_dir}  --  {made} frames')
    print(f'  boxes/frame  min {c.min()} p50 {int(np.median(c))} p90 {int(np.percentile(c,90))} '
          f'max {c.max()}     [real: organic p50 66 max 168 | 41 p50 20 max 65]')
    print(f'  rings/frame  min {nr.min()} p50 {int(np.median(nr))} max {nr.max()}')
    for tag, d in (('segment-segment', si), ('ring-ring', ri)):
        if not len(d):
            print(f'  {tag:<16s} no pairs'); continue
        nz = d[d > 0]
        print(f'  {tag:<16s} {len(d)} pairs, {len(nz)} overlap | '
              f'>0.1 {int((d>0.1).sum())}  >0.2 {int((d>0.2).sum())}  >0.3 {int((d>0.3).sum())}  '
              f'>0.4 {int((d>0.4).sum())}  max {d.max() if len(d) else 0:.3f}')
    print('  [real segment-segment: organic max 0.154, 41 max 0.400; ring-ring: 1 pair in 2220]')


if __name__ == '__main__':
    main()
