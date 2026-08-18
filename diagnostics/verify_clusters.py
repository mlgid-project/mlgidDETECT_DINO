"""
Pre-launch verification for the azimuthal peak-cluster lever (MODIFICATIONS.md phase U).

HARD GATES — this script exits non-zero if any fail, and the training job is submitted with
`--dependency=afterok`, so a broken simulator cannot start a multi-day run unattended.

  G1 REGRESSION  with use_peak_clusters=False the patched simulator must be bit-identical to the
                 pre-patch simulation.py under the same RNG seeds (the default path is untouched).
  G2 GAP MATCH   the synthetic chi-gap distribution must match the REAL organic one: p10/p25/p50/p75
                 each within a factor of 2, and the fraction of gaps below 5 px inside [0.06, 0.25]
                 (real 0.125). This is the phase-R lesson applied up front — gate on the measured
                 quantity, not on the rationale.
  G3 CLUSTERING  a realistic share of synthetic peaks must have a same-q sibling, and clusters must
                 reach size >= 5 (the stock simulator caps at 4).
  G4 SANITY      peaks/frame in a plausible range; all boxes finite, non-inverted, inside the image.
  G5 ALIGNMENT   rendered images must actually carry signal inside the label boxes (mean intensity
                 in boxes > mean outside), i.e. the new labels point at real rendered peaks.
  G6 SMOKE       main.SimulationDataset builds with the real config and yields valid tensors.

Writes diagnostics/verify_clusters.png (synthetic vs real gap distribution) for later review.
GPU, ~5 min. See tmp_diag/run_verify_clusters.sbatch.
"""
import os, sys, random, importlib.util
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

QTOL = 8.0
N_FRAMES = 60
ORIG = '/mnt/lustre/work/schreiber/szb389/tmp_diag/simulation.py.bak'
FAILS = []


def gate(ok, name, msg):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def cluster_stats(box_list):
    """Same-q clustering statistics, computed exactly as on the real labels."""
    gaps, sizes, npf, inclust, tot = [], [], [], 0, 0
    for b in box_list:
        if len(b) == 0:
            continue
        npf.append(len(b)); tot += len(b)
        q = (b[:, 0] + b[:, 2]) / 2
        c = (b[:, 1] + b[:, 3]) / 2
        used = np.zeros(len(b), bool)
        for i in np.argsort(q):
            if used[i]:
                continue
            grp = np.nonzero((~used) & (np.abs(q - q[i]) < QTOL))[0]
            used[grp] = True
            sizes.append(len(grp))
            if len(grp) > 1:
                inclust += len(grp)
                gaps += list(np.diff(np.sort(c[grp])))
    return (np.array(gaps), np.array(sizes), np.array(npf),
            inclust / tot if tot else 0.0)


def gen(sim, n, keep_imgs=0):
    """Generate via simulate_img -- the REAL training entry point. simulate_labels() cannot be
    called standalone (simulate_img sets up detector_mask first), and only simulate_img applies the
    final label post-steps, so this exercises exactly what the dataloader sees."""
    out, imgs, fails = [], [], {}
    for _ in range(n):
        try:
            img, boxes, _, _ = sim.simulate_img()
            out.append(boxes.detach().cpu().numpy())
            if len(imgs) < keep_imgs:
                imgs.append((img.detach().cpu().numpy().squeeze(), out[-1]))
        except Exception as e:
            k = f"{type(e).__name__}: {e}"
            fails[k] = fails.get(k, 0) + 1
    if fails:
        print(f"    {sum(fails.values())}/{n} frames FAILED:")
        for k, v in sorted(fails.items(), key=lambda x: -x[1])[:4]:
            print(f"      {v:4d}x  {k}")
    return out, imgs, sum(fails.values()) / max(n, 1)


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}\n")
    import simulation as sim_mod
    from simulation import FastSimulation, SimulationConfig

    # ---------------- G1: default path is untouched ----------------
    print("G1 regression: patched(clusters OFF) vs pre-patch simulation.py")
    # explicit loader: spec_from_file_location cannot infer one for a '.bak' suffix
    from importlib.machinery import SourceFileLoader
    spec = importlib.util.spec_from_loader('simulation_orig', SourceFileLoader('simulation_orig', ORIG))
    orig = importlib.util.module_from_spec(spec); spec.loader.exec_module(orig)
    same = True
    for s in range(6):
        seed_all(1000 + s); a = FastSimulation(device=dev).simulate_img()[1].detach().cpu().numpy()
        seed_all(1000 + s); b = orig.FastSimulation(device=dev).simulate_img()[1].detach().cpu().numpy()
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-5):
            same = False
            print(f"    seed {1000+s}: MISMATCH shapes {a.shape} vs {b.shape}")
            break
    gate(same, 'G1 regression', 'default path bit-identical to pre-patch' if same
         else 'DEFAULT PATH CHANGED — patch is not opt-in')

    # ---------------- generate with clusters ON ----------------
    sc = SimulationConfig(); sc.use_peak_clusters = True
    seed_all(7)
    sim_on = FastSimulation(sim_config=sc, device=dev)
    boxes_on, montage, fail_on = gen(sim_on, N_FRAMES, keep_imgs=6)
    seed_all(7)
    sim_off = FastSimulation(device=dev)
    boxes_off, _, fail_off = gen(sim_off, N_FRAMES)

    # G0 must come first: a high frame-failure rate makes every later statistic meaningless, and
    # main.SimulationDataset.__getitem__ swallows simulate_img exceptions and retries -- so a broken
    # generator would silently cripple training throughput instead of crashing.
    gate(fail_on <= 0.05 and fail_off <= 0.05, 'G0 generation',
         f'frame failure rate: clusters ON {fail_on:.1%}, OFF {fail_off:.1%} (need <=5%)')
    if not boxes_on or not boxes_off:
        gate(False, 'G0 generation', 'no frames generated at all — aborting remaining gates')
        print(f"\nVERIFICATION FAILED: {', '.join(FAILS)}")
        return 1

    g_on, s_on, n_on, frac_on = cluster_stats(boxes_on)
    g_off, s_off, n_off, frac_off = cluster_stats(boxes_off)

    # ---------------- real target ----------------
    from util.configuration import Config
    from util.exp_preprocess import standard_preprocessing
    from util.pygidloader import PyGIDDataset
    DS = '/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5'
    cfg = Config(); cfg.PREPROCESSING_POLAR_SHAPE = [512, 1024]; cfg.POSTPROCESSING_SCORE = 0.1
    cfg.INPUT_DATASET = DS
    ds = PyGIDDataset(cfg, path=DS, preprocess_func=standard_preprocessing, buffer_size=3,
                      load_labels=True)
    real = [np.array(gc.polar_labels.boxes) for gc in ds.iter_images()]
    if hasattr(ds, 'close'):
        ds.close()
    g_re, s_re, n_re, frac_re = cluster_stats(real)

    print(f"\n  {'quantity':34s} {'REAL organic':>14s} {'sim CLUSTERS ON':>16s} {'sim OFF':>10s}")
    print(f"  {'peaks / frame':34s} {np.mean(n_re):14.1f} {np.mean(n_on):16.1f} {np.mean(n_off):10.1f}")
    print(f"  {'peaks with a same-q sibling':34s} {frac_re:13.1%} {frac_on:15.1%} {frac_off:9.1%}")
    mx_ = lambda a: int(a.max()) if len(a) else 0
    print(f"  {'max cluster size':34s} {mx_(s_re):14d} {mx_(s_on):16d} {mx_(s_off):10d}")
    for p in (10, 25, 50, 75):
        f = lambda g: np.percentile(g, p) if len(g) else float('nan')
        print(f"  {'chi-gap p%-2d (px)' % p:34s} {f(g_re):14.1f} {f(g_on):16.1f} {f(g_off):10.1f}")
    f5 = lambda g: np.mean(g < 5) if len(g) else float('nan')
    print(f"  {'chi-gaps < 5 px':34s} {f5(g_re):13.1%} {f5(g_on):15.1%} {f5(g_off):9.1%}")

    # ---------------- G2: gap distribution match ----------------
    print("\nG2 gap-distribution match vs real organic")
    ok2 = len(g_on) > 100
    if not ok2:
        gate(False, 'G2 gaps', f'only {len(g_on)} synthetic gaps generated')
    else:
        for p in (10, 25, 50, 75):
            r, o = np.percentile(g_re, p), np.percentile(g_on, p)
            good = (o / r < 2.0) and (r / o < 2.0)
            ok2 &= good
            print(f"    p{p:<2d}: real {r:6.1f}  sim {o:6.1f}  ratio {o/r:4.2f}  "
                  f"{'ok' if good else 'OUT OF BAND (need 0.5-2.0)'}")
        t = f5(g_on)
        good = 0.06 <= t <= 0.25
        ok2 &= good
        print(f"    frac<5px: real {f5(g_re):.3f}  sim {t:.3f}  "
              f"{'ok' if good else 'OUT OF BAND (need 0.06-0.25)'}")
        gate(ok2, 'G2 gaps', 'synthetic chi-gap distribution matches real')

    # ---------------- G3: clustering present ----------------
    gate(0.40 <= frac_on <= 0.97 and mx_(s_on) >= 5, 'G3 clustering',
         f'{frac_on:.1%} of peaks have a same-q sibling (real {frac_re:.1%}), '
         f'max cluster size {mx_(s_on)} (stock sim caps at 4: off-run max {mx_(s_off)})')

    # ---------------- G4: sanity ----------------
    allb = np.concatenate([b for b in boxes_on if len(b)])
    finite = np.isfinite(allb).all()
    noninv = bool(((allb[:, 2] > allb[:, 0]) & (allb[:, 3] > allb[:, 1])).all())
    inim = bool((allb[:, 0] >= -1).all() and (allb[:, 2] <= sim_mod.WIDTH + 1).all())
    gate(finite and noninv and inim and 20 <= np.mean(n_on) <= 500, 'G4 sanity',
         f'{len(allb)} boxes: finite={finite} non-inverted={noninv} in-image={inim} '
         f'peaks/frame={np.mean(n_on):.0f}')

    # ---------------- G5: labels sit on rendered signal ----------------
    print("\nG5 alignment: does rendered signal actually appear inside the new boxes?")
    ratios = []
    for im, bb in montage:
        m = np.zeros(im.shape, bool)
        for b in bb:
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(im.shape[1], max(x2, x1 + 1)); y2 = min(im.shape[0], max(y2, y1 + 1))
            m[y1:y2, x1:x2] = True
        if m.any() and (~m).any():
            ratios.append(float(im[m].mean() / max(im[~m].mean(), 1e-6)))
    gate(len(ratios) > 0 and np.median(ratios) > 1.05, 'G5 alignment',
         f'mean intensity inside boxes / outside = {np.median(ratios):.2f}x (need >1.05)'
         if ratios else 'no images rendered')

    # ---------------- G6: dataset smoke ----------------
    print("\nG6 dataset smoke via main.SimulationDataset + the real config")
    try:
        from main import get_args_parser, SimulationDataset
        from util.slconfig import SLConfig
        args = get_args_parser().parse_args([])
        for k, v in SLConfig.fromfile(
                os.path.join(_REPO, 'config/DINO/DINO_4scale_swin_clusters.py'))._cfg_dict.to_dict().items():
            setattr(args, k, v)
        dsx = SimulationDataset(args)          # __init__(self, args, transforms=None, device='cuda')
        it = dsx[0]
        img_t = it[0] if isinstance(it, (tuple, list)) else it
        ok6 = torch.is_tensor(img_t) and torch.isfinite(img_t).all()
        gate(ok6, 'G6 smoke', f'SimulationDataset[0] ok, image {tuple(img_t.shape)}')
    except Exception as e:
        gate(False, 'G6 smoke', f'{type(e).__name__}: {e}')

    # ---------------- figure ----------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    bins = np.logspace(0, np.log10(400), 40)
    ax[0].hist(g_re, bins=bins, density=True, alpha=.55, label=f'REAL organic (n={len(g_re)})')
    ax[0].hist(g_on, bins=bins, density=True, alpha=.55, label=f'sim clusters ON (n={len(g_on)})')
    if len(g_off):
        ax[0].hist(g_off, bins=bins, density=True, histtype='step', lw=1.6,
                   label=f'sim OFF (n={len(g_off)})')
    ax[0].axvline(5, ls=':', c='k'); ax[0].set_xscale('log')
    ax[0].set_xlabel('chi-gap between same-q peaks (px)'); ax[0].set_ylabel('density')
    ax[0].set_title('chi-gap distribution (G2)'); ax[0].legend(fontsize=8)
    mx = max(mx_(s_re), mx_(s_on), mx_(s_off), 1)
    for lab, sv in (('REAL', s_re), ('ON', s_on), ('OFF', s_off)):
        h = np.bincount(sv, minlength=mx + 1)[1:mx + 1]
        ax[1].plot(range(1, mx + 1), h / h.sum(), 'o-', label=lab)
    ax[1].set_xlabel('peaks per same-q cluster'); ax[1].set_ylabel('fraction')
    ax[1].set_title('cluster size (G3)'); ax[1].legend(fontsize=8); ax[1].set_yscale('log')
    if montage:
        im, bb = montage[0]
        ax[2].imshow(im, origin='lower', aspect='auto', cmap='gray')
        for b in bb[:400]:
            ax[2].add_patch(plt.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False,
                                          ec='lime', lw=0.5))
        ax[2].set_title('synthetic frame + labels (G5)'); ax[2].set_xlabel('q'); ax[2].set_ylabel('chi')
    fig.suptitle('Phase U pre-launch verification — azimuthal peak clusters', fontsize=12)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verify_clusters.png')
    fig.savefig(out, dpi=100, bbox_inches='tight')
    print(f"\nsaved {out}")

    print("\n" + "=" * 70)
    if FAILS:
        print(f"VERIFICATION FAILED: {', '.join(FAILS)}")
        print("Training job will NOT start (submitted with --dependency=afterok).")
        return 1
    print("ALL GATES PASSED — training may launch.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
