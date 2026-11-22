"""Region sweep (CPU): how does every model score in every region of the latent map, and are the differences between models real?

  HM_DATA_DIR=<dir with 41.h5, organic_labeled.h5> python heatmap/region_sweep.py --dets <dets dir> --regions <latent_map/vith_sk2> --out <dir> [--k 6 8 10]

Input
  dets/<run>.pkl    dump_dets.py --no_img on the 49 labelled frames (organic 8 + 41 41), labels scaled to 3.0,3.0 (queue_sweep_dump.sh; Galvani models: sbatch/sweep_dump_galvani.sbatch)
  regions_k<k>.csv  latent_map.py: region of every frame (rows with set organic / 41 are the labelled frames, frame = index in the set)
Family = run name without seed, '_galvani' and the box suffix; the seeds of one family are repeats of the SAME recipe, so their spread is the noise floor.

Per k, per region, per model: AP (the repo's evaluator, +nms detections, scaled GT) on the frames of the region pooled, plus recall / precision at score 0.3 and the number of GT boxes.
Then, with the seed spread as yardstick:
  1. check:    AP over all 49 frames recomputed from the dumps against the AP recorded in the run folder (if given in models.tsv)
  2. variance: how much of the AP differences between (model, region) cells is region, family, seed noise (balanced sums of squares on family x region cell means)
  3. special:  interaction family x region against the seed noise (approximate F test). A family that is a specialist is strong in some regions and weak in others relative to the others; if all families share their
               strong and weak regions, the interaction is small and region effects are the same for every model
  4. per region: best and worst families with the gap in units of the seed spread (only where >= 2 seeds exist)
Outputs: model_region_ap_k<k>.csv, family_region_ap_k<k>.png (absolute and relative to the median family), report.md.
Regions with fewer than --min_frames labelled frames are shown but never interpreted."""
import argparse, csv, glob, os, pickle, re, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import types
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
from heatmap import evaluation as E


def family_of(run):
    r = run.replace('_galvani', '')
    r = re.sub(r'_newbox_3\.0_3\.0', '', r)
    seed = re.search(r'_seed(\d+)', r)
    return re.sub(r'_seed\d+', '_seedN', r), (seed.group(1) if seed else '0')


def short(fam):
    f = fam.replace('hm_', '').replace('_seedN', '').replace('_ridge', '').replace('_tf32', '').replace('dinov3', 'dv3')
    return f.replace('convnextL', 'cnxL').replace('UNFROZEN', 'unfr').replace('frozen', 'froz').replace('simmim', 'swin')


def ap_of(dets, gts, cfg):
    try:
        r = E.evaluate_dets(dets, gts, cfg, thr_list=(0.3,)); t = r['thr'][0.3]
        return r['ap'], t['recall'], t['precision'], t['n_gt']
    except Exception:                                  # no matched detection at all (KeyError in the evaluator): AP 0 by convention
        n_gt = sum(len(g['gt']) for g in gts)
        return (0.0, 0.0, 0.0, n_gt) if n_gt else (np.nan, np.nan, np.nan, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dets', required=True); p.add_argument('--regions', required=True); p.add_argument('--out', required=True)
    p.add_argument('--k', type=int, nargs='+', default=[6, 8, 10]); p.add_argument('--min_frames', type=int, default=4)
    p.add_argument('--competitive', type=float, default=0.60, help='families with a mean AP below this are failing recipes: shown, but left out of the second variance/interaction test and the best/worst lines')
    p.add_argument('--sets', nargs='+', default=['organic', '41'], help='labelled sets whose frames enter the region analysis (the check against the recorded APs always uses both)')
    p.add_argument('--models', default=None, help='models.tsv (recorded APs for the check); optional')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = E.make_config(E.DATASETS['organic'])
    rec = {}
    if a.models and os.path.exists(a.models):
        for r in csv.DictReader(open(a.models), delimiter='\t'):
            rec[r['run']] = (float(r['organic_ap']), float(r['ap41']))
    reg = {}                                           # k -> {(set, idx): region}
    for k in a.k:
        reg[k] = {}
        for r in csv.DictReader(open(f'{a.regions}/regions_k{k}.csv')):
            if r['set'] in a.sets:
                reg[k][(r['set'], int(r['frame']))] = int(r['region'])
    rows, check = [], []
    files = sorted(glob.glob(f'{a.dets}/*.pkl'))
    print(len(files), 'detection files', flush=True)
    for n, f in enumerate(files):
        run = os.path.basename(f)[:-4]
        d = pickle.load(open(f, 'rb'))['sets']
        keys, dets, gts = [], [], []
        for s in ('organic', '41'):
            for i, fr in enumerate(d[s]):
                keys.append((s, i)); dets.append((np.asarray(fr['boxes'], np.float32).reshape(-1, 4), np.asarray(fr['scores'], np.float32)))
                gts.append(dict(gt=fr['gt'], gtconf=fr['gtconf'], mask=fr['mask']))
        fam, seed = family_of(run)
        if run in rec:                                 # whole-set check, per set, like evaluate.py
            for j, s in enumerate(('organic', '41')):
                idx = [i for i, kk in enumerate(keys) if kk[0] == s]
                check.append((run, s, ap_of([dets[i] for i in idx], [gts[i] for i in idx], cfg)[0], rec[run][j]))
        for k in a.k:
            for r in sorted(set(reg[k].values())):
                idx = [i for i, kk in enumerate(keys) if reg[k].get(kk) == r]
                ap, rc, pr, ng = ap_of([dets[i] for i in idx], [gts[i] for i in idx], cfg)
                rows.append(dict(run=run, family=fam, seed=seed, k=k, region=r, n_frames=len(idx), n_gt=ng, ap=ap, recall=rc, precision=pr))
        print(f'[{n + 1}/{len(files)}] {run}', flush=True)
    for k in a.k:
        with open(f'{a.out}/model_region_ap_k{k}.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows if r['k'] == k]
    rep = ['# Region sweep', '', f'{len(files)} models, {len(set(family_of(os.path.basename(f)[:-4])[0] for f in files))} families, 49 labelled frames (organic 8, 41 41). AP = repo evaluator on the pooled frames of a region, scaled GT 3.0,3.0, +nms.', '']
    if check:
        diff = [abs(c[2] - c[3]) for c in check]
        rep += ['## 1. Check: AP recomputed from the dumps vs recorded', f'{len(check)} (model, set) pairs, max |difference| {max(diff):.4f}, median {np.median(diff):.4f}.',
                'Differences above 0.002 mean the dump differs from the run\'s own evaluation (listed):', '']
        rep += [f'- {c[0]} {c[1]}: dump {c[2]:.4f} vs recorded {c[3]:.4f}' for c in check if abs(c[2] - c[3]) > 0.002][:20] or ['- none']
        rep.append('')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from scipy import stats
    for k in a.k:
        R = [r for r in rows if r['k'] == k]
        regions = sorted({r['region'] for r in R}); fams = sorted({r['family'] for r in R})
        nfr = {g: next(r['n_frames'] for r in R if r['region'] == g) for g in regions}; ngt = {g: next(r['n_gt'] for r in R if r['region'] == g) for g in regions}
        ok = [g for g in regions if nfr[g] >= a.min_frames]
        cell = collections.defaultdict(list)
        for r in R:
            if r['region'] in ok and np.isfinite(r['ap']):
                cell[(r['family'], r['region'])].append(r['ap'])
        fm = np.array([[np.mean(cell[(f, g)]) if cell[(f, g)] else np.nan for g in ok] for f in fams])
        fs = np.array([[np.std(cell[(f, g)], ddof=1) if len(cell[(f, g)]) > 1 else np.nan for g in ok] for f in fams])
        nseed = {f: max((len(cell[(f, g)]) for g in ok), default=0) for f in fams}
        rep += [f'## k = {k}', '', 'Region | labelled frames | GT boxes | note', '---|---|---|---'] + [f'{g} | {nfr[g]} | {ngt[g]} | {"" if g in ok else "too few frames, not interpreted"}' for g in regions] + ['']
        # variance decomposition on cell means of families with >= 2 seeds
        fam_mean = np.nanmean(fm, 1)
        std_sim = lambda f: 'realsim' not in f and 'PygidPos' not in f      # recipes that change only the backbone / training length, with the legacy simulator
        for label, keep in (('all families', lambda i: True), (f'competitive families only (mean AP >= {a.competitive:.2f}; failing recipes removed)', lambda i: fam_mean[i] >= a.competitive),
                            ('legacy simulator only (no realsim / pygidSIM-position variants): backbone and training differences', lambda i: std_sim(fams[i]))):
            multi = [i for i, f in enumerate(fams) if nseed[f] >= 2 and keep(i)]
            if len(multi) >= 2 and len(ok) >= 2:
                M = fm[multi]; gm = np.nanmean(M); fmean = np.nanmean(M, 1, keepdims=True); rmean = np.nanmean(M, 0, keepdims=True)
                res = M - fmean - rmean + gm
                ss_reg = np.nansum((rmean - gm) ** 2) * len(multi); ss_fam = np.nansum((fmean - gm) ** 2) * len(ok); ss_int = np.nansum(res ** 2)
                within = [(len(cell[(fams[i], g)]) - 1, np.var(cell[(fams[i], g)], ddof=1) * (len(cell[(fams[i], g)]) - 1)) for i in multi for g in ok if len(cell[(fams[i], g)]) > 1]
                df_w = sum(d for d, _ in within); ms_w = sum(s for _, s in within) / max(df_w, 1)
                nbar = np.mean([len(cell[(fams[i], g)]) for i in multi for g in ok if cell[(fams[i], g)]])
                df_i = (len(multi) - 1) * (len(ok) - 1); ms_i = ss_int / df_i
                Fv = ms_i / (ms_w / nbar); pv = 1 - stats.f.cdf(Fv, df_i, df_w)
                tot = ss_reg + ss_fam + ss_int
                contrib = np.nansum(res ** 2, 1) / ss_int; top = np.argsort(-contrib)[:4]
                rep += [f'*{label}:*', f'**Variance of the cell means** ({len(multi)} families with >= 2 seeds x {len(ok)} regions): region {ss_reg / tot:.0%}, family {ss_fam / tot:.0%}, family x region {ss_int / tot:.0%}. '
                        f'Seed noise inside a cell: sd {np.sqrt(ms_w):.3f} AP.', f'**Specialisation test** (family x region interaction against seed noise, approximate F test): F = {Fv:.2f}, df = ({df_i}, {df_w}), p = {pv:.3g}. '
                        + ('Interaction is larger than seed noise: some families really are relatively better in some regions.' if pv < 0.05 else 'No evidence that any family is relatively better in particular regions: the strong and weak regions are shared.'),
                    'Largest shares of the interaction: ' + ', '.join(f'{short(fams[multi[t]])} {contrib[t]:.0%}' for t in top), '']
        # table of family means
        rep += ['Mean AP per family and region (sd over seeds in brackets, n seeds):', '', 'family | n | ' + ' | '.join(f'R{g}' for g in ok) + ' | mean', '---|---|' + '---|' * (len(ok) + 1)]
        for i, f in enumerate(fams):
            rep.append(f'{short(f)} | {nseed[f]} | ' + ' | '.join('-' if np.isnan(fm[i, j]) else f'{fm[i, j]:.3f}' + ('' if np.isnan(fs[i, j]) else f' ({fs[i, j]:.3f})') for j in range(len(ok))) + f' | {np.nanmean(fm[i]):.3f}')
        rep.append('')
        sd_pool = np.sqrt(ms_w) if len(multi) >= 2 and len(ok) >= 2 else np.nan
        comp = np.array([fam_mean[i] >= a.competitive for i in range(len(fams))])
        nc = int(comp.sum()); chance = stats.norm.ppf(0.5 ** (1 / max(nc, 1)))
        rep += [f'Best and worst among the {nc} competitive families per region, gap to their median in units of the seed sd. Picking the best of {nc} noisy values gives about +{chance:.1f} sd by chance alone, '
                'so only a best family clearly above that is a finding:', '']
        for j, g in enumerate(ok):
            col = np.where(comp, fm[:, j], np.nan); med = np.nanmedian(col); order = np.argsort(np.where(np.isnan(col), -9, col))
            hi, lo = order[-1], order[np.where(~np.isnan(col[order]))[0][0]]
            rep.append(f'- R{g} ({nfr[g]} frames): best {short(fams[hi])} {col[hi]:.3f} ({(col[hi] - med) / sd_pool:+.1f} sd), worst {short(fams[lo])} {col[lo]:.3f} ({(col[lo] - med) / sd_pool:+.1f} sd), median {med:.3f}')
        rep.append('')
        fig, ax = plt.subplots(1, 2, figsize=(2.2 + 1.3 * len(ok) * 2, 0.35 * len(fams) + 2.2), sharey=True)
        for a_, M_, ttl, cm, kw in ((ax[0], fm, 'AP', 'viridis', {}), (ax[1], fm - np.nanmedian(fm, 0, keepdims=True), 'AP minus median family of the region', 'RdBu', dict(vmin=-0.15, vmax=0.15))):
            im = a_.imshow(M_, aspect='auto', cmap=cm, **kw); a_.set_title(f'k={k}: {ttl}', fontsize=9)
            a_.set_xticks(range(len(ok))); a_.set_xticklabels([f'R{g}\nn={nfr[g]}' for g in ok], fontsize=7)
            for (ii, jj), v in np.ndenumerate(M_):
                if np.isfinite(v):
                    a_.text(jj, ii, f'{v:.2f}', ha='center', va='center', fontsize=6, color='white' if cm == 'viridis' else 'black')
            plt.colorbar(im, ax=a_, fraction=0.04)
        ax[0].set_yticks(range(len(fams))); ax[0].set_yticklabels([f'{short(f)} (n={nseed[f]})' for f in fams], fontsize=6)
        fig.tight_layout(); fig.savefig(f'{a.out}/family_region_ap_k{k}.png', dpi=110); plt.close(fig)
    open(f'{a.out}/report.md', 'w').write('\n'.join(rep) + '\n')
    print('written', a.out)


if __name__ == '__main__':
    main()
