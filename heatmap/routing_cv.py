"""Does 'one model per region' pay off on frames the choice was NOT made on?  (CPU, same dumps as region_sweep.py)

  HM_DATA_DIR=<dir> python heatmap/routing_cv.py --dets <dets dir> --regions <latent_map/vith_sk2> --k 6 --out <file.md> [--repeats 20] [--min_frames 6]

Per repeat: the labelled frames of every region (with >= --min_frames frames) are split at random into halves A and B.
  global : the family with the best frame-weighted mean of its region APs on A  -> its score on B
  routed : per region the family with the best AP on A in that region           -> the score on B of the routed choices (frame-weighted mean over regions)
  oracle : per region the family with the best AP on B itself (what in-sample selection would claim; optimistic, not reachable)
A family's AP in a cell = mean over its seeds (models of the same recipe). Only families with >= 2 seeds on the legacy simulator (no realsim / pygidSIM-position variants) take part.
routed - global > 0 on B means routing generalises; oracle - global shows how much of the apparent gain is selection on noise."""
import argparse, csv, glob, os, pickle, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from heatmap import evaluation as E
from heatmap.region_sweep import ap_of, family_of, short

p = argparse.ArgumentParser()
p.add_argument('--dets', required=True); p.add_argument('--regions', required=True); p.add_argument('--k', type=int, default=6); p.add_argument('--out', required=True)
p.add_argument('--repeats', type=int, default=20); p.add_argument('--min_frames', type=int, default=6); p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
cfg = E.make_config(E.DATASETS['organic'])
reg = {(r['set'], int(r['frame'])): int(r['region']) for r in csv.DictReader(open(f'{a.regions}/regions_k{a.k}.csv')) if r['set'] in ('organic', '41')}
models = collections.defaultdict(list)                      # family -> [(dets, gts) per seed]
keys = None
for f in sorted(glob.glob(f'{a.dets}/*.pkl')):
    run = os.path.basename(f)[:-4]; fam, _ = family_of(run)
    if 'realsim' in fam or 'PygidPos' in fam or fam.startswith('dino'):
        continue
    d = pickle.load(open(f, 'rb'))['sets']
    ks = [(s, i) for s in ('organic', '41') for i in range(len(d[s]))]
    keys = keys or ks
    dets = [(np.asarray(fr['boxes'], np.float32).reshape(-1, 4), np.asarray(fr['scores'], np.float32)) for s in ('organic', '41') for fr in d[s]]
    gts = [dict(gt=fr['gt'], gtconf=fr['gtconf'], mask=fr['mask']) for s in ('organic', '41') for fr in d[s]]
    models[fam].append((dets, gts))
models = {f: v for f, v in models.items() if len(v) >= 2}
fams = sorted(models)
cnt = collections.Counter(reg[k] for k in keys if k in reg)
regions = sorted(r for r, n in cnt.items() if n >= a.min_frames)
members = {r: [i for i, k in enumerate(keys) if reg.get(k) == r] for r in regions}
print(len(fams), 'families,', sum(len(v) for v in models.values()), 'models, regions', {r: len(members[r]) for r in regions}, flush=True)
rng = np.random.default_rng(a.seed)


def fam_ap(f, idx):
    v = [ap_of([m[0][i] for i in idx], [m[1][i] for i in idx], cfg)[0] for m in models[f]]
    return float(np.nanmean(v))


res = []
for rep in range(a.repeats):
    A, B = {}, {}
    for r in regions:
        m = rng.permutation(members[r]); h = len(m) // 2; A[r], B[r] = list(m[:h]), list(m[h:])
    apA = {(f, r): fam_ap(f, A[r]) for f in fams for r in regions}
    apB = {(f, r): fam_ap(f, B[r]) for f in fams for r in regions}
    wA = np.array([len(A[r]) for r in regions], float); wB = np.array([len(B[r]) for r in regions], float)
    glob_f = max(fams, key=lambda f: np.dot(wA, [apA[(f, r)] for r in regions]))
    score = lambda pick: float(np.dot(wB, [apB[(pick(r), r)] for r in regions]) / wB.sum())
    routed = {r: max(fams, key=lambda f: apA[(f, r)]) for r in regions}
    s_glob = score(lambda r: glob_f); s_rout = score(lambda r: routed[r]); s_orac = score(lambda r: max(fams, key=lambda f: apB[(f, r)]))
    best_B_global = max(np.dot(wB, [apB[(f, r)] for r in regions]) / wB.sum() for f in fams)
    res.append((s_glob, s_rout, s_orac, glob_f, [routed[r] for r in regions], best_B_global))
    print(f'repeat {rep + 1}/{a.repeats}: global {s_glob:.4f} routed {s_rout:.4f} oracle {s_orac:.4f}', flush=True)
g = np.array([r[0] for r in res]); ro = np.array([r[1] for r in res]); o = np.array([r[2] for r in res])
gain = ro - g
lines = [f'# Routing by region, cross-validated over frames (k = {a.k})', '',
         f'{len(fams)} families (>= 2 seeds, legacy simulator), regions {regions} with {[len(members[r]) for r in regions]} labelled frames, {a.repeats} random half splits.', '',
         f'| score on the held-out half (frame-weighted mean of region APs) | mean | sd over splits |', '---|---|---',
         f'| global: best single family chosen on the other half | {g.mean():.4f} | {g.std():.4f} |', f'| routed: best family per region chosen on the other half | {ro.mean():.4f} | {ro.std():.4f} |',
         f'| oracle: best family per region chosen ON the scored half (not reachable) | {o.mean():.4f} | {o.std():.4f} |', '',
         f'routed - global: mean {gain.mean():+.4f}, sd {gain.std():.4f}, routed better in {(gain > 0).mean():.0%} of the splits.',
         f'oracle - global: mean {(o - g).mean():+.4f} (the apparent gain that selection on the same frames would claim; the part above routed - global is selection on noise).', '',
         'Families picked as global best: ' + ', '.join(f'{short(f)} {n}x' for f, n in collections.Counter(r[3] for r in res).most_common(4)), '']
for j, r in enumerate(regions):
    lines.append(f'Region {r} ({len(members[r])} frames) routed to: ' + ', '.join(f'{short(f)} {n}x' for f, n in collections.Counter(x[4][j] for x in res).most_common(3)))
open(a.out, 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
