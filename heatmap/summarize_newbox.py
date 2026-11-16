"""Regenerates hm_runs/newbox/results_table_newbox.md and hm_runs/newbox/curves/ from the run directories (no hand-edited numbers).

  python heatmap/summarize_newbox.py [--runs_dir /mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs]

NEW convention only: box = centre +- 1 sigma (--box_coef 2,2 and variants), evaluation on the visibility-fixed files in newbox_visfix.
  (a) runs retrained on the new files: every hm_* folder directly in <runs_dir>/newbox/ that has an evaluate_final.txt
  (b) 'old checkpoint on new files': every folder in <runs_dir>/newbox/old_on_new/ (evaluate.py + score_sweep.py of an old-convention
      checkpoint on the new files), with the old AP of the same checkpoint read from its old run folder (found via runs_layout.run_dirs)
  (c) how to read: the numbers in the text are computed here from (a) and (b).
results_table.md (summarize_runs.py) stays old-convention only; runs_layout.run_dirs does not scan newbox/, so the two never mix.
Curves: the newbox runs are linked into a scratch folder and plot_all_curves.py is run on it; the result is copied to newbox/curves/.
"""
import os, re, sys, json, argparse, statistics as st, shutil, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runs_layout import run_dirs  # noqa: E402

RUNS = '/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs'
MD5 = {'organic_labeled.h5': ('organics_visfix.h5', '39ae5381ea86cbe5763d44e5a291f325', '8 frames / 817 GT'),
       '41.h5': ('perovskites_nan_wedge_visfix.h5', '4b7d168c2170e2f8d9e5e85645950805', '41 frames / 1678 GT')}
PROTO = '+nms (class-aware NMS), score floor 0.1, top-225, QMatcher min_iou 0.1'


def parse_eval(path, ds):
    """+nms numbers of dataset ds ('organic' / '41') from an evaluate_final.txt: AP, and score>0.3 recall/prec/ring/FP."""
    t = open(path).read()
    i = t.index(f'[hm+nms] {ds}:'); seg = t[i:i + 1500]
    d = {'ap': float(re.search(r'ap_total ([0-9.]+)', seg).group(1))}
    m = re.search(r'score>0\.3: recall ([0-9.]+) \(seg ([0-9.]+), ring ([0-9.]+)\) precision ([0-9.]+) preds (\d+) FP (\d+)', seg)
    if m:
        d.update(rec=float(m.group(1)), ring=float(m.group(3)), prec=float(m.group(4)), fp=int(m.group(6)))
    return d


def sweep_floor(path, ds, floor, topk=225):
    """AP at a score floor from a score_sweep.log (decode nms, top-K topk); None if absent."""
    if not os.path.exists(path):
        return None
    t = open(path).read()
    m = re.search(rf'=== {re.escape(ds)} \([^)]*\) \| decode nms \| top-K {topk} ===(.*?)(?:\n===|\Z)', t, re.S)
    if not m:
        return None
    m2 = re.search(rf'floor {floor}\s+: AP ([0-9.]+)', m.group(1))
    return float(m2.group(1)) if m2 else None


def train_info(rdir):
    a = {}
    try:
        a = json.load(open(os.path.join(rdir, 'args.json')))
    except Exception:
        pass
    log = os.path.join(rdir, 'train.log')
    spe, last = None, None
    if os.path.exists(log):
        t = open(log).read()
        v = [int(x) for x in re.findall(r'^\[epoch \d+\] loss .*\((\d+)s\)\s*$', t, re.M)]
        spe = st.median(v) if v else None
        e = re.findall(r'^\[epoch (\d+)\] loss', t, re.M)
        last = int(e[-1]) if e else None
    return a, spe, last


def f4(x):
    return '-' if x is None else f'{x:.4f}'


def f3(x):
    return '-' if x is None else f'{x:.3f}'


def label(a, name):
    kind = 'run C recipe (bs 8 x 125, lr 4.2e-4, drops 90/112)' if a.get('lr_drops') else 'plain H1 recipe (bs 4 x 250, lr 3e-4, drop 45)'
    return kind


def parse_train_final(log, epochs):
    """-> (organic, 41) dicts from the last-epoch '+nms (deployed)' lines of train.log (in-training eval, scaled GT via HM_GT_BOX_COEF=auto), or None.
    Used only where the final evaluate_final_gtscaled.txt is empty (Galvani lustre quota); the checkpoint of those runs is intact."""
    if not os.path.exists(log) or not epochs:
        return None
    t = open(log, errors='replace').read(); out = []
    for ds in ('organic', '41'):
        m = re.search(rf'^\[epoch {epochs - 1}\] {ds} \+nms \(deployed\): AP ([0-9.]+) \| score>0\.3 recall ([0-9.]+) prec ([0-9.]+) ring recall ([0-9.]+) FP (\d+)', t, re.M)
        if not m:
            return None
        out.append(dict(ap=float(m.group(1)), rec=float(m.group(2)), prec=float(m.group(3)), ring=float(m.group(4)), fp=int(m.group(5))))
    return out


def norm_bc(bc):
    """'2.0,2.0' (Galvani) and '2,2' (colorbox) are the same convention; other strings are kept"""
    try:
        v = [float(x) for x in str(bc).split(',')]
        return '2,2' if v == [2.0, 2.0] else str(bc)
    except ValueError:
        return str(bc)


def recipe(a):
    """H1 = plain recipe; C = run C recipe; extras for the queue-11 variants: head lr, 120 epochs"""
    if 'pretrain_runs' in str(a.get('bb_path', '')):   # DINOv3 weights pretrained further on our corpus: own group, not averaged with the public weights
        return f"{'unfrozen' if a.get('unfreeze') else 'frozen'} {a['bb']}, corpus pretraining ({os.path.basename(os.path.dirname(a['bb_path']))})"
    if str(a.get('bb', '')).startswith('dinov3'):      # frozen public DINOv3 weights instead of the SimMIM swin, otherwise plain H1: own group
        return f"unfrozen {a['bb']} long, bb lr " + ('%.0e' % a.get('lr_backbone', 0)).replace('e-0', 'e-') if a.get('unfreeze') else f"frozen {a['bb']} (H1 recipe)"
    if a.get('unfreeze'):
        return 'unfrozen-long bb lr ' + ('%.0e' % a.get('lr_backbone', 0)).replace('e-0', 'e-')
    if a.get('lr_drops'):
        return 'C'
    if a.get('sim_pos_bank'):            # legacy simulator with ONLY the peak positions from a pygidSIM peak bank: own group, not part of the H1 baseline mean
        mode = a.get('sim_pos_mode', 'structure')     # structure = one random structure + fill (default); pooled / pooled_flat = i.i.d. draws from the pool of all peaks (flat = uniform density over the image)
        w = ' + widths' if a.get('sim_pos_widths') else ''          # --sim_pos_widths 1: the peak widths of the same bank peaks too
        return f'legacy sim + pygidSIM positions{w} (H1 recipe)' if mode == 'structure' else f'legacy sim + pygidSIM positions{w}, {mode} (H1 recipe)'
    if a.get('sim') == 'realbkg' and 'Gnome60k' in a.get('out', ''):    # E13a settings, only the bank changed (48,750 GNoME + 11,250 organic; the bank is an env var, not in args.json)
        return 'realsim, GNoME + organic bank' + (', flat spot positions' if any('spot_flat' in o for o in a.get('realsim_opt', [])) else '') + ' (H1 recipe)'
    if a.get('sim') == 'realbkg' and any('structure_frames' in o for o in a.get('realsim_opt', [])):    # old realsim renderer fed with the stored pygidSIM peak tables
        return 'realsim renderer + pygidSIM peak tables (H1 recipe)'
    if a.get('sim') == 'realsim_v2':     # realsim_v2 (GNoME structures, pygidSIM), H1 recipe: its own group, not part of the H1 baseline mean
        return 'realsim_v2 (H1 recipe)'
    if a.get('sim') == 'realbkg':        # real-background simulator (realsim), H1 recipe: its own group, not part of the H1 baseline mean
        if a.get('mix_plain'):
            return 'realsim + plain mix %.2f (H1 recipe)' % a['mix_plain']
        return 'realsim (H1 recipe)'
    r = 'H1'
    lr = a.get('lr')
    if lr is not None and abs(lr - 3e-4) > 1e-9:
        r += ' lr' + ('%.0e' % lr).replace('e-0', 'e-')
    if a.get('epochs') not in (None, 60):
        r += f" ep{a.get('epochs')}"
    return r


def epcell(x, ag):
    """Epochs column: a run stopped before its last epoch and scored from epoch<N>_checkpoint.pth shows 'N+1 of E'."""
    for f in os.listdir(os.path.join(RUNS, x['dir'])) if os.path.isdir(os.path.join(RUNS, x['dir'])) else []:
        m = re.match(r'epoch(\d+)_checkpoint\.pth$', f)
        if m and not os.path.exists(os.path.join(RUNS, x['dir'], 'final_checkpoint.pth')):
            return f"{int(m.group(1)) + 1} of {ag.get('epochs')} (stopped)"
    return ag.get('epochs', '-')


def short_run(name):
    """short Run-column name (the full folder name is in the Run dir column): hm_simmim_frozen_ridge_tf32_seed1_newbox_2.00_2.00 -> seed1_2.00_2.00"""
    return name.replace('hm_simmim_UNFROZEN_ridge_long_', 'UNFROZEN_').replace('hm_simmim_frozen_ridge_', '').replace('tf32_', '').replace('_newbox', '').replace('long_', 'long_')


def align(text):
    """pads the cells of every markdown table block to the widest cell of its column (readable as raw text)"""
    out, blk = [], []
    def flush():
        if not blk:
            return
        rows = [[c.strip() for c in l.strip().strip('|').split('|')] for l in blk]
        w = [max(len(r[i]) for r in rows if i < len(r)) for i in range(max(len(r) for r in rows))]
        for r in rows:
            if set(''.join(r)) <= set(':-'):
                out.append('|' + '|'.join((':' + '-' * (w[i] - 1) if c.startswith(':') and not c.endswith(':') else '-' * (w[i] - 1) + ':' if c.endswith(':') else '-' * w[i]).join(' ') if False else ' ' + (':' + '-' * (w[i] - 1) if c.startswith(':') else '-' * (w[i] - 1) + ':' if c.endswith(':') else '-' * w[i]) + ' ' for i, c in enumerate(r)) + '|')
            else:
                out.append('| ' + ' | '.join(c.ljust(w[i]) for i, c in enumerate(r)) + ' |')
        blk.clear()
    for l in text.split('\n'):
        if l.startswith('|'):
            blk.append(l)
        else:
            flush(); out.append(l)
    flush()
    return '\n'.join(out)


def method(name):
    return 'run C recipe' if '_long_' in name else 'plain tf32 recipe'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs_dir', default=RUNS); ap.add_argument('--no_curves', action='store_true')
    a = ap.parse_args()
    nb = os.path.join(a.runs_dir, 'newbox')
    rel = lambda p: os.path.relpath(p, a.runs_dir)
    rd = os.path.join(nb, 'runs')      # run folders live in newbox/runs/; table, curves, plots, old_on_new stay in newbox/
    runs = [(r, os.path.join(rd, r)) for r in sorted(os.listdir(rd)) if r.startswith('hm_') and os.path.isdir(os.path.join(rd, r))]
    def proxy_eval(d):
        gs = os.path.join(d, 'evaluate_final_gtscaled.txt')
        if os.path.exists(gs) and os.path.getsize(gs) == 0 and not os.path.exists(os.path.join(d, 'evaluate_final.txt')):
            return parse_train_final(os.path.join(d, 'train.log'), train_info(d)[0].get('epochs'))
        return None
    done, other = [], []
    for r, d in runs:
        (done if (os.path.exists(os.path.join(d, 'evaluate_final.txt')) or proxy_eval(d) is not None or '+nms] 41: ap_total' in (open(os.path.join(d, 'evaluate_final_gtscaled.txt')).read() if os.path.exists(os.path.join(d, 'evaluate_final_gtscaled.txt')) else '')) else other).append((r, d))
    recs = []
    for r, d in done:
        args, spe, last = train_info(d)
        px = proxy_eval(d)
        if px:      # final scoring file empty (Galvani lustre quota): last-epoch in-training evaluation, scaled GT
            args['box_coef'] = norm_bc(args.get('box_coef'))
            recs.append(dict(run=r, dir=rel(d), args=args, spe=spe, of=None, vf=None, o=px[0], v=px[1], gt='epoch-%d in-training (scaled)' % (args['epochs'] - 1), proxy=True, v02=None, o02=None))
            continue
        args['box_coef'] = norm_bc(args.get('box_coef'))
        ef = os.path.join(d, 'evaluate_final.txt') if os.path.exists(os.path.join(d, 'evaluate_final.txt')) else os.path.join(d, 'evaluate_final_gtscaled.txt')
        o, v = parse_eval(ef, 'organic'), parse_eval(ef, '41')
        sw = os.path.join(d, 'score_sweep.log')
        bc = str(args.get('box_coef', ''))
        gs = os.path.join(d, 'evaluate_final_gtscaled.txt')
        if os.path.exists(gs) and '+nms] 41: ap_total' in open(gs).read():
            os_, vs_, gt = parse_eval(gs, 'organic'), parse_eval(gs, '41'), 'scaled'
        elif bc == '2,2':
            os_, vs_, gt = o, v, 'scale 1 (2,2)'
        else:
            os_, vs_, gt = o, v, 'FIXED ONLY (scaled eval missing)'
        recs.append(dict(run=r, dir=rel(d), args=args, spe=spe, of=o, vf=v, o=os_, v=vs_, gt=gt, v02=sweep_floor(sw, '41', '0.2'), o02=sweep_floor(sw, 'organic', '0.2')))
    recs.sort(key=lambda x: (str(x['args'].get('box_coef')) != '2,2', x['args'].get('lr_drops') is not None, x['args'].get('box_coef', ''), x['args'].get('seed', 0)))

    L = ['# Heatmap detector, group "newbox": NEW box convention (box = centre +- 1 sigma) on the re-fitted validation files', '',
         'Generated by `heatmap/summarize_newbox.py` from the run folders in `hm_runs/newbox/` (no hand-edited numbers). `results_table.md` holds the OLD convention only; the two are never mixed.', '',
         '- **Training boxes**: `--box_coef a,w` = a_coef,w_coef in sigma units as FULL extent; `2,2` = box centre +- 1 sigma (new convention). Old convention: 2.80,1.30 with the old, wrongly scaled validation widths.',
         f'- **Data** (`HM_DATA_DIR=~/Documents/datasets/validationsSets/newbox_visfix`, visibility-fixed copies; clamp patch for infinite ring angle_width in `util/pygidloader.py`, overhanging boxes are kept): ' +
         '; '.join(f'`{k}` = {v[0]}, md5 {v[1]}, {v[2]}' for k, v in MD5.items()) + '. The raw uploads (wrong visibility) are not used.',
         f'- **Evaluation protocol** (all numbers): {PROTO}; in table (a) the AP, recall, precision and FP columns are scored against the GT scaled to the training box_coef of each run (GT scale (x,y) = (w_coef/2, a_coef/2), ring boxes in x only; `evaluate_final_gtscaled.txt`, code origin/heatmap d9b2d82/91946fc): this is the headline, since box size and GT size only make sense together; unscaled-GT numbers are not shown. Recall / precision / ring recall / FP use score > 0.3; score floor 0.1 only. Frozen SimMIM swin, ridge ring target, TF32, legacy simulator unless noted.', '',
         '## (a) Retrained on the new files', '',
         '| Run | Recipe | Seed | box_coef | Epochs | s/epoch | AP org | AP 41 | Recall org | Prec org | Ring org | FP org | Recall 41 | Prec 41 | Ring 41 | FP 41 |',
         '|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
    data = f'visfix {MD5["organic_labeled.h5"][1][:8]} / {MD5["41.h5"][1][:8]}'
    for x in recs:
        ag = x['args']
        L.append(f"| {short_run(x['run'])}{'*' if x['gt'].startswith('FIXED') else ''}{' †' if x.get('proxy') else ''} | {recipe(ag)} | {ag.get('seed', '-')} | {ag.get('box_coef', '-')} | {epcell(x, ag)} | {'-' if x['spe'] is None else int(x['spe'])} | "
                 f"{f4(x['o']['ap'])} | {f4(x['v']['ap'])} | {f3(x['o'].get('rec'))} | {f3(x['o'].get('prec'))} | {f3(x['o'].get('ring'))} | {x['o'].get('fp', '-')} | "
                 f"{f3(x['v'].get('rec'))} | {f3(x['v'].get('prec'))} | {f3(x['v'].get('ring'))} | {x['v'].get('fp', '-')} |")
    L += ['', '**Recipe**: H1 = plain recipe (bs 4 x 250, head lr 3e-4, lr drop at epoch 45, 60 epochs), `H1 lr<x>` = other head lr, `H1 ep120` = 120 epochs with lr drop at 90 (box shape a_coef,w_coef in the box_coef column); C = run C recipe (bs 8 x 125, lr 4.2e-4, drops 90/112, 120 epochs). **Run** = folder name without the prefix `hm_simmim_frozen_ridge_` and without `tf32_` / `newbox_`: the folder is `newbox/runs/hm_simmim_frozen_ridge_tf32_<run>` with `newbox_` before the box_coef (e.g. `seed1_2.00_2.00` = `newbox/runs/hm_simmim_frozen_ridge_tf32_seed1_newbox_2.00_2.00`; `_galvani` = trained on Galvani, same files under `newbox/`). Data: `newbox_visfix` (md5 39ae5381 / 4b7d168c). Scoring: GT scaled to the run\'s box_coef (scale 1 for 2,2); a `*` after a run name would mean that no scaled evaluation exists for it.']
    if not recs:
        L.append('| (no finished run yet) |' + ' |' * 15)
    # mean/spread of recipes that exist with several seeds at the same box_coef
    grp = {}
    for x in recs:
        grp.setdefault((recipe(x['args']), x['args'].get('box_coef')), []).append(x)
    gal = lambda y: y['run'].endswith('_galvani')
    def src(sel):
        return ', '.join(('G' if gal(y) else 'C') + str(y['args'].get('seed')) + ('†' if y.get('proxy') else '') for y in sel)
    def mstd(sel, sub, key, fmt):
        vals = [y[sub][key] for y in sel if key in y[sub]]
        if not vals:
            return '-'
        return fmt(st.mean(vals)) + (f' ± {fmt(st.stdev(vals))}' if len(vals) > 1 else '')
    base = grp.get(('H1', '3.0,3.0'), [])
    bo = st.mean(y['o']['ap'] for y in base) if base else None
    bv = st.mean(y['v']['ap'] for y in base) if base else None
    dlt = lambda sel, sub, b: '-' if b is None else f"{st.mean(y[sub]['ap'] for y in sel) - b:+.4f}"
    mrows, gsel = [], []
    for (lab, bc), v in sorted(grp.items(), key=lambda kv: (kv[0][0] != 'H1', float(str(kv[0][1]).split(',')[0]), str(kv[0][1]))):
        # where Galvani replicates exist the core group is Galvani seeds + colorbox seed 42; colorbox seed-1/2 runs are separate rows and enter only the "all runs" row
        core = [y for y in v if gal(y) or y['args'].get('seed') == 42] if any(gal(y) for y in v) else v
        for sel, tag in ((core, ''), (v, ' (all runs)')):
            if not sel or (tag and len(core) == len(v)): continue
            gsel.append((f"{bc}{tag} {lab}", len(sel), src(sel), st.mean(y['o']['ap'] for y in sel), st.mean(y['v']['ap'] for y in sel)))
            mrows.append(f"| {bc}{tag} | {lab} | {len(sel)} | {src(sel)} | {mstd(sel, 'o', 'ap', f4)} | {mstd(sel, 'v', 'ap', f4)} | {mstd(sel, 'o', 'rec', f3)} | {mstd(sel, 'o', 'prec', f3)} | {mstd(sel, 'o', 'ring', f3)} | {mstd(sel, 'o', 'fp', lambda t: f'{t:.0f}')} | "
                         f"{mstd(sel, 'v', 'rec', f3)} | {mstd(sel, 'v', 'prec', f3)} | {mstd(sel, 'v', 'ring', f3)} | {mstd(sel, 'v', 'fp', lambda t: f'{t:.0f}')} | {dlt(sel, 'o', bo)} | {dlt(sel, 'v', bv)} |")
    glines = []
    if mrows:
        glines = ['', '## (b) Mean ± SD per box_coef', '',
                  'Mean ± sample SD (n-1) over the runs of each box_coef, scaled-GT numbers as in (a); n=1 rows have no SD. The last two columns are the difference of the group mean to the mean of all H1 runs at 3.0,3.0 (the baseline; all sources). Sources: G = Galvani, C = colorbox1, followed by the seed. Where Galvani replicates exist the core group is Galvani seeds 1, 2 + colorbox seed 42; the colorbox seed-1/2 runs (queue 10) are separate rows in (a) and enter only the "all runs" row.', '',
                  '| box_coef | Recipe | n | Runs (source+seed) | AP org | AP 41 | Recall org | Prec org | Ring org | FP org | Recall 41 | Prec 41 | Ring 41 | FP 41 | d AP org vs 3.0,3.0 H1 | d AP 41 vs 3.0,3.0 H1 |',
                  '|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|'] + mrows
    L += glines
    if recs:
        def top(items, key, k=3):
            return sorted(items, key=key, reverse=True)[:k]
        W_O, W_V = 817, 1678      # GT boxes of organic_labeled.h5 / 41.h5 (see Data line); combined = GT-count-weighted mean of the two APs
        comb = lambda o, v: (W_O * o + W_V * v) / (W_O + W_V)
        rl = lambda x: f"{short_run(x['run'])}{' †' if x.get('proxy') else ''} ({recipe(x['args'])}, {x['args'].get('box_coef')})"
        cell = lambda name, o, v: f"{name}: org {f4(o)} / 41 {f4(v)} / combined {f4(comb(o, v))}"
        L += ['', '## (b2) Best results (scaled GT, +nms; AP only)', '',
              f'Ranked by AP. **combined** = GT-count-weighted mean of AP org and AP 41: ({W_O} x AP org + {W_V} x AP 41) / {W_O + W_V} (weights {W_O / (W_O + W_V):.3f} / {W_V / (W_O + W_V):.3f}; organic 8 frames / {W_O} GT, 41 set 41 frames / {W_V} GT); not a pooled AP. Every entry shows org / 41 / combined. Top 3 each; † = epoch-119 in-training values (not final scorings). Single runs are one seed, so differences of a few 0.01 (organic) / 0.005 (41) are within the seed spread; the group means (n>=2) are the more reliable ranking.', '',
              '**Single runs**', '', '| Rank | Best AP org | Best AP 41 | Best combined |', '|:--|:--|:--|:--|']
        bo_ = top(recs, lambda x: x['o']['ap']); bv_ = top(recs, lambda x: x['v']['ap']); bc_ = top(recs, lambda x: comb(x['o']['ap'], x['v']['ap']))
        for i in range(len(bo_)):
            L.append('| ' + str(i + 1) + ' | ' + ' | '.join(cell(rl(y[i]), y[i]['o']['ap'], y[i]['v']['ap']) for y in (bo_, bv_, bc_)) + ' |')
        gg = [g for g in gsel if g[1] >= 2]
        L += ['', '**Group means (n>=2, table (b))**', '', '| Rank | Best AP org | Best AP 41 | Best combined |', '|:--|:--|:--|:--|']
        go_ = top(gg, lambda g: g[3]); gv_ = top(gg, lambda g: g[4]); gc_ = top(gg, lambda g: comb(g[3], g[4]))
        for i in range(len(go_)):
            L.append('| ' + str(i + 1) + ' | ' + ' | '.join(cell(f"{y[i][0]} (n={y[i][1]}, {y[i][2]})", y[i][3], y[i][4]) for y in (go_, gv_, gc_)) + ' |')
    if other:
        L += ['', '**Folders without evaluate_final.txt** (running, failed or skipped; see `hm_runs/_tools/queue7_status.txt` / `queue8_status.txt` / `queue9_status.txt`):', '']
        for r, d in other:
            _, _, last = train_info(d)
            L.append(f'- `{rel(d)}`: last training epoch in train.log = {last if last is not None else "none"}')
    notes = []
    for q in ('queue7_status.txt', 'queue8_status.txt'):
        p = os.path.join(a.runs_dir, '_tools', q)
        if os.path.exists(p):
            notes += [f'{q}: {l}' for l in open(p).read().splitlines() if re.search(r'SKIP [a-z]*\s*hm_.*would not|FAILED|failed, continuing', l)]
    if notes:
        L += ['', '**Queue notes (failures / skipped runs)**:', ''] + [f'- {n}' for n in notes]

    # (b) how to read (scaled GT only; the old checkpoints on the new files were scored against the fixed GT only and are not shown)
    L += ['', '## (c) How to read this table', '',
          '- Every AP, recall, precision, ring recall and FP number is scored against the GT scaled to the training box convention of the run (GT scale (x,y) = (w_coef/2, a_coef/2)); box_coef 2,2 has scale 1. Scoring a larger training box against the unscaled +-1 sigma GT is not meaningful and is not shown anywhere in this file.',
          '- Old-convention APs (`old_convention/results_table.md`) are NOT comparable with the APs here: the validation boxes were re-fitted (stored widths are now 2 sigma, drawn box = centre +- sigma) and the visibility was fixed. Compare only within this file.',
          '- The old-convention checkpoints (box_coef 2.80,1.30) on the new files (`newbox/old_on_new/`) were only scored against the unscaled GT; they are therefore not tabulated. Re-scoring with their scaled GT (HM_GT_BOX_COEF=2.80,1.30) is pending.',
          '- Rows with the suffix `_galvani` were trained and scored on the Galvani cluster (same recipe, files copied from `origin/heatmap:heatmap/galvani_results/`); the others on colorbox1. s/epoch of the Galvani rows is cluster hardware and not comparable.',
          '- Recipe `unfrozen-long bb lr x` = swin backbone unfrozen (grad checkpointing), run C recipe (bs 8 x 125, head lr 4.2e-4, lr drops 90/112, 120 epochs), backbone lr x; all trained on Galvani (`_galvani`), box_coef 3.0,3.0. Their delta columns compare with the frozen H1 3.0,3.0 baseline (different recipe).',
          '- **†** = the final scoring file of that run (`evaluate_final_gtscaled.txt`) is EMPTY (Galvani lustre quota, checkpoint intact); the numbers are the last-epoch (epoch 119) in-training `+nms (deployed)` evaluation from `train.log` (same scaled GT via `HM_GT_BOX_COEF=auto`, same protocol, same checkpoint weights). Cross-check on the one unfrozen run that has both (seed 1, bb lr 1e-5): organic identical (AP 0.6504 both), 41 AP 0.7227 in-training vs 0.7224 final scoring. Update 2026-10-29: the two † rows (seed 2 bb 1e-5, seed 1 bb 4e-5) were re-scored on Galvani (final checkpoint, `evaluate_final_gtscaled.txt`) and now carry the final-scoring values; no † row remains unless the symbol appears again in a future run.',
          '- organic has 8 frames / 817 GT: epoch-to-epoch AP varies by several 0.01 during training, so only the final-checkpoint numbers (`evaluate_final_gtscaled.txt`) are used; seed SDs are 0.003-0.016, so single-seed differences below ~0.01-0.02 are noise.',
          '', '## (d) Failed / incomplete attempts (NOT in any mean) and setup notes (as reported by the Galvani session)', '',
          '- **Galvani lustre group-quota incidents (runs lost 2026-10-24/25)**: writes failed, so some files are empty or runs crashed. Consequences: (i) two unfrozen rows had an empty `evaluate_final_gtscaled.txt` (checkpoint intact), temporarily shown with epoch-119 in-training values (†); re-scored on 2026-10-29, now final values; (ii) the crashed attempts below.',
          '- FAILED (3 times, lustre quota): unfrozen-swin seed 2, backbone lr 4e-5 (3.0,3.0, Galvani): job 2999317 crashed at epoch ~115 (truncated checkpoint); rerun 3000585 (2026-10-25 05:30) and rerun 3002608 (2026-10-25 08:28) both crashed in epoch 1; no result, not tabulated.',
          '- FAILED: unfrozen-swin seed 3, backbone lr 1e-5 / 4e-5 (3.0,3.0, Galvani): jobs 3002606 / 3002607 crashed in epoch 1 (2026-10-25, lustre quota). All three configs were resubmitted on 2026-10-26 as jobs 3004985 (seed 3 bb1e-5), 3004986 (seed 3 bb4e-5) and 3004987 (seed 2 bb4e-5); all three FAILED on 2026-10-26 after about 4 min (crash in epoch 1 when saving the first checkpoint, lustre group quota full), so they never trained and no result exists (Galvani session, 2026-10-29 10:32). Nothing is pending on Galvani. The three configs were submitted again on 2026-10-29 10:48 as jobs 3020571 (seed 3 bb1e-5), 3020572 (seed 3 bb4e-5), 3020573 (seed 2 bb4e-5); all three COMPLETED rc=0 (about 6 h each, finished about 16:50) and are tabulated as `_galvani` rows (origin/heatmap f9846ec).',
          '- Recipe `realsim (H1 recipe)` = the H1 recipe (60 epochs, head lr 3e-4, lr drop 45) trained on the real-background simulator (`--sim realbkg`, preset r6) instead of the legacy simulator; it is its own group and NOT part of the H1 baseline mean. `realsimE9_seed42_3.0_3.0` = composition of Galvani E9 (E6 settings + `mask_wedge_only_frac=0.2`, `mask_keep=[0,3,6,7]`, no edge fix) at box_coef 3,3, one sim worker, seed 42. Epochs `57 of 60 (stopped)`: attempt 3 was killed by the RAM watchdog during epoch 57 (another project\'s processes took RAM); the user decided to score the epoch-56 checkpoint (`epoch56_checkpoint.pth`, lr already dropped at epoch 45) instead of resuming. The in-training +nms value of epoch 56 (41: 0.6169) differs from the scoring file (41: 0.6134) as for E1 earlier (reason not examined). Diagnosis files: `_scratch/e9_diag/`. Recipe `realsim_v2 (H1 recipe)` = the H1 recipe trained on the new simulator realsim_v2 (`--sim realsim_v2`, code `realsim_v2/`): 60,000 images (one GNoME structure per image, pygidSIM spots/rings, 240 shards in `realsim_v2_data/v2_60k`, generation 2026-11-02 13:29 to 2026-11-03 00:52, 0 failures), generation settings min_snr 1.1, iomin_sr 0.5, mode probabilities oriented/both/rings 0.25/0.5/0.25; training: random vertical and horizontal flips (probability 0.5 each), no plain mix, deployed contrast, boxes at 3.0,3.0 rebuilt from the stored sigmas, about 110 s/epoch, run 2026-11-03 03:08 to 05:02. `realsim_v2flip_seed42_3.0_3.0` is that run. `realsimE9flip_seed42_3.0_3.0` = the same E9 composition plus `--sim_flip 1` (random vertical and horizontal flip, probability 0.5 each, as the legacy simulator); started 2026-11-02 11:56, stopped by the user after epoch 23 (no clear improvement), resumed from `checkpoint.pth` at epoch 24 on 2026-11-03 00:51 and finished epoch 59 at 03:08 (the sim random stream restarted at the resume). Recipe `legacy sim + pygidSIM positions (H1 recipe)` = the legacy simulator (plain H1 image creation, in-process) with ONLY the peak positions taken from pygidSIM (`--sim_pos_bank bank_v3peaks_v3only.npz`, `SimulationConfig.pos_bank`): per frame and kind the legacy count n is kept, the n positions are a random subset of one random v3 structure (move_fromMW=True) filled up from further structures (on average 8 structures per kind and frame), x = q/q_max x 1024 with q_max drawn from 3.6-4.0, y = chi/90 x 512; widths, intensities, NMS, gap / dark-area filters, background and noise are the legacy code (identical output with the option off, checked on 12 frames); run `legacyPygidPos_seed42_3.0_3.0`, 2026-11-04 11:38 to 13:40, about 120 s/epoch. Recipe `realsim renderer + pygidSIM peak tables (H1 recipe)`: the old realsim renderer fed with peak tables of the pygidSIM shards (`realsimV1v2peaks`, stopped by the user after epoch 24; `realsimV1v3peaks` with the v3 shards plus organic filler was killed four times by the RAM watchdog on 2026-11-04 11:04-11:08 at epoch 44 and abandoned: no scoring file, in-training +nms of epoch 44 organic 0.5622 / 41 0.5644, best in-training organic 0.6053 and 41 0.6393).',
          '- DINO baseline A (SEPARATE DINO pipeline, not part of this heatmap table) was trained at +-1 sigma, i.e. box_coef 2,2, so it is consistent with the stored GT (no GT scaling needed). A DINO run at the 3.0-equivalent box convention is deferred until the heatmap settings are chosen. (Setup note taken from the Galvani session; no DINO number is tabulated here.)']
    txt = align('\n'.join(L) + '\n')
    open(os.path.join(nb, 'results_table_newbox.md'), 'w').write(txt)
    print(txt)

    if not a.no_curves and runs:
        view = os.path.join(a.runs_dir, '_scratch', 'newbox_view')
        shutil.rmtree(view, ignore_errors=True); os.makedirs(view)
        for r, d in done:
            os.symlink(d, os.path.join(view, r))
        pc = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_all_curves.py')
        subprocess.run([sys.executable, pc, '--runs_dir', view, '--out', os.path.join(view, 'curves')], check=False, stdout=subprocess.DEVNULL)
        out = os.path.join(nb, 'curves')
        shutil.rmtree(out, ignore_errors=True)
        if os.path.isdir(os.path.join(view, 'curves')):
            shutil.copytree(os.path.join(view, 'curves'), out)
            print(f'curves -> {out}', file=sys.stderr)


if __name__ == '__main__':
    main()
