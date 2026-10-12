"""Where the run directories of hm_runs live: grouped in subfolders. stdlib only.

  run_dirs(runs_dir)      -> [(run name, path)] for every hm_* run directory at the top level or one level down in a group folder (works before AND after
                             the move, so summarize_runs.py / plot_all_curves.py / the queue logic find a run wherever it is)
  group_of(name)          -> the group folder a run belongs to
  python heatmap/runs_layout.py [--runs_dir ...] [--apply] [--undo]
      dry run by default: prints old -> new for every run that would move. --apply moves them and writes <runs_dir>/_tools/moved_runs.tsv;
      --undo moves everything listed in that file back. A directory that a running train.py writes to (its --out) is never moved.
"""
import os, sys, argparse, subprocess, shutil

RUNS = '/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs'
SKIP = {'curves', 'plots', 'q_threshold', 'score_sweeps'}          # derived results, stay at the top level
GROUPS = ['frozen_simmim_plain', 'run_C_recipe', 'head_variants', 'channel_variants', 'backbone_variants', 'unfrozen', 'realsim']


def group_of(name):
    n = name
    if 'realsim' in n:
        return 'realsim'
    if n.lower().startswith('hm_simmim_unfrozen'):
        return 'unfrozen'
    if 'boxconv1' in n or 'random_frozen' in n:
        return 'backbone_variants'
    if 'long120' in n:
        return 'frozen_simmim_plain'
    if '_long_tf32' in n:
        return 'run_C_recipe'
    if any(k in n for k in ('_wide', 'stride1', 'ringhead', 'zeroinv')):
        return 'head_variants'
    if any(k in n for k in ('_chan', 'coord')):
        return 'channel_variants'
    return 'frozen_simmim_plain'


def run_dirs(runs_dir=RUNS):
    out = []
    for e in sorted(os.listdir(runs_dir)):
        p = os.path.join(runs_dir, e)
        if not os.path.isdir(p):
            continue
        if e.startswith('hm_'):
            out.append((e, p))
        elif e in GROUPS:
            out += [(r, os.path.join(p, r)) for r in sorted(os.listdir(p)) if r.startswith('hm_') and os.path.isdir(os.path.join(p, r))]
    return out


def running_outs():
    try:
        ps = subprocess.run(['pgrep', '-af', 'heatmap/train.py'], capture_output=True, text=True).stdout.splitlines()
    except Exception:
        return set()
    outs = set()
    for l in ps:
        t = l.split()
        if '--out' in t:
            outs.add(os.path.realpath(t[t.index('--out') + 1]))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', default=RUNS); ap.add_argument('--apply', action='store_true'); ap.add_argument('--undo', action='store_true')
    ap.add_argument('--skip', nargs='*', default=[], help='run names (or substrings) that must not move, e.g. a run whose queue is still doing its post steps')
    a = ap.parse_args()
    man = os.path.join(a.runs_dir, '_tools', 'moved_runs.tsv')
    if a.undo:
        for l in open(man).read().splitlines():
            old, new = l.split('\t')
            if os.path.exists(new) and not os.path.exists(old):
                shutil.move(new, old); print('undo', new, '->', old)
        return
    busy = running_outs()
    moves = []
    for e in sorted(os.listdir(a.runs_dir)):
        p = os.path.join(a.runs_dir, e)
        if e.startswith('hm_') and os.path.isdir(p):
            moves.append((p, os.path.join(a.runs_dir, group_of(e), e)))
        elif e in ('_smoke', '_smoke_coord', '_interim'):
            moves.append((p, os.path.join(a.runs_dir, '_scratch', e)))
    for old, new in moves:
        skip = os.path.realpath(old) in busy or any(k in os.path.basename(old) for k in a.skip)
        print(('SKIP (a train.py is writing to it) ' if skip else ('MOVE ' if a.apply else 'would move ')) + f'{os.path.relpath(old, a.runs_dir)} -> {os.path.relpath(new, a.runs_dir)}')
    if not a.apply:
        print('dry run; add --apply to move'); return
    with open(man, 'a') as mf:
        for old, new in moves:
            if os.path.realpath(old) in busy or any(k in os.path.basename(old) for k in a.skip) or os.path.exists(new):
                continue
            os.makedirs(os.path.dirname(new), exist_ok=True)
            shutil.move(old, new); mf.write(f'{old}\t{new}\n')


if __name__ == '__main__':
    main()
