"""Per-epoch curves of ALL heatmap runs, organised in one folder (matplotlib, headless, no GPU, no torch).

  python heatmap/plot_all_curves.py [--runs_dir /mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs] [--out <runs_dir>/curves]

Reads, per run dir hm_*: exp_ap_organic.txt / exp_ap_41.txt (written at every evaluation epoch during training; the
"+nms (deployed)" AP is the `apnms` column, only present in newer logs) and train.log (training loss per epoch).
Writes into --out:
  README.txt                          what each file is
  curves_data.csv                     every number plotted (run, set, epoch, AP native, AP +nms, recall/precision @0.3, close-pair recalls)
  overview_apnms.png                  ONE figure, two panels (organic, 41): +nms AP vs epoch, all runs overlaid, DINO references dashed
                                      (runs whose log has no +nms column are listed in a note, they are in overview_native.png)
  overview_native.png                 same with the NATIVE AP (peak picking, no NMS), which every run logged
  overview_small_multiples.png        one small panel per run: +nms AP solid, native AP dashed, organic + 41, same y axis (compare shapes here)
  per_run/<tag>.png                   per run: +nms AP, native AP, recall@0.3, precision@0.3, close-pair (NN<5px) recall, training loss
Honest notes: curves are single seeds and the evaluation noise between neighbouring evaluations is ~0.03 AP before the learning-rate
drop; +nms AP is missing (gap in the line) for evaluations logged before that column existed; runs still training are marked (running).
DINO reference lines are the final +nms numbers scored by the same code (ssl1 0.568/0.744, lr 4e-5 0.622/0.763), not per-epoch curves."""
import argparse, csv, math, os, re, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_runs import read_eval, read_loss, style, SURFACE, INK, INK2, GRID, REF       # noqa: E402
from summarize_runs import NAME                                                      # noqa: E402

ORG, V41 = '#2a78d6', '#eb6834'
EXTRA_NAME = {'ridge_chanhemask_tf32': 'chanhemask: HE, mask (stopped at epoch 2)'}
STOPPED = {'ridge_chanhemask_tf32'}          # stopped by the user after epoch 2: kept in the CSV, left out of the figures
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#000000', '#7f7f7f']
SETS = [('organic', ORG), ('41', V41)]
REFS = {'organic': [('ssl1 0.568', 0.568), ('lr 4e-5 0.622', 0.622)], '41': [('ssl1 0.744', 0.744), ('lr 4e-5 0.763', 0.763)]}
COLNAMES = ['ap_native', 'recall_0.3', 'precision_0.3', 'recall_chigap_lt5', 'recall_nn_lt5', 'ap_nms']


def tag_of(run):
    t = run.replace('hm_simmim_frozen_', '').replace('_2.80_1.30', '')
    return t if t in NAME else (run if run in NAME else t)


def title_of(run):
    return EXTRA_NAME.get(tag_of(run), NAME.get(tag_of(run), tag_of(run)))


def drops_of(run):
    return (90, 112) if 'long' in run else (45,)


def load(runs_dir):
    runs = []
    for run in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, run)
        if not (run.startswith('hm_') and os.path.isdir(d)):
            continue
        ev = {ds: read_eval(os.path.join(d, f'exp_ap_{ds}.txt')) for ds, _ in SETS}
        if not any(ev.values()):
            continue
        done = os.path.exists(os.path.join(d, 'evaluate_final.txt'))
        st = 'stopped' if tag_of(run) in STOPPED else ('finished' if done else 'running')
        runs.append(dict(run=run, dir=d, ev=ev, loss=read_loss(os.path.join(d, 'train.log')), done=done, status=st))
    # reading order of summarize_runs.LABELS, unknown runs last
    order = {k: i for i, k in enumerate(NAME)}
    runs.sort(key=lambda r: order.get(tag_of(r['run']), 99))
    return runs


def series(ev, ci):
    ep = list(ev)
    return ep, [ev[e][ci] for e in ep]


def finite(ep, ys):
    pts = [(e, y) for e, y in zip(ep, ys) if y is not None and not math.isnan(y)]
    return [p[0] for p in pts], [p[1] for p in pts]


def write_csv(runs, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['run', 'name', 'set', 'epoch'] + COLNAMES)
        for r in runs:
            for ds, _ in SETS:
                for e, v in r['ev'][ds].items():
                    w.writerow([r['run'], title_of(r['run']), ds, e] + [('' if math.isnan(x) else f'{x:.4f}') for x in v])


def suffix(r):
    return '' if r['status'] == 'finished' else f' ({r["status"]})'


def last_epoch(r):
    return max((max(v) for v in r['ev'].values() if v), default=0)


def overview(runs, path, ci, what):
    runs = [r for r in runs if r['status'] != 'stopped']
    has = [r for r in runs if any(finite(*series(r['ev'][ds], ci))[0] for ds, _ in SETS)]
    missing = [title_of(r['run']) for r in runs if r not in has]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=SURFACE, sharey=True)
    for si, (ds, _) in enumerate(SETS):
        ax = axes[si]; style(ax)
        if ci == 5:
            for lab, v in REFS[ds]:
                ax.axhline(v, color=REF, ls='--', lw=1)
                ax.text(0.995, v + 0.004, 'DINO ' + lab, fontsize=8, color=INK2, ha='right', transform=ax.get_yaxis_transform())
        for ri, r in enumerate(has):
            ep, ys = finite(*series(r['ev'][ds], ci))
            if ep:
                ax.plot(ep, ys, color=PALETTE[ri % len(PALETTE)], lw=1.6, label=title_of(r['run']) + suffix(r))
        ax.set_ylim(0.3, 0.8); ax.set_xlabel('epoch', color=INK2)
        ax.set_title(f'{ds}: AP, {what}', loc='left', color=INK, fontsize=11)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].set_ylabel('AP', color=INK2)
    h, l = axes[1].get_legend_handles_labels()
    if h:
        axes[1].legend(h, l, frameon=False, fontsize=8, loc='lower right', labelcolor=INK)
    note = ('' if not missing else 'Not drawn (older logs without this column; see overview_native.png): ' + ', '.join(missing) + '.   ')
    fig.suptitle(f'Heatmap detector: AP ({what}) per evaluation epoch, all runs (single seeds; evaluation noise ~0.03 before the lr drop)',
                 x=0.01, ha='left', color=INK, fontsize=12)
    fig.text(0.01, 0.012, note + 'Dashed grey = final DINO +nms numbers (ssl1, lr 4e-5), not per-epoch curves.', fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(path, dpi=130, facecolor=SURFACE); plt.close(fig)


def small_multiples(runs, path):
    runs = [r for r in runs if r['status'] != 'stopped']
    n = len(runs); cols = 4; rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.6 * rows), facecolor=SURFACE, sharey=True, squeeze=False)
    for k, r in enumerate(runs):
        ax = axes[k // cols][k % cols]; style(ax)
        has_nms = False
        for ds, col in SETS:
            for lab, v in REFS[ds][1:]:
                ax.axhline(v, color=col, ls=':', lw=0.8, alpha=0.6)
            ep, ys = finite(*series(r['ev'][ds], 0))
            if ep:
                ax.plot(ep, ys, color=col, lw=1.2, ls='--', alpha=0.55)
            ep, ys = finite(*series(r['ev'][ds], 5))
            if ep:
                has_nms = True
                ax.plot(ep, ys, color=col, lw=1.8, label=ds); ax.annotate(f'{ys[-1]:.3f}', (ep[-1], ys[-1]), textcoords='offset points', xytext=(3, 3), fontsize=8, color=col)
        if not has_nms:
            ax.text(0.5, 0.08, 'no +nms column in this older log:\nnative AP only (dashed)', transform=ax.transAxes, ha='center', fontsize=8, color=INK2)
        for dp in drops_of(r['run']):
            ax.axvline(dp, color=GRID, lw=1.2, zorder=0)
        ax.set_xlim(-1, max(last_epoch(r), 1) + 2)
        ax.set_ylim(0.3, 0.8); ax.set_title(title_of(r['run']) + suffix(r), loc='left', fontsize=9, color=INK)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        if k % cols == 0:
            ax.set_ylabel('AP', color=INK2, fontsize=9)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis('off')
    fig.suptitle('AP per evaluation epoch, one panel per run: +nms solid, native dashed; blue organic, orange 41; dotted = DINO lr 4e-5 final +nms; grey vertical = lr drop',
                 x=0.01, ha='left', color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=110, facecolor=SURFACE); plt.close(fig)


def per_run(r, path):
    panels = [('AP, +nms (deployed)', 5), ('AP, native (no NMS)', 0), ('recall @ score>0.3', 1),
              ('precision @ score>0.3', 2), ('recall, NN dist < 5 px', 4)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), facecolor=SURFACE)
    xmax = max(last_epoch(r), 1) + 2
    for k, (t, ci) in enumerate(panels):
        ax = axes[k // 3][k % 3]; style(ax)
        drawn = False
        for ds, col in SETS:
            ep, ys = finite(*series(r['ev'][ds], ci))
            if ci == 5:
                for lab, v in REFS[ds]:
                    ax.axhline(v, color=col, ls=':', lw=0.8, alpha=0.6)
            if ep:
                drawn = True
                ax.plot(ep, ys, color=col, lw=1.8, marker='o', ms=3, label=ds); ax.annotate(f'{ys[-1]:.3f}', (ep[-1], ys[-1]), textcoords='offset points', xytext=(3, 3), fontsize=8, color=col)
        if not drawn:
            ax.text(0.5, 0.5, 'not logged in this older run', transform=ax.transAxes, ha='center', fontsize=9, color=INK2)
        for dp in drops_of(r['run']):
            ax.axvline(dp, color=GRID, lw=1.2, zorder=0)
        ax.set_xlim(-1, xmax)
        ax.set_ylim(0.2, 0.85) if ci in (0, 5) else ax.set_ylim(0, 1)
        ax.set_title(t, loc='left', fontsize=10, color=INK)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax = axes[1][2]; style(ax); ax.set_title('training loss (heat + 4 x reg)', loc='left', fontsize=10, color=INK)
    lo = r['loss']
    if lo:
        ep = list(lo); ax.plot(ep, [lo[e][0] for e in ep], color=INK2, lw=1.8)
    ax.set_xlim(-1, xmax)
    h, l = axes[0][0].get_legend_handles_labels()
    if h:
        axes[0][0].legend(h, l, frameon=False, fontsize=9, labelcolor=INK)
    for ax in axes[1]:
        ax.set_xlabel('epoch', color=INK2, fontsize=9)
    fig.suptitle(f'{title_of(r["run"])}{suffix(r)}  |  {r["run"]}  |  grey vertical = lr drop; dotted = DINO lr 4e-5 final +nms',
                 x=0.01, ha='left', color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=110, facecolor=SURFACE); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs_dir', default='/mnt/DATA/mlgidDETECT_DINO_HEATMAP/hm_runs'); p.add_argument('--out', default=None)
    a = p.parse_args()
    out = a.out or os.path.join(a.runs_dir, 'curves')
    os.makedirs(os.path.join(out, 'per_run'), exist_ok=True)
    runs = load(a.runs_dir)
    print(f'{len(runs)} runs with evaluation logs')
    write_csv(runs, os.path.join(out, 'curves_data.csv'))
    overview(runs, os.path.join(out, 'overview_apnms.png'), 5, '+nms, deployed: floor 0.1 / top-225')
    overview(runs, os.path.join(out, 'overview_native.png'), 0, 'native decode, no NMS')
    small_multiples(runs, os.path.join(out, 'overview_small_multiples.png'))
    for r in runs:
        if r['status'] != 'stopped':
            per_run(r, os.path.join(out, 'per_run', re.sub(r'[^A-Za-z0-9_.-]+', '_', tag_of(r['run'])) + '.png'))
    open(os.path.join(out, 'README.txt'), 'w').write(__doc__ + '\n\nRuns included:\n' + '\n'.join(
        f'  {r["run"]}  ->  {title_of(r["run"])}  ({r["status"]}, last eval epoch {last_epoch(r)})' for r in runs) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
