"""Plot the in-training eval curves of one or more heatmap runs (matplotlib, headless).

  python heatmap/plot_runs.py simmim=<run_dir> [random=<run_dir> ...] --out curves.png

Reads <run_dir>/exp_ap_organic.txt, exp_ap_41.txt (lines: epoch, ap, recall0.3, prec0.3, chigap<5) and
<run_dir>/train.log ("[epoch N] loss L heat H reg R") for a second loss figure (<out stem>_loss.png).
Grey dashed lines are REFERENCES from earlier notes, not scored by this code: ap_total of ssl1 / dino_lr4e5_1
and ssl1's recall / precision / close-pair recall. The heatmap curves are the NATIVE decode (peak picking, no NMS)
at its own box convention, so the dashed lines are a rough guide, not a like-for-like comparison.
Colours: first three validated categorical slots (blue, orange, aqua), fixed order by run order.
"""
import argparse, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

SERIES = ['#2a78d6', '#eb6834', '#1baf7a']          # categorical slots 1-3 (validated all-pairs)
SURFACE, INK, INK2, GRID, REF = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e1', '#8a8985'
SETS = ['organic', '41']
COLS = [('AP (native, no NMS)', 0), ('recall @ score>0.3', 1), ('precision @ score>0.3', 2),
        ('recall, chi-gap < 5 px', 3), ('recall, NN dist < 5 px', 4), ('AP, +nms (deployed)', 5)]
# references: (set, column) -> [(label, value)]
REFS = {('organic', 0): [('ssl1 AP 0.568', 0.568), ('dino_lr4e5_1 AP 0.608', 0.6081)],
        ('41', 0): [('ssl1 AP 0.762', 0.762), ('dino_lr4e5_1 AP 0.761', 0.7613)],
        ('organic', 1): [('ssl1 recall 0.537', 0.537)], ('41', 1): [('ssl1 recall 0.772', 0.772)],
        ('organic', 2): [('ssl1 prec 0.841', 0.841)], ('41', 2): [('ssl1 prec 0.705', 0.705)]
        }
NUM = r'([-+0-9.eE]+|nan)'
# chi-gap and Euclid <5px buckets are NOT the same peak sets as the earlier ssl1 reference (165 peaks), so no ref lines there


def read_eval(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    for line in open(path):
        m = re.match(rf'\s*(\d+)\s+{NUM}\s+recall0\.3\s+{NUM}\s+prec0\.3\s+{NUM}\s+chigap<5\s+{NUM}(?:\s+eu<5\s+{NUM}(?:\s+n<5\s+chi=\d+\s+eu=\d+(?:\s+apnms\s+{NUM})?)?)?', line)
        if m:   # later lines win (resumes); the eu<5 column only exists in newer logs
            rows[int(m.group(1))] = [float(m.group(i)) for i in range(2, 6)] + [float(m.group(6)) if m.group(6) else float('nan'), float(m.group(7)) if m.group(7) else float('nan')]
    return dict(sorted(rows.items()))


def read_loss(path):
    rows = {}
    if os.path.exists(path):
        for line in open(path):
            m = re.search(rf'\[epoch (\d+)\] loss {NUM} heat {NUM} reg {NUM}', line)
            if m:
                rows[int(m.group(1))] = [float(m.group(i)) for i in range(2, 5)]
    return dict(sorted(rows.items()))


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', help='name=run_dir')
    p.add_argument('--out', default='curves.png')
    p.add_argument('--no_refs', action='store_true')
    a = p.parse_args()
    runs = [r.split('=', 1) for r in a.runs]
    if len(runs) > 3:
        raise SystemExit('at most 3 runs (validated palette has 3 all-pairs slots)')
    fig, axes = plt.subplots(2, 6, figsize=(22, 7), facecolor=SURFACE, sharex=True)
    for si, ds in enumerate(SETS):
        for name_i, (title, ci) in enumerate(COLS):
            ax = axes[si][name_i]; style(ax)
            if not a.no_refs:
                for lab, v in REFS.get((ds, ci), []):
                    ax.axhline(v, color=REF, ls='--', lw=1)
            for ri, (name, d) in enumerate(runs):
                ev = read_eval(os.path.join(d, f'exp_ap_{ds}.txt'))
                if not ev:
                    continue
                ep = list(ev); ys = [ev[e][ci] for e in ep]
                ax.plot(ep, ys, color=SERIES[ri], lw=2, marker='o', ms=4, label=name)
                ax.annotate(f'{ys[-1]:.3f}', (ep[-1], ys[-1]), textcoords='offset points', xytext=(4, 4),
                            fontsize=8, color=INK)
            ax.set_ylim(0, 1)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            if si == 0:
                ax.set_title(title, fontsize=10, color=INK, loc='left')
            if name_i == 0:
                ax.set_ylabel(ds, color=INK, fontsize=11)
            if si == 1:
                ax.set_xlabel('epoch', color=INK2, fontsize=9)
    h, l = axes[0][0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc='upper right', ncol=len(l), frameon=False, fontsize=10, labelcolor=INK)
    fig.suptitle('Heatmap detector: in-training eval (native decode)', x=0.01, ha='left', color=INK, fontsize=13)
    if not a.no_refs:
        key = []
        for ds in SETS:
            parts = [lab for ci in range(3) for lab, _ in REFS.get((ds, ci), [])]
            key.append(f'{ds}: ' + ', '.join(parts))
        fig.text(0.01, 0.045, 'Dashed grey = earlier reference numbers (not scored by this code; different decode and box '
                 'convention).', fontsize=8, color=INK2)
        fig.text(0.01, 0.025, key[0], fontsize=8, color=INK2)
        fig.text(0.01, 0.005, key[1], fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(a.out, dpi=130, facecolor=SURFACE)
    print('saved', a.out)

    # training-loss figure
    fig2, ax2 = plt.subplots(1, 3, figsize=(13, 3.6), facecolor=SURFACE, sharex=True)
    any_loss = False
    for k, t in enumerate(['loss (heat + 4 x reg)', 'heat', 'reg']):
        style(ax2[k]); ax2[k].set_title(t, fontsize=10, color=INK, loc='left'); ax2[k].set_xlabel('epoch', color=INK2, fontsize=9)
        for ri, (name, d) in enumerate(runs):
            lo = read_loss(os.path.join(d, 'train.log'))
            if lo:
                any_loss = True
                ep = list(lo)
                ax2[k].plot(ep, [lo[e][k] for e in ep], color=SERIES[ri], lw=2, label=name)
    if any_loss:
        ax2[0].legend(frameon=False, fontsize=9, labelcolor=INK)
        out2 = os.path.splitext(a.out)[0] + '_loss.png'
        fig2.tight_layout(); fig2.savefig(out2, dpi=130, facecolor=SURFACE); print('saved', out2)


if __name__ == '__main__':
    main()
