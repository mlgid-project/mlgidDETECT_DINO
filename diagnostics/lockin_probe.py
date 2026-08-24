"""Phase Z.1 — lock-in check: did the head stop improving while its features kept improving?

Phase Z cleared the head's own training signal: trained ALONE on a frozen trunk, this exact head under
this exact loss reaches 0.31 px where it manages 3.83 px in the real run, and it still reaches 0.48 px
when close pairs are thinned to the real organic rate. Three differences from the real run survive —
the trunk was frozen (and frozen at convergence), every competing loss term was absent, and the frames
were the ladder's simplified ones. The first two are the conditions of JOINT training.

The joint-training story, if it is the right one, has a signature: the head is asked to regress boxes
from the very first epoch, when the features cannot yet separate a close pair and "emit one wide box
covering both" is genuinely the best available answer. It learns that, the features later improve, and
the head never revisits it. If so, the information AVAILABLE to the head should improve over training
while the information USED stays flat.

This measures exactly that, at the only two time points that survive on disk:

    checkpoint0279.pth   epoch 279, just before the lr drop at 280   (organic AP 0.542)
    checkpoint.pth       epoch 436, the end of the run               (organic AP 0.561)

Per checkpoint, on ONE fixed image set, phase Y's two numbers:
    AVAILABLE = ridge regression on the exact 256-dim vector the box head reads
    USED      = the trained head's own error on the same tokens

Reading it:
  ridge improves, head flat        -> the features kept getting better and the head did not follow.
                                      The lock-in signature, over this window.
  both flat                        -> also consistent with lock-in, but it happened before epoch 279
                                      and this pair of checkpoints cannot date it.
  both improve together            -> no lock-in over this window; the story would have to live
                                      earlier than epoch 279, which is not measurable from disk.

LIMIT, stated up front: two points is not a curve. `save_checkpoint_interval = 1000` in the swin
config and the run stopped at 436, so no intermediate epochs were ever written. This can show whether
the gap between available and used widened, narrowed or held across 157 epochs; it cannot say when the
head became stuck, and a flat-flat result is genuinely ambiguous about timing. Nothing here should be
written up as dating the lock-in.

GPU, ~6 min. See tmp_diag/run_lockin.sbatch.
"""
import os, sys, json

os.environ.setdefault('LADDER_AXIS', 'chi')
os.environ.setdefault('LADDER_ISO', '1')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import diagnostics.separation_ladder as L
L.N_IMG = 50                                   # phase Y's frame count, so numbers are comparable
from diagnostics.linear_readout_probe import probe, SEPS
from diagnostics.prominence_probe import CONFIG, OUT

_CUR = '/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation'
_RUN = f'{_CUR}/detector_runs/dino_ssl1'
# Same architecture, same run, two epochs. The ONLY variable is how long it had trained.
MODELS = [('ep279', f'{_RUN}/checkpoint0279.pth', CONFIG),
          ('ep436', f'{_RUN}/checkpoint.pth', CONFIG)]
AP = {'ep279': 0.542, 'ep436': 0.561}          # organic AP, from exp_ap_organic.txt


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  frames/rung={L.N_IMG}", flush=True)
    L.SEPS = SEPS
    data = L.make_images(dev)                  # one fixed image set for both checkpoints
    summary = {}
    for name, ckpt, cfg_file in MODELS:
        print(f"\n########## {name} ##########", flush=True)
        summary[name] = probe(name, ckpt, cfg_file, data, dev)

    print("\n" + "=" * 96)
    print("  Median |error| in px predicting a token's offset to ITS OWN peak")
    print("  AVAILABLE = ridge on the head's own input vector;  USED = the trained head itself")
    for nm in summary:
        r = summary[nm]
        print(f"\n  --- {nm}  (organic AP {AP.get(nm, float('nan')):.3f})")
        print(f"  {'sep':>5s} {'USED (head)':>13s} {'AVAILABLE (ridge)':>19s} {'gap':>8s}")
        for s in SEPS:
            e = r['enc'].get(s)
            if not e:
                continue
            h = r['head'].get(s, float('nan'))
            print(f"  {s:5d} {h:13.2f} {e['ridge']:19.2f} {h - e['ridge']:8.2f}")

    a, b = MODELS[0][0], MODELS[1][0]
    print("\n" + "=" * 96)
    print(f"  Change from {a} to {b} (negative = improved over 157 epochs)")
    print(f"  {'sep':>5s} {'d USED':>10s} {'d AVAILABLE':>13s} {'d gap':>10s}")
    for s in SEPS:
        ea, eb = summary[a]['enc'].get(s), summary[b]['enc'].get(s)
        if not ea or not eb:
            continue
        du = summary[b]['head'].get(s, float('nan')) - summary[a]['head'].get(s, float('nan'))
        da = eb['ridge'] - ea['ridge']
        print(f"  {s:5d} {du:10.2f} {da:13.2f} {du - da:10.2f}")

    fig, ax = plt.subplots(figsize=(7.6, 5))
    for nm, c in zip(summary, ('tab:orange', 'tab:blue')):
        r = summary[nm]
        ss = [s for s in SEPS if s in r['enc']]
        ax.plot(ss, [r['head'][s] for s in ss], 'o-', color=c, label=f'{nm}: USED (trained head)')
        ax.plot(ss, [r['enc'][s]['ridge'] for s in ss], 's--', color=c, alpha=.6,
                label=f'{nm}: AVAILABLE (ridge)')
    ax.set_xscale('log'); ax.set_xticks(SEPS); ax.set_xticklabels(SEPS)
    ax.set_xlabel('planted χ-separation (px)')
    ax.set_ylabel('median |error| predicting the offset to its own peak (px)')
    ax.set_title('Phase Z.1: does the head follow its features over training?')
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lockin.png')
    fig.savefig(p, dpi=110, bbox_inches='tight')
    print(f"\nsaved {p}")
    json.dump(summary, open(os.path.join(OUT, 'lockin_probe.json'), 'w'), indent=2, default=str)
    print("PROBE DONE")


if __name__ == '__main__':
    main()
