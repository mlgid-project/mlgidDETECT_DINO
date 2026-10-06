"""Do the realbkg models fragment rings in their OWN simulator too, or only on 41-style frames?

On real 41 the realbkg models emit almost no ring boxes: 3.7-4.7 predictions per GT ring, ring TP
score 0.22-0.30 against 0.72/0.47 on segments (ring_fp_probe.py). band_probe.py found the realbkg
training frames carry more unlabelled radial bands than labelled rings (v2: 9.95 vs 7.2/frame;
legacy 5.4 vs 16.5). Two readings:
  (a) rings are AMBIGUOUS inside the realbkg sim itself -> the model is unsure of rings even on
      its own training distribution
  (b) it handles its own rings and has simply never seen 41's dense ring forests
Each model is run on fresh frames from the legacy sim and from realbkg v1 (as conv3 trained) and
v2 (run 6), scored against those frames' own labels with the deployed postprocessing.

  python diagnostics/cross_sim_probe.py name=/ckpt.pth [...]
"""
import os, sys, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gap41_split as G
import peak_shape_compare as PS
from band_probe import realbkg_v1_as_trained, realbkg_v2
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.evaluation import Evaluator, get_full_conf_results

N = int(os.environ.get('XSIM_FRAMES', 60))


def frames_of(fn):
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    return [(img, m, gt) for img, m, gt in fn()]


def score(model, args, frames):
    cfg = G._config(G.DATASETS['41'])
    ev = Evaluator()
    tpr, tps, per_ring, n_r, h_r, n_s, h_s, npred = [], [], [], 0, 0, 0, 0, 0
    for img, m, gt in frames:
        with torch.no_grad():
            o = model(torch.tensor(img)[None, None].to(G.DEV).repeat(1, args.num_channels, 1, 1))
        c = filter_boxes(cfg, onnx_to_xyxy(cfg, G._P(), [o['pred_logits'].cpu().numpy(),
                                                       o['pred_boxes'].cpu().numpy()], num_select=225))
        pb = np.asarray(c.boxes, np.float32).reshape(-1, 4); sc = np.asarray(c.scores, np.float32)
        npred += len(pb)
        # sim GT has no confidence; get_full_conf_results needs every level (1.0/0.5/0.1) present, and
        # ap_total does not depend on the split, so cycle them
        ev.get_exp_metrics(c.boxes, c.scores, torch.tensor(gt),
                           np.resize(np.array([1.0, 0.5, 0.1], np.float32), len(gt)))
        ring = G.is_ring_geom(gt, m) if len(gt) else np.zeros(0, bool)
        if len(pb) and len(gt):
            _, ri, ci = G.matcher(torch.tensor(gt).float(), torch.tensor(pb).float())
            ri, ci = np.asarray(ri, int), np.asarray(ci, int)
        else:
            ri = ci = np.zeros(0, int)
        n_r += ring.sum(); h_r += ring[ri].sum(); n_s += (~ring).sum(); h_s += (~ring[ri]).sum()
        tpr += list(sc[ci][ring[ri]]); tps += list(sc[ci][~ring[ri]])
        pc = (pb[:, 0] + pb[:, 2]) / 2 if len(pb) else np.zeros(0)
        per_ring += [int(((pc >= g[0] - 2) & (pc <= g[2] + 2)).sum()) for g in gt[ring]]
    _, df2 = get_full_conf_results(ev.metrics)
    med = lambda v: float(np.median(v)) if len(v) else float('nan')
    return (f'ap {float(df2["ap_total"].values[0]):.3f} | ring recall {h_r/max(n_r,1):.3f} '
            f'TP score {med(tpr):.3f} preds/ring {np.mean(per_ring) if per_ring else float("nan"):.2f} | '
            f'seg recall {h_s/max(n_s,1):.3f} TP score {med(tps):.3f} | preds/frame {npred/len(frames):.0f}')


if __name__ == '__main__':
    targets = [a.split('=', 1) for a in sys.argv[1:]]
    if not targets:
        raise SystemExit(__doc__)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    if os.environ.get('XSIM_CONFIG'):
        # a candidate simulator config against the legacy reference, e.g. the r7 ring fix
        cfgp = os.environ['XSIM_CONFIG']
        sims = {'legacy': frames_of(lambda: PS.legacy(N)),
                os.path.basename(cfgp).replace('.py', ''): frames_of(lambda: PS.realbkg6(N, config=cfgp))}
    else:
        sims = {'legacy': frames_of(lambda: PS.legacy(N)),
                'realbkg_v1': frames_of(lambda: realbkg_v1_as_trained(N)),
                'realbkg_v2': frames_of(lambda: realbkg_v2(N))}
    for s, fr in sims.items():
        r = [int(G.is_ring_geom(g, m).sum()) for _, m, g in fr]
        print(f'{s}: {len(fr)} frames, {np.mean([len(g) for _, _, g in fr]):.1f} boxes/frame, '
              f'{np.mean(r):.1f} rings/frame', flush=True)
    for name, ckpt in targets:
        model, args, ep = G.load_model(ckpt)
        print(f'\n===== {name} (epoch {ep})', flush=True)
        for s, fr in sims.items():
            print(f'  on {s:<11} {score(model, args, fr)}', flush=True)
        del model
