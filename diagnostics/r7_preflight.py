"""Pre-flight for run 7: build the training dataset through main.py's OWN code path from the r7
config, print its epoch-0 wiring lines, draw frames through SimulationDataset.__getitem__, and time
them against r6 (the ring texture renders per ring, so cost must be checked before a 72 h run).

SimulationDataset hard-codes device='cuda'; on a CPU node the two simulators are forced onto the
CPU through constructor wrappers here, and the dataset's own device is set to 'cpu' after
construction. Nothing else differs from training.
"""
import os, sys, time, random, tempfile
import numpy as np
sys.path.insert(0, '/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
os.chdir('/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO')
import torch
import main
from util.slconfig import SLConfig
import realbkg_simulation as RS
import simulation as SIM
from diagnostics.cache_realbkg_donors import load_into

_rb, _fs = RS.RealBkgSimulation.__init__, SIM.FastSimulation.__init__
RS.RealBkgSimulation.__init__ = lambda self, *a, **k: _rb(self, *a, **dict(k, device='cpu'))
SIM.FastSimulation.__init__ = lambda self, *a, **k: _fs(self, *a, **dict(k, device='cpu'))
RS.RealBkgSimulation._load_donors = lambda self, *a, **k: load_into(
    self, '/mnt/lustre/work/schreiber/szb389/datasets/realbkg_donors_selected.h5')


def build(cfg_path):
    args = main.get_args_parser().parse_args(['-c', cfg_path, '--output_dir', tempfile.mkdtemp()])
    cfg = SLConfig.fromfile(cfg_path)
    for k, v in cfg._cfg_dict.to_dict().items():          # exactly main.py's merge
        if k not in vars(args):
            setattr(args, k, v)
    ds = main.SimulationDataset(args)
    ds.device = 'cpu'
    return ds


def time_frames(ds, n):
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    t, nb, nr = [], [], []
    for i in range(n):
        t0 = time.perf_counter()
        img, tgt = ds[i]
        t.append(time.perf_counter() - t0)
        assert torch.isfinite(img).all(), 'non-finite image'
        assert (tgt['boxes'] >= 0).all() and (tgt['boxes'] <= 1).all(), 'boxes outside [0,1]'
        nb.append(len(tgt['boxes'])); nr.append(int(tgt['labels'].sum()))
    t = np.array(t) * 1000
    return t, np.array(nb), np.array(nr)


if __name__ == '__main__':
    torch.set_num_threads(1)     # a dataloader worker is single-threaded
    out = {}
    for name in ('r6', 'r7'):
        print(f'\n##### {name}: SimulationDataset wiring lines', flush=True)
        ds = build(f'config/DINO/DINO_4scale_swin_realbkg_{name}.py')
        t, nb, nr = time_frames(ds, 120)
        out[name] = t
        print(f'{name}: {len(t)} frames via __getitem__ | ms/frame median {np.median(t):.0f} mean {t.mean():.0f} '
              f'p90 {np.percentile(t, 90):.0f} | boxes/frame {nb.mean():.1f} | label-1 (ring flag) /frame {nr.mean():.1f}',
              flush=True)
    r = np.median(out['r7']) / np.median(out['r6'])
    print(f'\nr7 / r6 per-frame cost: {r:.2f}x  -> run 6 epoch ~15 min, r7 epoch ~{15 * r:.0f} min '
          f'if the dataloader is the bottleneck (10 workers vs ~1.8 s/it GPU step at batch 2)', flush=True)
    print('PREFLIGHT OK')
