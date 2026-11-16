"""Simulator-side smoke test for the multi-CONTRAST channels (DINO_4scale_swin_mcc.py).

Checks the half diagnostics/mc_smoke.py does not: that FastSimulation.contrast_stack emits
four channels that are actually different images, in range, with the mask channel binary and
consistent with the zeroed pixels of the contrast channels.
Run on a GPU node:  python diagnostics/mcc_smoke.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from main import get_args_parser, SimulationDataset
from util.slconfig import SLConfig
from util.channels import CONTRAST_CHANNELS

config_file = sys.argv[1] if len(sys.argv) > 1 else 'config/DINO/DINO_4scale_swin_mcc.py'
args = get_args_parser().parse_args(['--config_file', config_file, '--output_dir', '/tmp/mcc_smoke'])
for k, v in SLConfig.fromfile(config_file)._cfg_dict.to_dict().items():
    if k not in vars(args):
        setattr(args, k, v)
args.device = 'cuda'; args.export = False

ds = SimulationDataset(args)
names = [c['name'] for c in CONTRAST_CHANNELS] + ['mask']
for n in range(3):
    img, tgt = ds[n]
    assert tuple(img.shape) == (4, 512, 1024), img.shape
    print(f"[sample {n}] {tuple(img.shape)}, {len(tgt['boxes'])} boxes")
    for i, nm in enumerate(names):
        c = img[i]
        print(f"    ch{i} {nm:18s} min={c.min():.3f} max={c.max():.3f} mean={c.mean():.4f}")
    m = img[3].bool()
    assert torch.isin(img[3], torch.tensor([0., 1.], device=img.device)).all(), 'mask not binary'
    for i in range(3):
        assert img[i][~m].abs().max() == 0, f'ch{i} nonzero outside the mask'
    # channels must not be duplicates of each other
    for i in range(3):
        for j in range(i + 1, 3):
            d = (img[i] - img[j]).abs().mean().item()
            assert d > 1e-4, f'ch{i} and ch{j} are identical (mean|diff| {d:.2e})'
    print(f"    pairwise mean|diff| 0-1 {(img[0]-img[1]).abs().mean():.4f}  "
          f"0-2 {(img[0]-img[2]).abs().mean():.4f}  1-2 {(img[1]-img[2]).abs().mean():.4f}")
print('MCC SIM SMOKE PASSED')
