"""Check that a backbone weights file really loads into the frozen swin (a silent strict=False partial load would be
a random backbone). Exit 1 if more than 4 keys are missing or any key is unexpected.
  python heatmap/check_backbone.py <weights.pth> [key_prefix]     (e.g. prefix 'backbone.0.' for a DINO checkpoint slice)"""
import os, sys, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import torch
from models.heatmap_head import HeatmapNet
from util.misc import clean_state_dict

path = sys.argv[1]; pre = sys.argv[2] if len(sys.argv) > 2 else ''
sd = torch.load(path, map_location='cpu'); sd = clean_state_dict(sd.get('model', sd))
sd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre) and 'head' not in k[len(pre):]}
net = HeatmapNet(None, freeze_backbone=True)
r = net.backbone.load_state_dict(sd, strict=False)
print(f'{path}: {len(sd)} tensors, missing {len(r.missing_keys)} {r.missing_keys[:6]}, unexpected {len(r.unexpected_keys)} {r.unexpected_keys[:6]}')
sys.exit(1 if (len(r.missing_keys) > 4 or r.unexpected_keys) else 0)
