"""Head-size / out_stride options: shapes, targets + loss + backward at every config, trainable-parameter counts.
Random-init net on CPU (a few minutes) or GPU:  python heatmap/test_head_config.py [--device cpu|cuda] [--ckpt <old ridge checkpoint.pth>]
--ckpt additionally checks that the DEFAULT head loads an existing checkpoint with strict=False and NO missing/unexpected head keys."""
import os, sys, types, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import torch
from models.heatmap_head import HeatmapNet, decode
from heatmap.targets_loss import build_targets, heatmap_loss

p = argparse.ArgumentParser(); p.add_argument('--device', default='cpu'); p.add_argument('--ckpt', default=None); a = p.parse_args()
CONFIGS = {'default (stride 2)': dict(),
           'stride 1': dict(out_stride=1),
           'wide (dim 256, tower 128 x4, stem 64)': dict(dim=256, tower_ch=128, tower_depth=4, stem_ch=64)}
H, W = 512, 1024
img = torch.randn(1, 1, H, W, device=a.device)
boxes = torch.tensor([[100., 200., 108., 212.], [104., 203., 110., 211.], [400., 50., 412., 450.]], device=a.device)
lab = torch.tensor([0, 0, 1], device=a.device)
for name, kw in CONFIGS.items():
    torch.manual_seed(0)
    m = HeatmapNet(None, freeze_backbone=True, **kw).to(a.device).train()
    st = kw.get('out_stride', 2)
    o = m(img)
    assert o['heat'].shape == (1, 2, H // st, W // st) and o['reg'].shape == (1, 4, H // st, W // st), (name, o['heat'].shape)
    h, r, w = build_targets(boxes, lab, H, W, st, 'ridge')
    loss, parts = heatmap_loss(o, h[None], r[None], w[None])
    loss.backward()
    n_tr = sum(q.numel() for q in m.parameters() if q.requires_grad)
    n_bb = sum(q.numel() for q in m.backbone.parameters())
    d = decode({k: v.detach() for k, v in o.items()}, st, 225)[0]
    print(f'{name}: out {tuple(o["heat"].shape[-2:])} | trainable {n_tr:,} (frozen swin {n_bb:,}) | loss {loss.item():.3f} | decoded {len(d[1])} peaks', flush=True)
    if a.ckpt and not kw:
        ck = torch.load(a.ckpt, map_location='cpu')
        r_ = m.load_state_dict(ck['model'], strict=False)
        bad = [k for k in list(r_.missing_keys) + list(r_.unexpected_keys) if not k.startswith('backbone.')]
        print('  checkpoint head keys missing/unexpected:', bad); assert not bad
print('OK')
