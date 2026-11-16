"""zero_invalid=True must equal feeding the model an image whose invalid pixels were zeroed beforehand, and must not change
anything when the invalid pixels are already 0 (the eval files). Random-init net, CPU or GPU:
  python heatmap/test_zero_invalid.py [--device cpu|cuda]"""
import os, sys, types, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import torch
from models.heatmap_head import HeatmapNet

p = argparse.ArgumentParser(); p.add_argument('--device', default='cpu'); a = p.parse_args()
torch.manual_seed(0)
H, W = 512, 1024
img = torch.randn(1, 1, H, W, device=a.device)
mask = torch.rand(1, H, W, device=a.device) > 0.4
mask[:, :, :200] = True
for mode in ('he', 'full'):
    ref = HeatmapNet(None, freeze_backbone=True, chan_mode=mode).to(a.device).eval()
    zi = HeatmapNet(None, freeze_backbone=True, chan_mode=mode, zero_invalid=True).to(a.device).eval()
    zi.load_state_dict(ref.state_dict())
    z = img.masked_fill(~mask[:, None], 0.)
    with torch.no_grad():
        o_ref_z = ref(z, mask); o_zi = zi(img, mask); o_zi_z = zi(z, mask); o_ref_raw = ref(img, mask)
    d1 = max((o_zi[k] - o_ref_z[k]).abs().max().item() for k in o_zi)
    d2 = max((o_zi_z[k] - o_ref_z[k]).abs().max().item() for k in o_zi)
    d3 = max((o_ref_raw[k] - o_ref_z[k]).abs().max().item() for k in o_zi)
    print(f'{mode}: zero_invalid(raw) vs plain(pre-zeroed) {d1:.2e} | zero_invalid(pre-zeroed) vs plain(pre-zeroed) {d2:.2e} | '
          f'(plain raw vs pre-zeroed, should be > 0: {d3:.2e})')
    assert d1 < 1e-5 and d2 < 1e-5 and d3 > 1e-4
print('OK')
