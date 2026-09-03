"""Smoke test for the 4-channel input path (config DINO_4scale_swin_mc.py).

[1] SSL checkpoint loads into the 4-channel backbone (zero-pad expansion in backbone.py);
    padded patch-embed channels 1-3 are exactly zero.
[2] SimulationDataset emits (4,512,1024) via build_channels.
[3] One batch: forward + DN loss + backward; new channels receive gradient.
[4] Step-0 equivalence: with zero weights on ch1-3, replacing those channels with junk
    must not change the model output at all.
Run on a GPU node:  python diagnostics/mc_smoke.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from main import build_model_main, get_args_parser, SimulationDataset, collate_fn
from util.slconfig import SLConfig

config_file = 'config/DINO/DINO_4scale_swin_mc.py'
parser = get_args_parser()
args = parser.parse_args(['--config_file', config_file, '--output_dir', '/tmp/mc_smoke'])
cfg = SLConfig.fromfile(config_file)
for k, v in cfg._cfg_dict.to_dict().items():
    if k not in vars(args):
        setattr(args, k, v)
args.device = 'cuda'
args.export = False

model, criterion, _ = build_model_main(args)
model.cuda(); criterion.cuda()

w = model.backbone[0].patch_embed.proj.weight
assert tuple(w.shape) == (192, 4, 4, 4), w.shape
print(f"[1] patch_embed {tuple(w.shape)}; ch0 mean|w| {w[:,0].abs().mean():.4f}; "
      f"ch1-3 max|w| {w[:,1:].abs().max():.2e} (must be 0.00e+00)")

ds = SimulationDataset(args)
im0, t0 = ds[0]; im1, t1 = ds[1]
assert tuple(im0.shape) == (4, 512, 1024), im0.shape
print(f"[2] sample {tuple(im0.shape)}; B1 range [{im0[1].min():.2f},{im0[1].max():.2f}]; "
      f"valid frac {im0[3].mean():.2f}; {len(t0['boxes'])} boxes")

samples, targets = collate_fn([(im0, t0), (im1, t1)])
model.train(); criterion.train()
outputs = model(samples, targets)
loss_dict = criterion(outputs, targets)
loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict if k in criterion.weight_dict)
loss.backward()
g = model.backbone[0].patch_embed.proj.weight.grad
print(f"[3] forward+backward OK, loss {loss.item():.3f}; grad mean|g| ch0 {g[:,0].abs().mean():.2e}, "
      f"ch1-3 {g[:,1:].abs().mean():.2e} (nonzero => new channels can learn)")

model.eval()
with torch.no_grad():
    x = im0.unsqueeze(0)
    x_junk = x.clone(); x_junk[:, 1:] = torch.randn_like(x_junk[:, 1:])
    o = model(x); o_junk = model(x_junk)
    d_logit = (o['pred_logits'] - o_junk['pred_logits']).abs().max().item()
    d_box = (o['pred_boxes'] - o_junk['pred_boxes']).abs().max().item()
print(f"[4] step-0 equivalence: junk in ch1-3 changes logits by {d_logit:.2e}, boxes by {d_box:.2e} "
      f"(must be 0 -> net starts exactly as a 1-channel model)")
print("SMOKE TEST PASSED" if d_logit == 0 and d_box == 0 else "STEP-0 CHECK FAILED")
