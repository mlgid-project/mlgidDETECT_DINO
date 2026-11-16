"""Per-channel mean and std of the public DINOv3 stride-16 features on corpus crops, for pretrain_dinov3.py --patch_stats.

  python heatmap/make_patch_stats.py --out <stats.pt> [--n 400] [--bb dinov3_vith16plus --weights <pth>]
Default backbone is the ConvNeXt-L (byte-identical to the old behaviour). For a ViT the stride-16 map is the token grid after the third tapped block (the map the patch loss uses).
"""
import argparse, importlib.util, random, types
import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location('pd', __file__.replace('make_patch_stats.py', 'pretrain_dinov3.py'))
pd = importlib.util.module_from_spec(spec); spec.loader.exec_module(pd)
DATA = '/mnt/DATA/mlgidDETECT_DINO_HEATMAP'
p = argparse.ArgumentParser()
p.add_argument('--out', required=True); p.add_argument('--n', type=int, default=400)
p.add_argument('--bb', default='dinov3_convnext_large'); p.add_argument('--weights', default=f'{DATA}/backbone_export/dinov3/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth')
a = p.parse_args()
dev = 'cuda'
net = pd.Net(a.bb, a.weights, types.SimpleNamespace(grad_ckpt=False, prototypes=1024, proto_patch=1024)).to(dev).eval()
ds = pd.Corpus([f'{DATA}/corpus_v2/scans', f'{DATA}/corpus_v2/scans_other'], f'{DATA}/moneta_sweep/holdout/pretrain_exclude_v2.tsv', (256, 512), (128, 256), 1, True)
random.seed(0)
feats = []
for i in random.sample(range(len(ds)), a.n):
    g, m, _ = ds[i]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        _, tm = net.features(g.to(dev))
    valid = F.avg_pool2d(m.to(dev)[:, None].float(), 16)[:, 0] > 0.5
    feats.append(tm.permute(0, 2, 3, 1)[valid].float().cpu())
f = torch.cat(feats)
sd = f.std(0) + 1e-6
sd = sd.clamp_min(0.1 * sd.median())      # floor: ViT-H+ has one near-constant channel (std 4e-4), dividing by it would blow up that channel ~2000x; no ConvNeXt-L channel is affected
torch.save(dict(mu=f.mean(0), sd=sd, n=len(f)), a.out)
print('patches', len(f), 'channels', f.shape[1], 'median std/|mean|', (f.std(0) / f.mean(0).abs()).median().item())
