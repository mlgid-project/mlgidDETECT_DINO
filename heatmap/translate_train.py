"""Sim-to-real translation, step 2: an unpaired residual GAN that restyles simulated frames toward the real frames of latent-map regions R4 + R5 (GPU, heatmap-smoke env).

  PYTHONPATH=. python heatmap/translate_train.py --data <latent_map/translate> --out <dir> [--minutes 90] [--n_train_sim 3600]

G: U-Net on (sim image, no-data mask, noise channel) -> residual r; output = clamp(x + r, 0, 1) * mask  (the residual form keeps the peaks where the simulator put them, so its boxes stay valid; the noise channel lets G produce real-looking texture).
D: PatchGAN on (image, mask), LSGAN; real = random crops of the R4 + R5 corpus frames (held-out beamtimes excluded), fake = G output on random crops of the first --n_train_sim sim frames (the rest are kept for evaluation).
Content loss: L1 between the blurred (sigma 2) output and input inside the mask (weight --w_content) + small L1 on the residual (--w_res), so the geometry survives while the texture / background / noise are free.
Writes <out>/G_last.pt, G_<it>.pt, samples_<it>.png (sim | translated | real), train.log."""
import argparse, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.InstanceNorm2d(o, affine=True), nn.LeakyReLU(0.2, True), nn.Conv2d(o, o, 3, padding=1), nn.InstanceNorm2d(o, affine=True), nn.LeakyReLU(0.2, True))


class G(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        ch = [c, c * 2, c * 4, c * 8]
        self.enc = nn.ModuleList([block(3, ch[0])] + [block(ch[i], ch[i + 1]) for i in range(3)])
        self.mid = block(ch[3], ch[3])
        self.dec = nn.ModuleList([block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(3))])
        self.out = nn.Conv2d(ch[0], 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)       # starts as identity

    def forward(self, x, m, noise=None):
        noise = torch.randn_like(x) if noise is None else noise
        h, skips = torch.cat([x, m, noise], 1), []
        for i, e in enumerate(self.enc):
            h = e(h); skips.append(h)
            if i < 3:
                h = F.avg_pool2d(h, 2)
        h = self.mid(h)
        for d, s in zip(self.dec, reversed(skips[:-1])):
            h = d(torch.cat([F.interpolate(h, size=s.shape[-2:], mode='bilinear', align_corners=False), s], 1))
        return (x + self.out(h)).clamp(0, 1) * m


class D(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(sn(nn.Conv2d(2, c, 4, 2, 1)), nn.LeakyReLU(0.2, True), sn(nn.Conv2d(c, c * 2, 4, 2, 1)), nn.LeakyReLU(0.2, True), sn(nn.Conv2d(c * 2, c * 4, 4, 2, 1)), nn.LeakyReLU(0.2, True),
                                 sn(nn.Conv2d(c * 4, c * 4, 4, 1, 1)), nn.LeakyReLU(0.2, True), sn(nn.Conv2d(c * 4, 1, 4, 1, 1)))

    def forward(self, x, m):
        return self.net(torch.cat([x, m], 1))


def blur(x, s=2.0):
    k = int(4 * s) | 1; t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2; g = torch.exp(-t ** 2 / (2 * s * s)); g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode='reflect'), g.view(1, 1, 1, k)); return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode='reflect'), g.view(1, 1, k, 1))


def crops(I, M, idx, ch, cw, rng, dev):
    xs, ms = [], []
    for j in idx:
        y0 = int(rng.integers(0, 512 - ch + 1)); x0 = int(rng.integers(0, 1024 - cw + 1))
        xs.append(np.asarray(I[j, y0:y0 + ch, x0:x0 + cw])); ms.append(np.asarray(M[j, y0:y0 + ch, x0:x0 + cw]))
    return (torch.from_numpy(np.stack(xs)).to(dev).float()[:, None] / 255, torch.from_numpy(np.stack(ms)).to(dev).float()[:, None])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True); p.add_argument('--out', required=True); p.add_argument('--minutes', type=float, default=90); p.add_argument('--n_train_sim', type=int, default=3600)
    p.add_argument('--bs', type=int, default=6); p.add_argument('--crop', type=int, nargs=2, default=[256, 512]); p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--w_content', type=float, default=10.0); p.add_argument('--w_res', type=float, default=0.5); p.add_argument('--seed', type=int, default=0); p.add_argument('--max_it', type=int, default=100000); p.add_argument('--duty', type=float, default=1.0, help='fraction of the time the GPU is kept busy (sleep after every iteration); < 1 keeps the shared desktop GPU responsive')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True); dev = 'cuda'
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    real = [(np.load(f'{a.data}/real_{r}.npy', mmap_mode='r'), np.load(f'{a.data}/real_{r}_mask.npy', mmap_mode='r'), int(np.load(f'{a.data}/real_{r}_n.npy'))) for r in (4, 5)]
    sI, sM = np.load(f'{a.data}/sim.npy', mmap_mode='r'), np.load(f'{a.data}/sim_mask.npy', mmap_mode='r')
    ntr = a.n_train_sim; print('real frames', [r[2] for r in real], 'sim train frames', ntr, flush=True)
    g, d = G().to(dev), D().to(dev)
    og = torch.optim.Adam(g.parameters(), a.lr, betas=(0.5, 0.999)); od = torch.optim.Adam(d.parameters(), a.lr, betas=(0.5, 0.999))
    ch, cw = a.crop; t0, it, log = time.time(), 0, open(f'{a.out}/train.log', 'a')
    while time.time() - t0 < a.minutes * 60 and it < a.max_it:
        t_it = time.time()
        ri = int(rng.integers(len(real))); rI, rM, rn = real[ri]
        xr, mr = crops(rI, rM, rng.integers(0, rn, a.bs), ch, cw, rng, dev)
        xs, ms = crops(sI, sM, rng.integers(0, ntr, a.bs), ch, cw, rng, dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            fake = g(xs, ms)
            pr, pf = d(xr, mr).float(), d(fake.detach(), ms).float()
            ld = (F.mse_loss(pr, torch.ones_like(pr)) + F.mse_loss(pf, torch.zeros_like(pf))) * 0.5
        od.zero_grad(set_to_none=True); ld.backward(); od.step()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            fake = g(xs, ms); pd = d(fake, ms).float()
            adv = F.mse_loss(pd, torch.ones_like(pd))
            cont = ((blur(fake.float()) - blur(xs)).abs() * ms).sum() / ms.sum().clamp(min=1); res = ((fake.float() - xs).abs() * ms).sum() / ms.sum().clamp(min=1)
            lg = adv + a.w_content * cont + a.w_res * res
        og.zero_grad(set_to_none=True); lg.backward(); og.step(); it += 1
        if a.duty < 1.0:
            torch.cuda.synchronize(); time.sleep((time.time() - t_it) * (1 / a.duty - 1))
        if it % 200 == 0:
            s = f'it {it} {(time.time() - t0) / 60:.1f} min  D {ld.item():.3f}  adv {adv.item():.3f}  content {cont.item():.4f}  res {res.item():.4f}'; print(s, flush=True); log.write(s + '\n'); log.flush()
        if it % 2000 == 0:
            torch.save(g.state_dict(), f'{a.out}/G_last.pt')
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            with torch.no_grad():
                k = min(3, a.bs); fig, ax = plt.subplots(k, 3, figsize=(15, 2.7 * k), squeeze=False)
                for r in range(k):
                    for c, (im, t) in enumerate(((xs[r], 'sim'), (fake[r].float(), 'translated'), (xr[r], 'real'))):
                        ax[r, c].imshow(im[0].cpu(), cmap='gray', vmin=0, vmax=1); ax[r, c].set_title(t if r == 0 else '', fontsize=8); ax[r, c].axis('off')
                fig.tight_layout(); fig.savefig(f'{a.out}/samples_{it:06d}.png', dpi=70); plt.close(fig)
    torch.save(g.state_dict(), f'{a.out}/G_last.pt'); print('done', it, 'iterations', flush=True)


if __name__ == '__main__':
    main()
