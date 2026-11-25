"""Continue pretraining the public DINOv3 ConvNeXt on the real-frame corpus, DINO style, on one GPU.

  python heatmap/pretrain_dinov3.py --out <dir> --corpus <dir>/scans [<dir>/scans_other] --exclude <pretrain_exclude.tsv>

The public checkpoint holds the backbone only, and the DINOv3 ConvNeXt was distilled from a 7B teacher that is not used here. So:
student and EMA teacher both start from the public weights, with new heads. Stage 1 trains the heads with the backbone frozen, stage 2 opens the backbone at a low lr.
Losses: image level (pooled stride 32 features, across views, teacher sees the global views) and patch level (stride 16 features: pixels of random 16 x 16 blocks are zeroed for the student, it has to predict the teacher's unmasked patch there).
Input: the detector's own contrast (clip 5 / 99.5, log, HE) from the stored real intensities, crops at native scale. No flips or rotation: q and chi are physical axes.
ConvNeXt has no batch norm, so --accum gives a larger batch at the same memory.
Against collapse (the 10k run of 2026-11-08 ended with 9 of 8192 image prototypes in use), all off by default so older runs stay reproducible:
  --sinkhorn N         teacher targets by Sinkhorn-Knopp equipartition (N iterations) over a queue of recent teacher outputs, instead of the running centre
  --t_warmup LO HI N   teacher temperature from LO to HI over the first N steps instead of the constant --t_teacher
  --koleo W            KoLeo spread term on the pooled features, views of the same index across images
  --gram W             Gram anchoring: the student's patch-to-patch similarity must stay close to that of the public backbone
  --proto_patch K      its own prototype count for the patch head
  --patch_stats F      standardise the stride-16 features per channel before the patch head (stats of the public backbone, make_patch_stats.py): they are nearly the same at every patch (cosine 0.92) and the first patch head saw targets at ln K for the whole run
  --patch_g1           fix the patch head's prototype norm at 1 (weight-norm gain was learnable and 0.54)
  --min_perp_frac F    stop (exit 3) when the raw teacher uses fewer than F * prototypes at a log step after --check_from
  --min_div F          stop (exit 3) when the batch-mean teacher perplexity is below F x min(prototypes, rows x mean single-row perplexity): independent of how sharp the targets get
<out>/backbone_teacher.pth has the keys of the public file, so it goes straight into train.py --bb_path."""
import argparse, glob, json, math, os, sys, time, types
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _n, _s in (('models', 'models'),):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import copy
from types import SimpleNamespace
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from models.dinov3_backbone import DinoV3Backbone
from util.exp_preprocess import apply_contrast

DEPLOYED = {'clip': (5.0, 99.5), 'log': True, 'gamma': None, 'he': True, 'clahe': None}


def decode(v, log_max):
    """stored uint16 -> intensity, no-data mask"""
    ok = v > 0
    return np.where(ok, 10.0 ** ((v.astype(np.float32) - 1) / 65534 * log_max) - 1, 0).astype(np.float32), ok


class Corpus(Dataset):
    def __init__(self, dirs, exclude, g, l, n_local, flip=False):
        out = {x.split('\t')[1] for x in open(exclude)} if exclude else set()
        found = [f for d in dirs for f in sorted(glob.glob(os.path.join(d, '*', '*.h5')))]
        self.files = [f for f in found if f not in out]
        print(f'exclude list: {len(out)} entries, {len(found) - len(self.files)} matched a corpus file (must be equal, else the paths do not fit this machine)', flush=True)
        counts = []
        for f in self.files:
            with h5py.File(f, 'r') as h:
                counts.append(h['images'].shape[0])
        self.start = np.concatenate([[0], np.cumsum(counts)])
        self.g, self.l, self.n_local, self.flip = g, l, n_local, flip
        print(f'corpus: {len(self.files)} scans, {self.start[-1]} frames, {len(out)} scans excluded', flush=True)

    def __len__(self):
        return int(self.start[-1])

    def crop(self, img, ok, size, rng):
        H, W = img.shape; h, w = size; best = None
        for _ in range(8):                                  # most of the view should hold data
            y, x = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
            fill = ok[y:y + h, x:x + w].mean()
            if best is None or fill > best[0]:
                best = (fill, y, x)
            if fill > 0.7:
                break
        _, y, x = best
        c, m = img[y:y + h, x:x + w], ok[y:y + h, x:x + w]
        if self.flip and rng.random() < 0.5:                # chi is axis 0, its direction is not the same in the corpus and the validation frames
            c, m = c[::-1], m[::-1]
        c = np.clip(c, 0, 1) ** rng.uniform(0.8, 1.25)      # contrast jitter
        c = c + rng.normal(0, rng.uniform(0, 0.02), c.shape)
        return torch.from_numpy(np.ascontiguousarray(c, np.float32))[None], torch.from_numpy(np.ascontiguousarray(m))

    def __getitem__(self, i):
        rng = np.random.default_rng()
        k = int(np.searchsorted(self.start, i, 'right') - 1)
        with h5py.File(self.files[k], 'r') as h:
            v = h['images'][i - self.start[k]]; lm = float(h.attrs['log_max'])
        raw, ok = decode(v, lm)
        img = apply_contrast(raw, ok, DEPLOYED)
        gl = [self.crop(img, ok, self.g, rng) for _ in range(2)]
        lo = [self.crop(img, ok, self.l, rng)[0] for _ in range(self.n_local)]
        return torch.stack([a for a, _ in gl]), torch.stack([b for _, b in gl]), torch.stack(lo)


class Head(nn.Module):
    def __init__(self, dim, hidden, bottleneck, n_proto):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, bottleneck))
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck, n_proto, bias=False))

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1))


class Net(nn.Module):
    def __init__(self, kind, weights, args):
        super().__init__()
        self.bb = DinoV3Backbone(kind, grad_ckpt=args.grad_ckpt)
        if weights:
            self.bb.net.load_state_dict(torch.load(weights, map_location='cpu'), strict=True)
        self.img = Head(self.bb.num_features[3], 2048, 256, args.prototypes)
        self.pat = Head(self.bb.num_features[2], 2048, 256, args.proto_patch or args.prototypes)
        c = self.bb.num_features[2]
        self.register_buffer('pmu', torch.zeros(c)); self.register_buffer('psd', torch.ones(c))
        self.patch_std = bool(getattr(args, 'patch_stats', ''))
        if self.patch_std:
            st = torch.load(args.patch_stats, map_location='cpu'); self.pmu.copy_(st['mu']); self.psd.copy_(st['sd'])
        if getattr(args, 'patch_g1', False):
            with torch.no_grad():
                self.pat.last.parametrizations.weight.original0.fill_(1.0)
            self.pat.last.parametrizations.weight.original0.requires_grad_(False)

    def patch(self, x):
        """patch head on stride-16 features [..., C], standardised per channel with the stats of the public backbone when --patch_stats is set"""
        return self.pat((x.float() - self.pmu) / self.psd if self.patch_std else x)

    def features(self, x):
        o = self.bb(SimpleNamespace(tensors=x))
        return o[3].tensors.mean((2, 3)), o[2].tensors      # pooled stride 32, map at stride 16


class Center(nn.Module):
    """DINO centering, keeps the teacher from collapsing on one prototype"""
    def __init__(self, n):
        super().__init__()
        self.register_buffer('c', torch.zeros(n))

    @torch.no_grad()
    def update(self, t, m=0.9):
        self.c.mul_(m).add_(t.float().mean(0), alpha=1 - m)


@torch.no_grad()
def sinkhorn(logits, tt, iters):
    """Sinkhorn-Knopp equipartition of teacher logits [n, K]: every prototype gets the same total mass, every row sums to 1. -> [n, K]"""
    q = torch.exp((logits.float() - logits.float().max()) / tt).t()
    k, n = q.shape
    q /= q.sum()
    for _ in range(iters):
        q /= q.sum(1, keepdim=True); q /= k
        q /= q.sum(0, keepdim=True); q /= n
    return (q * n).t()


class Queue:
    """The last n teacher outputs, so that Sinkhorn sees more rows than one small micro-batch (SwAV does the same)."""
    def __init__(self, n, k, dev):
        self.q = torch.zeros(n, k, device=dev, dtype=torch.float16); self.filled = 0; self.pos = 0

    def with_batch(self, x):
        return torch.cat([self.q[:self.filled].float(), x.float()]) if self.filled else x.float()

    def push(self, x):
        x = x.detach().to(torch.float16)[-len(self.q):]
        idx = (torch.arange(len(x), device=x.device) + self.pos) % len(self.q)
        self.q[idx] = x; self.pos = (self.pos + len(x)) % len(self.q); self.filled = min(self.filled + len(x), len(self.q))


def koleo(x):
    """-mean log distance to the nearest other point of the L2-normalised x [n, C]: keeps the features spread out"""
    x = F.normalize(x.float(), dim=-1)
    d = torch.cdist(x, x) + torch.eye(len(x), device=x.device) * 1e6
    return -torch.log(d.min(1).values + 1e-8).mean()


def gram_loss(student_maps, anchor_maps, keep):
    """Gram anchoring: mean squared difference between the patch-to-patch cosine similarities of the student and of the anchor, over the patches in keep [n, h, w]"""
    out, n = student_maps.new_zeros(()), 0
    for i in range(len(keep)):
        if keep[i].sum() < 8:
            continue
        xs = F.normalize(student_maps[i].permute(1, 2, 0)[keep[i]].float(), dim=-1)
        xa = F.normalize(anchor_maps[i].permute(1, 2, 0)[keep[i]].float(), dim=-1)
        out = out + ((xs @ xs.t() - xa @ xa.t()) ** 2).mean(); n += 1
    return out / max(n, 1)


def soft_ce(teacher, student, t_s):
    return -(teacher * F.log_softmax(student.float() / t_s, -1)).sum(-1)


def block_mask(b, h, w, ratio, device):
    """per image bool [b, h/16, w/16]: which 16 x 16 blocks the student loses"""
    return torch.rand(b, h // 16, w // 16, device=device) < ratio


def cosine(step, total, lo, hi):
    return lo + 0.5 * (hi - lo) * (1 + math.cos(math.pi * min(step / max(total, 1), 1.0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True); p.add_argument('--corpus', nargs='+', required=True); p.add_argument('--exclude', default='')
    p.add_argument('--bb', default='dinov3_convnext_large')
    p.add_argument('--weights', default='/mnt/DATA/mlgidDETECT_DINO_HEATMAP/backbone_export/dinov3/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth')
    p.add_argument('--bs', type=int, default=8); p.add_argument('--accum', type=int, default=8); p.add_argument('--workers', type=int, default=8)
    p.add_argument('--steps', type=int, default=30000, help='optimizer steps'); p.add_argument('--head_steps', type=int, default=1500, help='stage 1, backbone frozen')
    p.add_argument('--lr_head', type=float, default=1e-3); p.add_argument('--lr_backbone', type=float, default=1e-5); p.add_argument('--wd', type=float, default=0.04)
    p.add_argument('--global_crop', type=int, nargs=2, default=[256, 512]); p.add_argument('--local_crop', type=int, nargs=2, default=[128, 256]); p.add_argument('--n_local', type=int, default=4)
    p.add_argument('--prototypes', type=int, default=8192); p.add_argument('--mask_ratio', type=float, default=0.3); p.add_argument('--patch_weight', type=float, default=1.0)
    p.add_argument('--t_student', type=float, default=0.1); p.add_argument('--t_teacher', type=float, default=0.05); p.add_argument('--ema', type=float, nargs=2, default=[0.996, 1.0])
    p.add_argument('--flip_chi', action='store_true', help='each view flipped along chi with probability 0.5')
    p.add_argument('--sinkhorn', type=int, default=0); p.add_argument('--queue_img', type=int, default=4096); p.add_argument('--queue_patch', type=int, default=8192)
    p.add_argument('--t_warmup', type=float, nargs=3, default=None, metavar=('LO', 'HI', 'STEPS')); p.add_argument('--koleo', type=float, default=0.0); p.add_argument('--gram', type=float, default=0.0)
    p.add_argument('--proto_patch', type=int, default=0); p.add_argument('--patch_stats', default='', help='pt file with mu, sd of the public backbone patch features: the patch head sees standardised features'); p.add_argument('--patch_g1', action='store_true', help='patch head prototype norm fixed to 1'); p.add_argument('--min_div', type=float, default=0.0, help='stop (exit 3) when the batch-mean teacher perplexity falls below this fraction of its ceiling min(prototypes, rows x mean single-row perplexity): rows that sharpen use different prototypes (fraction near 1), a collapse makes them share (fraction near 1/rows). Replaces --min_perp_frac, whose absolute level falls with sharpening')
    p.add_argument('--min_perp_frac', type=float, default=0.0); p.add_argument('--check_from', type=int, default=600)
    p.add_argument('--grad_ckpt', action='store_true'); p.add_argument('--device', default='cuda'); p.add_argument('--save_every', type=int, default=1000); p.add_argument('--log_every', type=int, default=20)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    dev = a.device; amp = dev == 'cuda'
    ds = Corpus(a.corpus, a.exclude, tuple(a.global_crop), tuple(a.local_crop), a.n_local, a.flip_chi)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0, pin_memory=amp)
    student = Net(a.bb, a.weights, a).to(dev)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    kp = a.proto_patch or a.prototypes
    ci, cp = Center(a.prototypes).to(dev), Center(kp).to(dev)
    qi, qp = Queue(a.queue_img, a.prototypes, dev), Queue(a.queue_patch, kp, dev)
    anchor = None
    if a.gram > 0:
        anchor = copy.deepcopy(student.bb).eval().requires_grad_(False)      # the public weights, taken before any resume
    heads = list(student.img.parameters()) + list(student.pat.parameters())
    opt = torch.optim.AdamW([dict(params=heads, lr=a.lr_head), dict(params=list(student.bb.parameters()), lr=0.0)], weight_decay=a.wd)
    step = 0
    ck = os.path.join(a.out, 'checkpoint.pth')
    if os.path.exists(ck):
        s = torch.load(ck, map_location='cpu', weights_only=False)
        student.load_state_dict(s['student']); teacher.load_state_dict(s['teacher']); opt.load_state_dict(s['opt']); ci.load_state_dict(s['ci']); cp.load_state_dict(s['cp']); step = s['step']
        print('resumed at step', step, flush=True)
    log = open(os.path.join(a.out, 'train.log'), 'a')

    def say(*x):
        m = ' '.join(str(v) for v in x); print(m, flush=True); log.write(m + '\n'); log.flush()

    def save():
        torch.save(dict(student=student.state_dict(), teacher=teacher.state_dict(), opt=opt.state_dict(), ci=ci.state_dict(), cp=cp.state_dict(), step=step), ck + '.tmp')
        os.replace(ck + '.tmp', ck)
        torch.save(teacher.bb.net.state_dict(), os.path.join(a.out, 'backbone_teacher.pth'))

    t0 = time.time(); seen = 0; acc = dict(img=0.0, pat=0.0, perp=0.0, pat_h=0.0, pat_perp=0.0, raw_i=0.0, row_i=0.0, raw_p=0.0, ko=0.0, gr=0.0, n=0, n_p=0)
    student.train(); student.bb.requires_grad_(False)
    micro = 0
    while step < a.steps:
        for gv, gm, lv in dl:
            frozen = step < a.head_steps
            student.bb.requires_grad_(not frozen)
            gv, gm, lv = gv.to(dev, non_blocking=True), gm.to(dev, non_blocking=True), lv.to(dev, non_blocking=True)
            b, h, w = gv.shape[0], gv.shape[-2], gv.shape[-1]
            G = gv.reshape(b * 2, 1, h, w); M = gm.reshape(b * 2, h, w); L = lv.reshape(-1, 1, *lv.shape[-2:])
            tt = a.t_teacher if a.t_warmup is None else a.t_warmup[0] + (a.t_warmup[1] - a.t_warmup[0]) * min(step / max(a.t_warmup[2], 1), 1.0)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                with torch.no_grad():
                    tp, tm = teacher.features(G)
                    ti = teacher.img(tp); tq = teacher.patch(tm.permute(0, 2, 3, 1))
                bm = block_mask(b * 2, h, w, a.mask_ratio, G.device)
                drop = bm.repeat_interleave(16, 1).repeat_interleave(16, 2)[:, None]
                sp, sm = student.features(G.masked_fill(drop, 0))
                si = student.img(sp)
                lp, _ = student.features(L)
                sl = student.img(lp)
                valid = F.avg_pool2d(M[:, None].float(), 16)[:, 0] > 0.5
                use = bm & valid
                sq = student.patch(sm.permute(0, 2, 3, 1)[use]) if use.any() else None
            if a.sinkhorn:
                pi = sinkhorn(qi.with_batch(ti), tt, a.sinkhorn)[-len(ti):].reshape(b, 2, -1); qi.push(ti)
            else:
                pi = F.softmax((ti.float() - ci.c) / tt, -1).reshape(b, 2, -1)
            raw_i = F.softmax(ti.float() / tt, -1).mean(0)
            acc['raw_i'] += float(torch.exp(-(raw_i * (raw_i + 1e-12).log()).sum()))
            acc['row_i'] += float(torch.exp(-(F.softmax(ti.float() / tt, -1) * F.log_softmax(ti.float() / tt, -1)).sum(-1)).mean())      # mean perplexity of single teacher rows = how sharp the targets are
            si_ = si.reshape(b, 2, -1); sl_ = sl.reshape(b, a.n_local, -1)
            li = (soft_ce(pi[:, 0], si_[:, 1], a.t_student).mean() + soft_ce(pi[:, 1], si_[:, 0], a.t_student).mean()) / 2
            if a.n_local:
                li = (li + sum(soft_ce(pi[:, k], sl_[:, j], a.t_student).mean() for k in range(2) for j in range(a.n_local)) / (2 * a.n_local)) / 2
            lq = torch.zeros((), device=G.device)
            if sq is not None:
                tqu = tq[use]
                if a.sinkhorn:
                    pq = sinkhorn(qp.with_batch(tqu), tt, a.sinkhorn)[-len(tqu):]; qp.push(tqu[torch.randperm(len(tqu), device=tqu.device)[:1024]])
                else:
                    pq = F.softmax((tqu.float() - cp.c) / tt, -1)
                lq = soft_ce(pq, sq, a.t_student).mean()
                raw_p = F.softmax(tqu.float() / tt, -1).mean(0)
                acc['raw_p'] += float(torch.exp(-(raw_p * (raw_p + 1e-12).log()).sum())); acc['n_p'] += 1
                mean_q = pq.mean(0)
                acc['pat_h'] += float(-(pq * (pq + 1e-12).log()).sum(-1).mean())           # the patch loss cannot go below this: teacher targets near ln(prototypes) leave the student nothing to learn
                acc['pat_perp'] += float(torch.exp(-(mean_q * (mean_q + 1e-12).log()).sum()))
            loss = li + a.patch_weight * lq
            if a.koleo > 0:
                groups = [sp.reshape(b, 2, -1)[:, k] for k in range(2)] + [lp.reshape(b, a.n_local, -1)[:, k] for k in range(a.n_local)]
                ko = sum(koleo(g) for g in groups) / len(groups)
                loss = loss + a.koleo * ko; acc['ko'] += ko.item()
            if anchor is not None:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    am = anchor(SimpleNamespace(tensors=G))[2].tensors
                gr = gram_loss(sm, am, ~bm & valid)
                loss = loss + a.gram * gr; acc['gr'] += gr.item()
            (loss / a.accum).backward()
            if not a.sinkhorn:
                ci.update(ti); cp.update(tq[use] if sq is not None else tq.reshape(-1, tq.shape[-1]))
            mean_p = pi.reshape(-1, pi.shape[-1]).mean(0)
            acc['img'] += li.item(); acc['pat'] += lq.item(); acc['n'] += 1
            acc['perp'] += float(torch.exp(-(mean_p * (mean_p + 1e-12).log()).sum()))      # how many prototypes are in use: a collapse shows here first
            seen += b; micro += 1
            if micro % a.accum:
                continue
            frac = max(step - a.head_steps, 0) / max(a.steps - a.head_steps, 1)
            opt.param_groups[0]['lr'] = cosine(step, a.steps, a.lr_head * 0.01, a.lr_head)
            ramp = min(max(step - a.head_steps, 0) / 500, 1.0)                              # linear ramp after stage 1
            opt.param_groups[1]['lr'] = 0.0 if frozen else ramp * cosine(frac, 1.0, a.lr_backbone * 0.1, a.lr_backbone)
            nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            m = cosine(step, a.steps, a.ema[1], a.ema[0])
            with torch.no_grad():
                torch._foreach_mul_(list(teacher.parameters()), m)
                torch._foreach_add_(list(teacher.parameters()), [q.detach() for q in student.parameters()], alpha=1 - m)
            step += 1
            if step % a.log_every == 0:
                n = max(acc['n'], 1); dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 2**30 if amp else 0
                say(f"step {step}/{a.steps} {'heads' if frozen else 'all'} img {acc['img'] / n:.3f} patch {acc['pat'] / n:.3f} patch_teacher_H {acc['pat_h'] / n:.3f} patch_prototypes {acc['pat_perp'] / n:.0f} prototypes {acc['perp'] / n:.0f} raw_img {acc['raw_i'] / n:.0f} raw_patch {acc['raw_p'] / max(acc['n_p'], 1):.0f} tt {tt:.3f} koleo {acc['ko'] / n:.3f} gram {acc['gr'] / n:.4f} lr_bb {opt.param_groups[1]['lr']:.2e} {seen / dt:.2f} frames/s peak {mem:.1f} GB")
                if a.min_div > 0 and step >= a.check_from and acc['raw_i'] / n < a.min_div * min(a.prototypes, 2 * a.bs * acc['row_i'] / n):
                    save(); say(f"PRETRAIN STOPPED collapse at step {step}: the {2 * a.bs} teacher rows of a micro-batch share prototypes: batch-mean perplexity {acc['raw_i'] / n:.0f} of at most {min(a.prototypes, 2 * a.bs * acc['row_i'] / n):.0f}")
                    sys.exit(3)
                if a.min_perp_frac > 0 and step >= a.check_from and (acc['raw_i'] / n < a.min_perp_frac * a.prototypes or acc['raw_p'] / max(acc['n_p'], 1) < a.min_perp_frac * kp):
                    save(); say(f"PRETRAIN STOPPED collapse at step {step}: raw teacher uses {acc['raw_i'] / n:.0f} of {a.prototypes} image and {acc['raw_p'] / max(acc['n_p'], 1):.0f} of {kp} patch prototypes")
                    sys.exit(3)
                acc = dict(img=0.0, pat=0.0, perp=0.0, pat_h=0.0, pat_perp=0.0, raw_i=0.0, row_i=0.0, raw_p=0.0, ko=0.0, gr=0.0, n=0, n_p=0)
            if step % a.save_every == 0:
                save()
            if step >= a.steps:
                break
    save()
    say('PRETRAIN DONE', step)


if __name__ == '__main__':
    main()
