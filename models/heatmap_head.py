"""Heatmap-first detector: SSL swin backbone (optionally frozen) + FPN/UNet-style decoder.

Separate from the DINO code paths. Output is a dense map at stride `out_stride` (default 2):
    heat: (B, 2, H/s, W/s)  logits, class 0 = segment, 1 = ring   (peak = box centre)
    reg : (B, 4, H/s, W/s)  [dx, dy, log w, log h]
          dx, dy = true centre minus cell centre, in cell units; w, h in pixels (log).
Boxes in this repo are axis-aligned (q x chi), so there is no angle channel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dino.swin_transformer import build_swin_transformer
from util.misc import NestedTensor, clean_state_dict


def _gn(c):
    return nn.GroupNorm(min(16, c), c)


def _cbr(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), _gn(o), nn.ReLU(inplace=True))


def _cbd(i, o, d):
    """3x3 conv with dilation d (wider context at the same cost)"""
    return nn.Sequential(nn.Conv2d(i, o, 3, 1, d, dilation=d, bias=False), _gn(o), nn.ReLU(inplace=True))


CHAN_N = {'he': 1, 'he_mask': 2, 'full': 4, 'contrast': 4}


def build_channels(img, mask, mode):
    """Side channels for the trainable stem (the frozen swin always sees channel 0 only).
    img [B,H,W] (the HE image), mask [B,H,W] bool, True = valid pixel. Returns [B,C,H,W].
      he       : img unchanged (the original single-channel pipeline, byte-identical)
      he_mask  : [HE, valid-pixel mask]
      full     : [HE, B1 = HE - per-q-column masked median over chi, B2 = that column median, mask]
      contrast : img is already the [B,3,H,W] stack (log+HE, plain log, log+CLAHE) -> [stack, mask]
    Invalid pixels are 0 in every channel except the mask (same definition as the multi-channel branch)."""
    if mode == 'he':
        return img[:, None]
    m = mask.bool()
    if mode == 'contrast':
        return torch.cat([img, m.to(img.dtype)[:, None]], 1)
    he = img.masked_fill(~m, 0.)
    mf = m.to(img.dtype)
    if mode == 'he_mask':
        return torch.stack([he, mf], 1)
    assert mode == 'full', mode
    x = img.masked_fill(~m, float('inf'))
    srt, _ = torch.sort(x, dim=1)                              # per column, ascending; invalid sinks to the bottom
    n = m.sum(dim=1)                                           # valid count per column [B,W]
    idx = ((n - 1) // 2).clamp(min=0)
    med = torch.gather(srt, 1, idx[:, None, :]).expand_as(img)  # median of the valid values
    med = med.masked_fill((n == 0)[:, None, :], 0.).masked_fill(~m, 0.)
    b1 = (img - med).masked_fill(~m, 0.)
    return torch.stack([he, b1, med, mf], 1)


class HeatmapNet(nn.Module):
    def __init__(self, backbone_ckpt=None, backbone_prefix='', freeze_backbone=True,
                 out_stride=2, dim=128, window_size_h=48, window_size_w=6, amp_backbone=False,
                 chan_mode='he', zero_invalid=False, tower_ch=64, tower_depth=2, stem_ch=32, ring_head_stride=0,
                 drop_path_rate=0.0, grad_ckpt=False):
        super().__init__()
        self.zero_invalid = zero_invalid       # invalid (masked) pixels are set to 0 in EVERY input incl. the swin's (sim images are gray there, eval files 0)
        self.chan_mode = chan_mode             # extra input channels feed the stem only; backbone input is unchanged
        in_ch = CHAN_N[chan_mode]
        self.amp_backbone = amp_backbone       # bf16 autocast for the swin only; FPN/head stay fp32
        assert out_stride in (1, 2, 4)
        self.out_stride = out_stride
        self.freeze_backbone = freeze_backbone
        self.backbone = build_swin_transformer(
            'swin_L_384_22k', pretrain_img_size=384, out_indices=(0, 1, 2, 3), dilation=False,
            use_checkpoint=bool(grad_ckpt), drop_path_rate=drop_path_rate, window_size_h=window_size_h, window_size_w=window_size_w,
            patch_size_h=4, patch_size_w=4, in_chans=1)
        if backbone_ckpt:
            sd = torch.load(backbone_ckpt, map_location='cpu')
            sd = clean_state_dict(sd.get('model', sd))
            sd = {k[len(backbone_prefix):]: v for k, v in sd.items()
                  if k.startswith(backbone_prefix) and 'head' not in k[len(backbone_prefix):]}
            print('[heatmap] backbone load:', self.backbone.load_state_dict(sd, strict=False), flush=True)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        chs = self.backbone.num_features            # [192, 384, 768, 1536]
        self.lat = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in chs])
        self.smooth = nn.ModuleList([_cbr(dim, dim) for _ in chs[:3]])
        sc, tc = stem_ch, tower_ch                  # defaults (32 / 64 / depth 2) = the original head, same state-dict keys
        self.stem = nn.Sequential(_cbr(in_ch, sc, 3, 1 if out_stride == 1 else 2), _cbr(sc, sc))
        self.fuse = _cbr(dim + sc, dim)
        self.tower_h = nn.Sequential(_cbr(dim, tc), *[_cbr(tc, tc) for _ in range(tower_depth - 1)])
        self.tower_r = nn.Sequential(_cbr(dim, tc), *[_cbr(tc, tc) for _ in range(tower_depth - 1)])
        self.heat = nn.Conv2d(tc, 2, 1)
        self.reg = nn.Conv2d(tc, 4, 1)
        nn.init.constant_(self.heat.bias, -2.19)    # prior p = 0.1 (CenterNet)
        nn.init.zeros_(self.reg.weight); nn.init.zeros_(self.reg.bias)
        # typical box ~ 10 px: start log w/h there
        with torch.no_grad():
            self.reg.bias[2:] = 2.3
        # optional extra RING head on a coarse merged FPN map (stride 8 or 16): rings are as tall as the image but only a few px
        # wide, so a coarse map with a large receptive field (dilated convs) sees a whole ring at once. Rings only (1 class).
        self.ring_head_stride = ring_head_stride
        if ring_head_stride:
            assert ring_head_stride in (8, 16), ring_head_stride
            self.ring_level = {8: 1, 16: 2}[ring_head_stride]
            self.ring_tower = nn.Sequential(_cbd(dim, tc, 1), _cbd(tc, tc, 2), _cbd(tc, tc, 4), _cbd(tc, tc, 1))
            self.ring_heat = nn.Conv2d(tc, 1, 1)
            self.ring_reg = nn.Conv2d(tc, 4, 1)
            nn.init.constant_(self.ring_heat.bias, -2.19)
            nn.init.zeros_(self.ring_reg.weight); nn.init.zeros_(self.ring_reg.bias)
            with torch.no_grad():
                self.ring_reg.bias[2:] = 2.3

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, img, mask=None):
        """img [B,1,H,W] (HE image; [B,3,H,W] contrast stack when chan_mode == 'contrast');
        mask [B,H,W] bool (True = valid), needed unless chan_mode == 'he'. The swin sees channel 0 only."""
        B, _, H, W = img.shape
        if self.zero_invalid:
            assert mask is not None, 'zero_invalid needs the valid-pixel mask'
            img = img.masked_fill(~mask.to(img.device).bool()[:, None], 0.)
        if self.chan_mode == 'he':
            side = img
        elif self.chan_mode == 'contrast':
            assert mask is not None and img.shape[1] == 3
            side = build_channels(img, mask.to(img.device), 'contrast')
            img = img[:, :1]
        else:
            assert mask is not None, f"chan_mode '{self.chan_mode}' needs the valid-pixel mask"
            side = build_channels(img[:, 0], mask.to(img.device), self.chan_mode)
        mask = torch.zeros(B, H, W, dtype=torch.bool, device=img.device)
        ac = torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(self.amp_backbone and img.is_cuda))
        if self.freeze_backbone:
            with torch.no_grad(), ac:
                feats = self.backbone(NestedTensor(img, mask))
        else:
            with ac:
                feats = self.backbone(NestedTensor(img, mask))
        c = [feats[i].tensors.float() for i in range(4)]
        p = self.lat[3](c[3])
        mid = None
        for i in (2, 1, 0):
            p = self.lat[i](c[i]) + F.interpolate(p, size=c[i].shape[-2:], mode='nearest')
            p = self.smooth[i](p)
            if self.ring_head_stride and i == self.ring_level:
                mid = p                                  # merged map at the ring head's stride (carries levels deeper than itself)
        # p is at stride 4; bring to out_stride and fuse with the image stem
        s = self.stem(side)
        p = F.interpolate(p, size=s.shape[-2:], mode='bilinear', align_corners=False)
        if self.out_stride == 4:
            p = F.interpolate(p, size=(H // 4, W // 4), mode='bilinear', align_corners=False)
            s = F.adaptive_avg_pool2d(s, p.shape[-2:])
        x = self.fuse(torch.cat([p, s], 1))
        out = dict(heat=self.heat(self.tower_h(x)), reg=self.reg(self.tower_r(x)))
        if self.ring_head_stride:
            sc = self.ring_head_stride
            if mid.shape[-2:] != (H // sc, W // sc):
                mid = F.interpolate(mid, size=(H // sc, W // sc), mode='bilinear', align_corners=False)
            t = self.ring_tower(mid)
            out['ring_heat'] = self.ring_heat(t); out['ring_reg'] = self.ring_reg(t)
        return out


RING_DECODE_MODES = ('fine', 'coarse_rings', 'union')
N_RING_PEAKS = 75


@torch.no_grad()
def decode(out, stride, num_select=225, score_floor=0.0, mode=None):
    """Peak decoding. Without a ring head this is exactly decode_fine. With one (out has 'ring_heat'), `mode` (default env
    HM_RING_DECODE, else 'coarse_rings') selects:
      fine          segments AND rings from the fine map (the ring head is ignored; the plain model's decode)
      coarse_rings  segments from the fine map, rings ONLY from the coarse ring head
      union         fine map (both classes) + rings from the coarse head, the shared class-aware NMS removes duplicates
    Returns per-image lists of (boxes_xyxy_px [N,4], scores [N], labels [N]) sorted by score, at most num_select."""
    if 'ring_heat' not in out:
        return decode_fine(out, stride, num_select, score_floor)
    import os
    mode = mode or os.environ.get('HM_RING_DECODE', 'coarse_rings')
    assert mode in RING_DECODE_MODES, mode
    if mode == 'fine':
        return decode_fine(out, stride, num_select, score_floor)
    sc = stride * out['heat'].shape[-1] // out['ring_heat'].shape[-1]            # stride of the coarse map
    rings = decode_fine(dict(heat=out['ring_heat'], reg=out['ring_reg']), sc, N_RING_PEAKS, score_floor)
    if mode == 'coarse_rings':
        base = decode_fine(dict(heat=out['heat'][:, :1], reg=out['reg']), stride, num_select, score_floor)
    else:
        base = decode_fine(out, stride, num_select, score_floor)
    res = []
    for (b0, s0, c0), (b1, s1, _) in zip(base, rings):
        b = torch.cat([b0, b1]); s_ = torch.cat([s0, s1]); c = torch.cat([c0, torch.ones_like(s1, dtype=c0.dtype)])
        o = s_.argsort(descending=True)[:num_select]
        res.append((b[o], s_[o], c[o]))
    return res


@torch.no_grad()
def decode_fine(out, stride, num_select=225, score_floor=0.0):
    """3x3 max-pool peak picking (replaces NMS). Returns per-image lists of
    (boxes_xyxy_px [N,4], scores [N], labels [N]) sorted by score."""
    heat = out['heat'].sigmoid()
    B, C, h, w = heat.shape
    keep = (F.max_pool2d(heat, 3, 1, 1) == heat).float()
    peaks = heat * keep
    res = []
    for b in range(B):
        sc, idx = peaks[b].reshape(-1).topk(min(num_select, C * h * w))
        m = sc > score_floor
        sc, idx = sc[m], idx[m]
        cls = idx // (h * w)
        yx = idx % (h * w)
        ys, xs = yx // w, yx % w
        r = out['reg'][b][:, ys, xs]                         # [4, N]
        cx = (xs.float() + 0.5 + r[0]) * stride
        cy = (ys.float() + 0.5 + r[1]) * stride
        bw, bh = r[2].exp(), r[3].exp()
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)
        res.append((boxes, sc, cls))
    return res
