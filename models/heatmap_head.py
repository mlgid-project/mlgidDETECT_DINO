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


class HeatmapNet(nn.Module):
    def __init__(self, backbone_ckpt=None, backbone_prefix='', freeze_backbone=True,
                 out_stride=2, dim=128, window_size_h=48, window_size_w=6):
        super().__init__()
        assert out_stride in (1, 2, 4)
        self.out_stride = out_stride
        self.freeze_backbone = freeze_backbone
        self.backbone = build_swin_transformer(
            'swin_L_384_22k', pretrain_img_size=384, out_indices=(0, 1, 2, 3), dilation=False,
            use_checkpoint=False, window_size_h=window_size_h, window_size_w=window_size_w,
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
        self.stem = nn.Sequential(_cbr(1, 32, 3, 1 if out_stride == 1 else 2), _cbr(32, 32))
        if out_stride == 1:
            self.stem = nn.Sequential(_cbr(1, 32), _cbr(32, 32))
        self.fuse = _cbr(dim + 32, dim)
        self.tower_h = nn.Sequential(_cbr(dim, 64), _cbr(64, 64))
        self.tower_r = nn.Sequential(_cbr(dim, 64), _cbr(64, 64))
        self.heat = nn.Conv2d(64, 2, 1)
        self.reg = nn.Conv2d(64, 4, 1)
        nn.init.constant_(self.heat.bias, -2.19)    # prior p = 0.1 (CenterNet)
        nn.init.zeros_(self.reg.weight); nn.init.zeros_(self.reg.bias)
        # typical box ~ 10 px: start log w/h there
        with torch.no_grad():
            self.reg.bias[2:] = 2.3

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, img):
        B, _, H, W = img.shape
        mask = torch.zeros(B, H, W, dtype=torch.bool, device=img.device)
        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.backbone(NestedTensor(img, mask))
        else:
            feats = self.backbone(NestedTensor(img, mask))
        c = [feats[i].tensors for i in range(4)]
        p = self.lat[3](c[3])
        for i in (2, 1, 0):
            p = self.lat[i](c[i]) + F.interpolate(p, size=c[i].shape[-2:], mode='nearest')
            p = self.smooth[i](p)
        # p is at stride 4; bring to out_stride and fuse with the image stem
        s = self.stem(img)
        p = F.interpolate(p, size=s.shape[-2:], mode='bilinear', align_corners=False)
        if self.out_stride == 4:
            p = F.interpolate(p, size=(H // 4, W // 4), mode='bilinear', align_corners=False)
            s = F.adaptive_avg_pool2d(s, p.shape[-2:])
        x = self.fuse(torch.cat([p, s], 1))
        return dict(heat=self.heat(self.tower_h(x)), reg=self.reg(self.tower_r(x)))


@torch.no_grad()
def decode(out, stride, num_select=225, score_floor=0.0):
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
