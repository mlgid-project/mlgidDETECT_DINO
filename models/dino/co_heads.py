# Co-DINO / Co-DETR collaborative auxiliary head (arXiv:2211.12860), Phase-0 MVP.
# docs/CO_DINO_INVESTIGATION.md.
#
# A DETR decoder uses one-to-one Hungarian matching -> each GT peak supervises exactly
# one query -> the *encoder* features get sparse gradient. This module bolts ONE
# conventional dense detection head onto the encoder's multi-scale feature pyramid with
# a ONE-TO-MANY (FCOS center-sampling) label assignment, so *every* location near a GT
# peak receives a direct classification + box gradient. That dense encoder supervision is
# the lever we want for faint/high-q peak SENSITIVITY (the recall ceiling).
#
# STRICTLY TRAINING-ONLY: the head is only built/called under `self.training` and its
# output key never enters the ONNX export whitelist -> zero inference/deploy cost, the
# exported graph is byte-identical to baseline. See docs S5.
#
# MVP simplifications (vs full Co-DETR / FCOS), all noted for later phases:
#  - K=1 head (paper K=1 ~= 2/3 of the K=2 gain), FCOS center-sampling assigner only
#    (robust to our extreme-aspect arcs; ATSS anchor-IoU would starve thin segments).
#  - Box is predicted as an ABSOLUTE normalized cxcywh per location (sigmoid), not FCOS
#    ltrb-from-location. Assignment is still center-sampling; this keeps the coordinate
#    math trivial and is adequate for encoder supervision (the point is dense gradient).
#  - No per-level object-size ranges and no centerness branch (Phase-1 refinements).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy
from .utils import sigmoid_focal_loss


def memory_to_pyramid(memory, spatial_shapes, level_start_index):
    """Inverse of the encoder input flatten (deformable_transformer.py forward): split
    the token sequence `memory (bs, sum_hw, C)` back into per-level 2-D maps
    `List[(bs, C, H_l, W_l)]` using the already-in-scope shapes/offsets."""
    bs, _, C = memory.shape
    feats = []
    for lvl in range(spatial_shapes.shape[0]):
        H = int(spatial_shapes[lvl, 0]); W = int(spatial_shapes[lvl, 1])
        start = int(level_start_index[lvl])
        f = memory[:, start:start + H * W].transpose(1, 2).reshape(bs, C, H, W)
        feats.append(f)
    return feats


class CoHeads(nn.Module):
    """One dense detector head, shared across the pyramid levels (FCOS-style weight
    sharing). Consumes the encoder pyramid, emits per-level dense class logits + boxes."""

    def __init__(self, d_model=256, num_classes=2, num_levels=4, hidden=None):
        super().__init__()
        hidden = hidden or d_model
        self.num_classes = num_classes
        ng = math.gcd(hidden, 32)          # 32 groups at d_model=256; always divides hidden
        self.stem = nn.Sequential(
            nn.Conv2d(d_model, hidden, 3, padding=1), nn.GroupNorm(ng, hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(ng, hidden), nn.ReLU(inplace=True),
        )
        self.cls = nn.Conv2d(hidden, num_classes, 3, padding=1)
        self.box = nn.Conv2d(hidden, 4, 3, padding=1)
        # focal-loss prior on the cls bias (RetinaNet init): start pessimistic so the
        # dense background does not dominate the first steps.
        prior = 0.01
        nn.init.constant_(self.cls.bias, -float(torch.log(torch.tensor((1 - prior) / prior))))

    def forward(self, pyramid):
        cls_list, box_list = [], []
        for f in pyramid:
            x = self.stem(f)
            cls_list.append(self.cls(x))              # (bs, num_classes, H, W)
            box_list.append(self.box(x).sigmoid())    # (bs, 4, H, W) normalized cxcywh in (0,1)
        return {'cls': cls_list, 'box': box_list}


def _giou_loss_xyxy(pred, tgt, eps=1e-7):
    """Elementwise 1 - GIoU for matched (P,4) xyxy pairs."""
    x1 = torch.max(pred[:, 0], tgt[:, 0]); y1 = torch.max(pred[:, 1], tgt[:, 1])
    x2 = torch.min(pred[:, 2], tgt[:, 2]); y2 = torch.min(pred[:, 3], tgt[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    ap = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    at = (tgt[:, 2] - tgt[:, 0]).clamp(min=0) * (tgt[:, 3] - tgt[:, 1]).clamp(min=0)
    union = ap + at - inter + eps
    iou = inter / union
    cx1 = torch.min(pred[:, 0], tgt[:, 0]); cy1 = torch.min(pred[:, 1], tgt[:, 1])
    cx2 = torch.max(pred[:, 2], tgt[:, 2]); cy2 = torch.max(pred[:, 3], tgt[:, 3])
    carea = (cx2 - cx1) * (cy2 - cy1) + eps
    giou = iou - (carea - union) / carea
    return 1 - giou


class CoCriterion(nn.Module):
    """Dense one-to-many (FCOS center-sampling) loss for the collaborative aux head.
    Returns loss_co_cls / loss_co_bbox / loss_co_giou, which the engine's existing
    weighted sum picks up via weight_dict (no engine change)."""

    def __init__(self, num_classes=2, focal_alpha=0.25, center_radius=1.5):
        super().__init__()
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.center_radius = center_radius

    @torch.no_grad()
    def _assign(self, loc, sx, sy, gt_boxes):
        """Center-sampling assignment for one image at one flattened multi-level grid.
        loc: (N,2) normalized location centers; sx,sy: (N,) per-location level stride;
        gt_boxes: (G,4) cxcywh normalized. Returns matched gt index per location (-1 = bg)."""
        N = loc.shape[0]; G = gt_boxes.shape[0]
        gcx, gcy, gw, gh = gt_boxes.unbind(-1)
        gx1 = gcx - gw / 2; gy1 = gcy - gh / 2; gx2 = gcx + gw / 2; gy2 = gcy + gh / 2
        lx = loc[:, 0:1]; ly = loc[:, 1:2]                      # (N,1)
        r = self.center_radius
        # center region = box  intersect  [gc +/- r*stride]  (broadcast to (N,G))
        cxr1 = torch.maximum(gx1[None, :], gcx[None, :] - r * sx[:, None])
        cxr2 = torch.minimum(gx2[None, :], gcx[None, :] + r * sx[:, None])
        cyr1 = torch.maximum(gy1[None, :], gcy[None, :] - r * sy[:, None])
        cyr2 = torch.minimum(gy2[None, :], gcy[None, :] + r * sy[:, None])
        inside = (lx >= cxr1) & (lx <= cxr2) & (ly >= cyr1) & (ly <= cyr2)   # (N,G)
        # ambiguity: a location inside several GTs takes the smallest-area GT (FCOS rule)
        areas = (gw * gh).clamp(min=1e-8)[None, :].expand(N, G).clone()
        areas[~inside] = float('inf')
        min_area, matched = areas.min(dim=1)
        matched[~torch.isfinite(min_area)] = -1
        return matched

    def forward(self, co_head_outputs, targets):
        cls_list = co_head_outputs['cls']; box_list = co_head_outputs['box']
        bs = cls_list[0].shape[0]; nc = self.num_classes
        device = cls_list[0].device

        # flatten predictions across levels + build matching location grid / strides
        cls_pred, box_pred, locs, strides = [], [], [], []
        for cls_l, box_l in zip(cls_list, box_list):
            b, _, H, W = cls_l.shape
            ys = (torch.arange(H, device=device) + 0.5) / H
            xs = (torch.arange(W, device=device) + 0.5) / W
            gy, gx = torch.meshgrid(ys, xs, indexing='ij')
            locs.append(torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1))    # (HW,2)
            strides.append(torch.stack([torch.full((H * W,), 1.0 / W, device=device),
                                        torch.full((H * W,), 1.0 / H, device=device)], dim=1))
            cls_pred.append(cls_l.permute(0, 2, 3, 1).reshape(b, H * W, nc))
            box_pred.append(box_l.permute(0, 2, 3, 1).reshape(b, H * W, 4))
        cls_pred = torch.cat(cls_pred, dim=1)          # (bs, N, nc)
        box_pred = torch.cat(box_pred, dim=1)          # (bs, N, 4) cxcywh
        loc = torch.cat(locs, dim=0)                   # (N, 2)
        stride = torch.cat(strides, dim=0)             # (N, 2)
        N = loc.shape[0]

        cls_tgt = torch.zeros(bs, N, nc, device=device)
        pos_mask = torch.zeros(bs, N, dtype=torch.bool, device=device)
        box_tgt = torch.zeros(bs, N, 4, device=device)
        for b in range(bs):
            gt = targets[b]['boxes']
            lab = targets[b]['labels']
            if gt.numel() == 0:
                continue
            matched = self._assign(loc, stride[:, 0], stride[:, 1], gt)    # (N,)
            pos = matched >= 0
            if pos.any():
                pidx = pos.nonzero(as_tuple=True)[0]
                m = matched[pidx]
                cls_tgt[b, pidx, lab[m]] = 1.0
                box_tgt[b, pidx] = gt[m]
                pos_mask[b, pidx] = True

        num_pos = pos_mask.sum().clamp(min=1).float()
        loss_cls = sigmoid_focal_loss(cls_pred.reshape(-1, nc), cls_tgt.reshape(-1, nc),
                                      num_pos, alpha=self.focal_alpha)
        if pos_mask.any():
            pb = box_pred[pos_mask]; tb = box_tgt[pos_mask]
            loss_l1 = F.l1_loss(pb, tb, reduction='none').sum() / num_pos
            loss_giou = _giou_loss_xyxy(box_cxcywh_to_xyxy(pb), box_cxcywh_to_xyxy(tb)).sum() / num_pos
        else:
            loss_l1 = box_pred.sum() * 0.0
            loss_giou = box_pred.sum() * 0.0
        return {'loss_co_cls': loss_cls, 'loss_co_bbox': loss_l1, 'loss_co_giou': loss_giou}
