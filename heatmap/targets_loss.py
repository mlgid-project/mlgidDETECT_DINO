"""Label conversion (xyxy boxes -> centre heatmap + regression targets) and the loss."""
import torch
import torch.nn.functional as F

R = 10          # Gaussian window half-size in cells


def gauss_sigma(w_px, h_px, is_ring):
    """Small, size-scaled sigma (px): keeps peaks 4 px apart separable. Rings are tall, cap sigma_y."""
    sx = (w_px / 6).clamp(1.0, 3.0)
    sy = torch.where(is_ring, torch.full_like(h_px, 4.0), (h_px / 6).clamp(1.0, 4.0))
    return sx, sy


@torch.no_grad()
def build_targets(boxes_xyxy, labels, H, W, stride):
    """boxes_xyxy [N,4] px, labels [N] (0 seg / 1 ring) -> dict of dense targets on the stride grid.
    heat [2,h,w]; reg [4,h,w]; wgt [h,w] (1 where regression is supervised).
    Regression is supervised on the 3x3 cells around each centre cell, each assigned to the nearest GT."""
    dev = boxes_xyxy.device
    h, w = H // stride, W // stride
    heat = torch.zeros(2, h, w, device=dev)
    reg = torch.zeros(4, h, w, device=dev)
    wgt = torch.zeros(h, w, device=dev)
    N = boxes_xyxy.shape[0]
    if N == 0:
        return heat, reg, wgt
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
    bw = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=1.0)
    bh = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=1.0)
    ring = labels.bool()
    ix = (cx / stride).floor().clamp(0, w - 1).long()
    iy = (cy / stride).floor().clamp(0, h - 1).long()
    sx, sy = gauss_sigma(bw, bh, ring)
    d = torch.arange(-R, R + 1, device=dev)
    dy, dx = torch.meshgrid(d, d, indexing='ij')              # [K,K]
    gx = (ix[:, None, None] + dx)                             # [N,K,K]
    gy = (iy[:, None, None] + dy)
    valid = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
    g = torch.exp(-((dx * stride) ** 2 / (2 * sx[:, None, None] ** 2)
                    + (dy * stride) ** 2 / (2 * sy[:, None, None] ** 2)))
    g = torch.where(valid, g, torch.zeros_like(g))
    flat = (labels[:, None, None] * h * w + gy.clamp(0, h - 1) * w + gx.clamp(0, w - 1)).reshape(-1)
    heat.view(-1).scatter_reduce_(0, flat, g.reshape(-1), reduce='amax', include_self=True)
    # regression: 3x3 around each centre cell, nearest GT wins
    offs = torch.tensor([(ox, oy) for oy in (-1, 0, 1) for ox in (-1, 0, 1)], device=dev)   # [9,2]
    nx = ix[None] + offs[:, 0:1]                                   # [9,N]
    ny = iy[None] + offs[:, 1:2]
    ok = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
    gi = torch.arange(N, device=dev)[None].expand(9, N)
    cell = (ny * w + nx)[ok]
    gidx = gi[ok]
    dist = ((cx[gidx] / stride - (nx[ok] + 0.5)) ** 2 + (cy[gidx] / stride - (ny[ok] + 0.5)) ** 2)
    best = torch.full((h * w,), float('inf'), device=dev).scatter_reduce(0, cell, dist, 'amin')
    win = dist <= best[cell]
    arg = torch.full((h * w,), -1, device=dev, dtype=torch.long)
    arg[cell[win]] = gidx[win]
    arg = arg.view(h, w)
    m = arg >= 0
    a = arg[m]
    yy, xx = torch.nonzero(m, as_tuple=True)
    reg[0][m] = cx[a] / stride - (xx.float() + 0.5)
    reg[1][m] = cy[a] / stride - (yy.float() + 0.5)
    reg[2][m] = bw[a].log()
    reg[3][m] = bh[a].log()
    wgt[m] = 1.0
    return heat, reg, wgt


def focal_loss(logits, gt, alpha=2.0, beta=4.0):
    """CenterNet penalty-reduced focal loss; positives are cells with gt == 1."""
    p = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pl = torch.log(p) * (1 - p) ** alpha * pos
    nl = torch.log(1 - p) * p ** alpha * (1 - gt) ** beta * neg
    npos = pos.sum().clamp(min=1.0)
    return -(pl.sum() + nl.sum()) / npos


def heatmap_loss(out, heat_gt, reg_gt, wgt, reg_coef=1.0):
    lh = focal_loss(out['heat'], heat_gt)
    n = wgt.sum().clamp(min=1.0)
    l1 = (F.l1_loss(out['reg'], reg_gt, reduction='none') * wgt[:, None]).sum() / n / 4
    # offsets are in cell units, sizes in log-px: report separately
    return lh + reg_coef * l1 * 4, dict(loss_heat=lh.item(), loss_reg=l1.item())
