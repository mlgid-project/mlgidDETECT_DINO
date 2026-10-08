"""Label conversion (xyxy boxes -> centre heatmap + regression targets) and the loss."""
import torch
import torch.nn.functional as F

R = 10          # Gaussian window half-size in cells
RIDGE_CAP = 40.0   # px, upper bound for the ring ridge sigma_y


def gauss_sigma(w_px, h_px, is_ring):
    """Small, size-scaled sigma (px): keeps peaks 4 px apart separable. Rings are tall, cap sigma_y."""
    sx = (w_px / 6).clamp(1.0, 3.0)
    sy = torch.where(is_ring, torch.full_like(h_px, 4.0), (h_px / 6).clamp(1.0, 4.0))
    return sx, sy


@torch.no_grad()
def build_targets(boxes_xyxy, labels, H, W, stride, ring_mode='legacy'):
    """boxes_xyxy [N,4] px, labels [N] (0 seg / 1 ring) -> dict of dense targets on the stride grid.
    heat [2,h,w]; reg [4,h,w]; wgt [h,w] (1 where regression is supervised).
    Regression is supervised on the 3x3 cells around each centre cell, each assigned to the nearest GT.
    ring_mode 'legacy': every box gets the same small Gaussian (rings: sigma_y capped at 4 px).
    ring_mode 'ridge': ring boxes get a tall ridge target (sigma_y = h/6, capped at RIDGE_CAP px) and ALL cells on the
    ridge (target >= 0.5, +-1 cell in x) regress the same box -- a ring has no landmark along chi, so its centre is
    ambiguous and the target should not demand one exact cell."""
    dev = boxes_xyxy.device
    h, w = H // stride, W // stride
    heat = torch.zeros(2, h, w, device=dev)
    reg = torch.zeros(4, h, w, device=dev)
    wgt = torch.zeros(h, w, device=dev)
    N = boxes_xyxy.shape[0]
    if N == 0:
        return heat, reg, (wgt[None].repeat(4, 1, 1) if ring_mode == 'ridge' else wgt)
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
    bw = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=1.0)
    bh = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=1.0)
    ring = labels.bool()
    ix = (cx / stride).floor().clamp(0, w - 1).long()
    iy = (cy / stride).floor().clamp(0, h - 1).long()
    ridge = ring & (ring_mode == 'ridge')
    sx, sy = gauss_sigma(bw, bh, ring)
    d = torch.arange(-R, R + 1, device=dev)
    dy, dx = torch.meshgrid(d, d, indexing='ij')              # [K,K]
    gx = (ix[:, None, None] + dx)                             # [N,K,K]
    gy = (iy[:, None, None] + dy)
    valid = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
    g = torch.exp(-((dx * stride) ** 2 / (2 * sx[:, None, None] ** 2)
                    + (dy * stride) ** 2 / (2 * sy[:, None, None] ** 2)))
    g = torch.where(valid & ~ridge[:, None, None], g, torch.zeros_like(g))
    flat = (labels[:, None, None] * h * w + gy.clamp(0, h - 1) * w + gx.clamp(0, w - 1)).reshape(-1)
    heat.view(-1).scatter_reduce_(0, flat, g.reshape(-1), reduce='amax', include_self=True)
    # ring ridges (ring_mode == 'ridge'): tall Gaussian along chi, peak 1.0 at the centre cell
    xc_l, yc_l, gi_l = [], [], []
    for i in torch.nonzero(ridge).squeeze(1).tolist():
        syi = float(min(max(bh[i].item() / 6, 4.0), RIDGE_CAP)); sxi = float(sx[i].item())
        ky = int(3 * syi / stride) + 1
        ys = torch.arange(max(int(iy[i]) - ky, 0), min(int(iy[i]) + ky + 1, h), device=dev)
        xs = torch.arange(max(int(ix[i]) - R, 0), min(int(ix[i]) + R + 1, w), device=dev)
        gy = torch.exp(-((ys - iy[i]) * stride) ** 2 / (2 * syi ** 2))
        gx = torch.exp(-((xs - ix[i]) * stride) ** 2 / (2 * sxi ** 2))
        blk = gy[:, None] * gx[None, :]
        heat[1, ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1] = torch.maximum(heat[1, ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1], blk)
        ky50 = int(1.1774 * syi / stride)                          # cells with target >= 0.5
        for oy in range(-ky50, ky50 + 1):
            for ox in (-1, 0, 1):
                yc_l.append(int(iy[i]) + oy); xc_l.append(int(ix[i]) + ox); gi_l.append(i)
    # regression: 3x3 around each centre cell, nearest GT wins
    offs = torch.tensor([(ox, oy) for oy in (-1, 0, 1) for ox in (-1, 0, 1)], device=dev)   # [9,2]
    nx = ix[None] + offs[:, 0:1]                                   # [9,N]
    ny = iy[None] + offs[:, 1:2]
    ok = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
    gi = torch.arange(N, device=dev)[None].expand(9, N)
    cell_x, cell_y, gidx = nx[ok], ny[ok], gi[ok]
    if xc_l:                                                       # extra ridge cells for rings
        ex = torch.tensor(xc_l, device=dev); ey = torch.tensor(yc_l, device=dev); eg = torch.tensor(gi_l, device=dev)
        v = (ex >= 0) & (ex < w) & (ey >= 0) & (ey < h)
        cell_x = torch.cat([cell_x, ex[v]]); cell_y = torch.cat([cell_y, ey[v]]); gidx = torch.cat([gidx, eg[v]])
    cell = cell_y * w + cell_x
    dist = ((cx[gidx] / stride - (cell_x + 0.5)) ** 2 + (cy[gidx] / stride - (cell_y + 0.5)) ** 2)
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
    if ring_mode == 'ridge':
        # per-channel weights [4,h,w]; ridge-only cells (>1 cell from a ring's centre cell) skip the y-offset:
        # a ring has no landmark along chi, so that offset is unknowable and would dominate the loss
        w4 = wgt[None].repeat(4, 1, 1)
        ro = ridge[a] & ((yy - iy[a]).abs() > 1)
        w4[1, yy[ro], xx[ro]] = 0.0
        return heat, reg, w4
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
    w4 = wgt[:, None].expand(-1, 4, -1, -1) if wgt.dim() == 3 else wgt      # [B,h,w] or per-channel [B,4,h,w]
    n = (w4.amax(1) > 0).sum().clamp(min=1).float()
    l1 = (F.l1_loss(out['reg'], reg_gt, reduction='none') * w4).sum() / n / 4
    # offsets are in cell units, sizes in log-px: report separately
    return lh + reg_coef * l1 * 4, dict(loss_heat=lh.item(), loss_reg=l1.item())
