"""Polar resampling, box labels and overlap rules for the realistic GIWAXS images (moved from realsim_tests.ipynb, cell 3).

Usage:

    from polar_labels import to_polar, box_labels, boxes_from_labels, to_raw_output
    pol, q_max = to_polar(res, params)
    lab = box_labels(res, params, q_max=q_max, pol=pol)
    boxes = boxes_from_labels(lab, ratio=(3.0, 3.0), q_max=q_max, fmt="px_cxcywh")
    img, mask = to_raw_output(pol)
"""
import warnings

import numpy as np
from scipy.ndimage import map_coordinates

Q_MAX = float(np.hypot(5.0, 5.0))                     # default |q| range of the polar image for q ranges (0, 5) x (0, 5)
K_FWHM = 2 * np.sqrt(2 * np.log(2))                  # FWHM = K_FWHM * sigma (2.3548)
POLAR_SHAPE = (512, 1024)                            # (n_chi rows, n_q columns) = model input


def polar_xy(r, chi, chi_from="qz"):
    """(|q|, chi) -> (q_xy, q_z). chi is measured from the q_z axis ("qz") or from the q_xy axis ("qxy")."""
    return (r * np.sin(chi), r * np.cos(chi)) if chi_from == "qz" else (r * np.cos(chi), r * np.sin(chi))


def to_polar(res, params, shape=POLAR_SHAPE, q_max=None, chi_from="qz", ss=3):
    """Resample the Cartesian image onto a (chi, |q|) grid of `shape`: |q| in [0, q_max], chi in [0, 90 deg].

    Each polar pixel is the mean of ss x ss bilinear samples (anti-aliasing, so the sharp peaks are not
    undersampled). Masked / outside pixels are excluded from the mean; a polar pixel with less than half
    valid area becomes NaN. Assumes both q ranges start at 0. Returns the polar image and q_max used.
    """
    img, valid = res["img"], res["valid"]
    (x0, x1), (z0, z1) = params.q_xy_range, params.q_z_range
    nz, nx = img.shape
    q_max = q_max or float(np.hypot(x1, z1))                      # corner of the image, 7.07 A^-1 for a 5 x 5 range
    n_chi, n_q = shape
    r_f = (np.arange(n_q * ss) + 0.5) * q_max / (n_q * ss)
    c_f = (np.arange(n_chi * ss) + 0.5) * (np.pi / 2) / (n_chi * ss)
    R, C = np.meshgrid(r_f, c_f)
    X, Z = polar_xy(R, C, chi_from)
    coords = [(Z - z0) / (z1 - z0) * nz - 0.5, (X - x0) / (x1 - x0) * nx - 0.5]
    num = map_coordinates(np.where(valid, img, 0.0), coords, order=1, mode="constant", cval=0.0)
    den = map_coordinates(valid.astype(np.float32), coords, order=1, mode="constant", cval=0.0)
    block = lambda a: a.reshape(n_chi, ss, n_q, ss).mean(axis=(1, 3))
    num, den = block(num), block(den)
    pol = np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan)
    return pol.astype(np.float32), q_max


def peak_boxes_polar(res, params, ratio=(3.0, 3.0), shape=POLAR_SHAPE, q_max=None, chi_from="qz", min_px=2.0, pol = None, min_valid = 0.2):
    """Boxes in polar space for the oriented spots of a make_sample() result.

    ratio = (radial, angular): FULL width in units of sigma, sigma = FWHM / 2.3548 (same widths as the renderer).
    Boxes are clipped to the polar frame. Returns
      boxes    (N, 4): [q_center (A^-1), chi_center (rad), width_q (A^-1), width_chi (rad)]
      boxes_px (N, 4): [x_center, y_center, width, height] in pixels of the polar image (x = |q| column, y = chi row)
      keep     (N,)  : indices into res["q"], res["I"], res["mi"]
    """
    sh, dchi = res["meta"]["shape"], np.deg2rad(res["meta"]["mosaic_fwhm_deg"])
    size_fwhm = 2 * np.pi * 0.9 / (sh["size_nm"] * 10.0)
    (x0, x1), (z0, z1) = params.q_xy_range, params.q_z_range
    nz, nx = res["img"].shape
    q_max = q_max or float(np.hypot(x1, z1))                      # corner of the image, 7.07 A^-1 for a 5 x 5 range
    n_chi, n_q = shape
    dq, dc = q_max / n_q, (np.pi / 2) / n_chi                     # polar pixel sizes
    boxes, keep = [], []
    for i, (qxy, qz) in enumerate(zip(res["q"][0], res["q"][1])):
        rr = np.hypot(qxy, qz)
        if rr == 0 or rr >= q_max:
            continue
        f_r = np.sqrt(sh["inst_fwhm"] ** 2 + size_fwhm ** 2 + (sh["disorder"] * rr ** 2) ** 2)
        f_t = np.sqrt(f_r ** 2 + (rr * dchi) ** 2)
        w_q = max(ratio[0] * f_r / K_FWHM, min_px * dq)               # A^-1
        w_c = max(ratio[1] * f_t / K_FWHM / rr, min_px * dc)          # rad
        chi_c = np.arctan2(qxy, qz) if chi_from == "qz" else np.arctan2(qz, qxy)
        q_lo, q_hi = max(rr - w_q / 2, 0), min(rr + w_q / 2, q_max)           # clip to the frame
        c_lo, c_hi = max(chi_c - w_c / 2, 0), min(chi_c + w_c / 2, np.pi / 2)
        if pol is not None:                                           # keep the peak if enough of its box is visible
            xa, xb = int(q_lo / dq), max(int(np.ceil(q_hi / dq)), int(q_lo / dq) + 1)
            ya, yb = int(c_lo / dc), max(int(np.ceil(c_hi / dc)), int(c_lo / dc) + 1)
            if np.isfinite(pol[ya:yb, xa:xb]).mean() < min_valid:
                continue
        boxes.append([(q_lo + q_hi) / 2, (c_lo + c_hi) / 2, q_hi - q_lo, c_hi - c_lo])
        keep.append(i)
    boxes = np.array(boxes).reshape(-1, 4)
    boxes_px = boxes / np.array([dq, dc, dq, dc])
    return boxes, boxes_px, np.array(keep, dtype=int)


def ring_boxes_polar(res, params, ratio=(3.0, 3.0), shape=POLAR_SHAPE, q_max=None, pol=None, min_valid=0.075, min_px=2.0, min_rows=4):
    """Boxes in polar space for the powder rings of a make_sample() result (modes "rings" and "both").

    A ring is a vertical line in polar space: centre at chi = 45 deg, full angular extent 0 to 90 deg,
    radial width = ratio[0] * sigma_r with the same radial FWHM as the spots (no mosaic term for rings).
    A ring is kept if its visible share of the arc that the detector can show at that |q| (corner rings only reach
    a few degrees of chi) is at least min_valid, and at least min_rows chi rows are visible.
    Returns boxes, boxes_px (same layout as peak_boxes_polar) and keep (indices into res["ring_q"], res["ring_I"]).
    """
    sh = res["meta"]["shape"]
    size_fwhm = 2 * np.pi * 0.9 / (sh["size_nm"] * 10.0)
    (x0, x1), (z0, z1) = params.q_xy_range, params.q_z_range
    q_max = q_max or float(np.hypot(x1, z1))
    n_chi, n_q = shape
    dq, dc = q_max / n_q, (np.pi / 2) / n_chi
    boxes, keep = [], []
    for i, q0 in enumerate(res["ring_q"]):
        if q0 <= 0 or q0 >= q_max:
            continue
        f_r = np.sqrt(sh["inst_fwhm"] ** 2 + size_fwhm ** 2 + (sh["disorder"] * q0 ** 2) ** 2)
        w_q = max(ratio[0] * f_r / K_FWHM, min_px * dq)               # A^-1
        q_lo, q_hi = max(q0 - w_q / 2, 0), min(q0 + w_q / 2, q_max)
        if pol is not None:                                           # keep the ring if enough of its visible arc is left
            xa, xb = int(q_lo / dq), max(int(np.ceil(q_hi / dq)), int(q_lo / dq) + 1)
            vis = np.isfinite(pol[:, xa:xb]).mean()                   # visible share of the whole 0-90 deg strip
            geom = max(np.arcsin(min(x1 / q0, 1.0)) - np.arccos(min(z1 / q0, 1.0)), 0) / (np.pi / 2)  # share inside the detector
            if geom <= 0 or vis < min_valid * geom or vis * n_chi < min_rows:
                continue
        boxes.append([(q_lo + q_hi) / 2, np.pi / 4, q_hi - q_lo, np.pi / 2])
        keep.append(i)
    boxes = np.array(boxes).reshape(-1, 4)
    boxes_px = boxes / np.array([dq, dc, dq, dc])
    return boxes, boxes_px, np.array(keep, dtype=int)

def peak_sigmas(res, q_abs, ring=False):
    """Per-peak sigma in |q| (A^-1) and in chi (rad); same widths as the renderer, sigma = FWHM / 2.3548.
    For rings the angular sigma is NaN (a ring box always spans the full 0-90 deg)."""
    sh, dchi = res["meta"]["shape"], np.deg2rad(res["meta"]["mosaic_fwhm_deg"])
    size_fwhm = 2 * np.pi * 0.9 / (sh["size_nm"] * 10.0)
    q_abs = np.asarray(q_abs, dtype=float)
    f_r = np.sqrt(sh["inst_fwhm"] ** 2 + size_fwhm ** 2 + (sh["disorder"] * q_abs ** 2) ** 2)
    f_t = np.sqrt(f_r ** 2 + (q_abs * dchi) ** 2)
    sig_chi = np.full_like(q_abs, np.nan) if ring else f_t / K_FWHM / np.maximum(q_abs, 1e-9)
    return f_r / K_FWHM, sig_chi


def boxes_from_labels(lab, ratio=(3.0, 3.0), shape=POLAR_SHAPE, q_max=None, fmt="px_cxcywh", flip_chi=False, min_px=2.0):
    """Boxes of any width ratio / format from box_labels(). Boxes are clipped to the polar frame.

    fmt      : "px_cxcywh"   [x_center, y_center, width, height] in pixels of the polar image (x = |q|, y = chi row)
               "px_xyxy"     [x_min, y_min, x_max, y_max] in pixels
               "norm_cxcywh" like px_cxcywh divided by the image width / height (values in 0..1)
               "phys"        [q_center (A^-1), chi_center (rad), width_q (A^-1), width_chi (rad)]
    flip_chi : False = row 0 is chi = 0 deg (the array as produced by to_polar), True = row 0 is chi = 90 deg
               (then flip the image too: np.flipud(pol)).
    Rings: width_chi is always the full 90 deg, the ratio only changes their radial width.
    """
    n_chi, n_q = shape
    q_max = q_max or Q_MAX
    dq, dc = q_max / n_q, (np.pi / 2) / n_chi
    ring = lab["kind"] == 1
    w_q = np.maximum(ratio[0] * lab["sigma_q"], min_px * dq)
    w_c = np.where(ring, np.pi / 2, np.maximum(ratio[1] * np.nan_to_num(lab["sigma_chi"]), min_px * dc))
    q_lo, q_hi = np.maximum(lab["q_c"] - w_q / 2, 0), np.minimum(lab["q_c"] + w_q / 2, q_max)
    c_lo, c_hi = np.maximum(lab["chi_c"] - w_c / 2, 0), np.minimum(lab["chi_c"] + w_c / 2, np.pi / 2)
    if fmt == "phys":
        return np.column_stack([(q_lo + q_hi) / 2, (c_lo + c_hi) / 2, q_hi - q_lo, c_hi - c_lo])
    x1, x2 = q_lo / dq, q_hi / dq
    y1, y2 = c_lo / dc, c_hi / dc
    if flip_chi:
        y1, y2 = n_chi - y2, n_chi - y1
    if fmt == "px_xyxy":
        return np.column_stack([x1, y1, x2, y2])
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    if fmt == "px_cxcywh":
        return np.column_stack([cx, cy, w, h])
    if fmt == "norm_cxcywh":
        return np.column_stack([cx / n_q, cy / n_chi, w / n_q, h / n_chi])
    raise ValueError(f"unknown fmt {fmt!r}")

def to_raw_output(pol, flip_chi=False):
    """Raw model-input pair from a polar image: (counts, mask).

    counts : float32, raw detector counts (no scaling); masked pixels (wedge, gaps, beam stop, outside the detector) = 0
    mask   : uint8, 1 = valid pixel, 0 = masked. A valid pixel can legitimately be 0 counts (dead pixel, empty), so the
             mask is the only way to tell them apart.
    flip_chi: same row convention as boxes_from_labels(flip_chi=...); pass the same value to both.
    """
    mask = np.isfinite(pol)
    img = np.where(mask, pol, 0.0).astype(np.float32)
    mask = mask.astype(np.uint8)
    if flip_chi:
        img, mask = np.flipud(img), np.flipud(mask)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)

def _sl(lo, hi, n):
    """Integer slice covering [lo, hi) in pixel-edge coordinates, at least 1 pixel, clipped to [0, n]."""
    a = int(np.clip(np.floor(lo), 0, n - 1))
    return slice(a, int(np.clip(np.ceil(hi), a + 1, n)))


def overlap_pairs(lab, ratio=(3.0, 3.0), shape=POLAR_SHAPE, q_max=None):
    """All pairs of labelled boxes that overlap in the polar image (pixel coordinates).

    type  : 0 spot-spot, 1 ring-ring, 2 spot-ring (spot is always `i`, ring `j`)
    iou   : intersection / union
    iomin : intersection / area of the smaller box (a spot lying inside a ring strip has a tiny IoU but IoMin ~ 1)
    Needs lab["snr"] (set it with box_snr). Also returns the boxes (x1, y1, x2, y2), snr and sim intensity of both
    members and their indices into `lab`.
    """
    xyxy = boxes_from_labels(lab, ratio=ratio, shape=shape, q_max=q_max, fmt="px_xyxy")
    if len(xyxy) < 2:
        return []
    x1, y1, x2, y2 = xyxy.T
    iw = np.clip(np.minimum(x2[:, None], x2[None, :]) - np.maximum(x1[:, None], x1[None, :]), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2[None, :]) - np.maximum(y1[:, None], y1[None, :]), 0, None)
    inter, area = iw * ih, (x2 - x1) * (y2 - y1)
    iou = inter / (area[:, None] + area[None, :] - inter)
    iomin = inter / np.minimum(area[:, None], area[None, :])
    kind, out = lab["kind"], []
    for i, j in zip(*np.where(np.triu(inter > 0, 1))):
        a, b = (i, j) if kind[i] <= kind[j] else (j, i)
        out.append(dict(i=int(a), j=int(b), type=0 if kind[a] + kind[b] == 0 else (1 if kind[a] + kind[b] == 2 else 2),
                        iou=float(iou[i, j]), iomin=float(iomin[i, j]), box_i=xyxy[a], box_j=xyxy[b],
                        snr_i=float(lab["snr"][a]), snr_j=float(lab["snr"][b]),
                        I_i=float(lab["intensity"][a]), I_j=float(lab["intensity"][b])))
    return out


def box_snr(pol, xyxy, kind, pad=2.0, return_signal=False):
    """Peak-to-noise ratio of every box in the polar image the model sees (NaN = cannot be judged).

    signal = median of the central third of the box minus the local background (median: robust to hot pixels).
    background = median and noise = 1.4826 * MAD of the surroundings: spots use a frame of `pad` box sizes around
    the box, rings use the strips left and right of the ring. Noise is per pixel, so a long thin ring is easier
    to see than its SNR suggests (judge rings with a lower threshold).
    return_signal=True also returns the signal itself (counts above background), the image intensity of each box.
    """
    n_chi, n_q = pol.shape
    out = np.full(len(xyxy), np.nan)
    sig = np.full(len(xyxy), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j, ((x1, y1, x2, y2), k) in enumerate(zip(xyxy, kind)):
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            xs = _sl(cx - w / 6, cx + w / 6, n_q)
            ys = slice(0, n_chi) if k == 1 else _sl(cy - h / 6, cy + h / 6, n_chi)      # ring: whole column
            core = pol[ys, xs]
            if not np.isfinite(core).any():                                              # centre masked: use the whole box
                core = pol[slice(0, n_chi) if k == 1 else _sl(y1, y2, n_chi), _sl(x1, x2, n_q)]
            if k == 1:
                around = np.concatenate([pol[:, _sl(x1 - pad * w, x1, n_q)].ravel(), pol[:, _sl(x2, x2 + pad * w, n_q)].ravel()])
            else:
                sx, sy = _sl(x1 - pad * w, x2 + pad * w, n_q), _sl(y1 - pad * h, y2 + pad * h, n_chi)
                reg = pol[sy, sx].copy()
                bx, by = _sl(x1, x2, n_q), _sl(y1, y2, n_chi)
                reg[by.start - sy.start:by.stop - sy.start, bx.start - sx.start:bx.stop - sx.start] = np.nan   # without the box
                around = reg.ravel()
            around = around[np.isfinite(around)]
            if around.size < 10 or not np.isfinite(core).any():
                continue
            bg = np.median(around)
            noise = max(1.4826 * np.median(np.abs(around - bg)), 1e-3)
            sig[j] = np.nanmedian(core) - bg
            out[j] = sig[j] / noise
    return (out, sig) if return_signal else out


def _select(lab, ok):
    """Subset of a labels dict (arrays and the `mi` list) by a boolean mask."""
    idx = np.flatnonzero(ok)
    return {k: ([v[i] for i in idx] if k == "mi" else v[ok]) for k, v in lab.items()}


def resolve_overlaps(lab, ratio=(3.0, 3.0), shape=POLAR_SHAPE, q_max=None, iou_ss=0.1, iou_rr=0.1, iomin_sr=0.0):
    """Decide which overlapping boxes stay. Needs lab["snr"] and lab["signal"] (set by box_labels).

    Labels are visited from the strongest to the weakest image signal (counts above background); a label is removed
    if a stronger label that is still kept conflicts with it:
      spot  - spot : IoU >= iou_ss        -> the weaker spot is removed
      ring  - ring : IoU >= iou_rr        -> the weaker ring is removed
      spot  - ring : the spot is removed only if the ring is stronger than the spot and IoMin >= iomin_sr (0 = any overlap);
                     a spot that is stronger than the ring keeps both boxes (the ring never loses its box)
    Returns keep (bool per label) and why {removed index: (kept index, pair type, IoU, IoMin)}.
    """
    n = len(lab["kind"])
    keep, why = np.ones(n, dtype=bool), {}
    if n < 2:
        return keep, why
    height = np.nan_to_num(np.asarray(lab["signal"], dtype=float), nan=-np.inf)      # unmeasurable = weakest
    order = np.argsort(-height, kind="stable")
    rank = np.empty(n, dtype=int); rank[order] = np.arange(n)
    nb = {}
    for p in overlap_pairs(lab, ratio=ratio, shape=shape, q_max=q_max):
        nb.setdefault(p["i"], []).append((p["j"], p)); nb.setdefault(p["j"], []).append((p["i"], p))
    for b in order:
        for a, p in nb.get(b, []):
            if rank[a] > rank[b] or not keep[a]:                                     # only stronger labels still kept
                continue
            if p["type"] == 0:
                hit = p["iou"] >= iou_ss
            elif p["type"] == 1:
                hit = p["iou"] >= iou_rr
            else:
                hit = lab["kind"][b] == 0 and p["iomin"] >= iomin_sr                 # weaker spot on a stronger ring
            if hit:
                keep[b] = False; why[int(b)] = (int(a), p["type"], p["iou"], p["iomin"]); break
    return keep, why


def box_labels(res, params, shape=POLAR_SHAPE, q_max=None, chi_from="qz", pol=None, ratio=(3.0, 3.0),
               min_snr=(1.1, 1.1), resolve=True, iou_ss=0.1, iou_rr=0.1, iomin_sr=0.0):
    """Ratio-independent labels of all boxed spots and rings of one sample.

    `ratio` is used to decide which peaks are visible enough to be labelled (min_valid rule) and for the overlap
    test; the stored widths are sigmas, so boxes of any ratio can be derived later with boxes_from_labels.
    Returns a dict of arrays: kind (0 spot, 1 ring), src (index into res["q"] / res["ring_q"]), q_c (A^-1),
    chi_c (rad), sigma_q (A^-1), sigma_chi (rad, NaN for rings), intensity (simulated), snr and signal (measured in
    the polar image: peak-to-noise, counts above background; NaN = cannot be judged) and mi (Miller indices, None for rings).
    min_snr  : default (1.1, 1.1); None = no cut; (spot, ring) thresholds drop boxes whose snr is below it (NaN is kept).
    resolve  : True = remove overlapping boxes with resolve_overlaps (thresholds iou_ss, iou_rr, iomin_sr).
    """
    _, _, keep = peak_boxes_polar(res, params, ratio=ratio, shape=shape, q_max=q_max, chi_from=chi_from, pol=pol)
    _, _, rkeep = ring_boxes_polar(res, params, ratio=ratio, shape=shape, q_max=q_max, pol=pol)
    qxy, qz = res["q"][0][keep], res["q"][1][keep]
    q_s = np.hypot(qxy, qz)
    chi_s = np.arctan2(qxy, qz) if chi_from == "qz" else np.arctan2(qz, qxy)
    q_r = np.asarray(res["ring_q"])[rkeep]
    sq_s, sc_s = peak_sigmas(res, q_s)
    sq_r, sc_r = peak_sigmas(res, q_r, ring=True)
    lab = dict(kind=np.r_[np.zeros(len(keep)), np.ones(len(rkeep))].astype(np.int8),
               src=np.r_[keep, rkeep].astype(int),
               q_c=np.r_[q_s, q_r], chi_c=np.r_[chi_s, np.full(len(rkeep), np.pi / 4)],
               sigma_q=np.r_[sq_s, sq_r], sigma_chi=np.r_[sc_s, sc_r],
               intensity=np.r_[np.asarray(res["I"])[keep], np.asarray(res["ring_I"])[rkeep]],
               mi=[res["mi"][i] for i in keep] + [None] * len(rkeep))
    lab["snr"] = np.full(len(lab["kind"]), np.nan)
    lab["signal"] = np.full(len(lab["kind"]), np.nan)
    if pol is not None and len(lab["kind"]):
        xyxy = boxes_from_labels(lab, ratio=ratio, shape=shape, q_max=q_max, fmt="px_xyxy")
        lab["snr"], lab["signal"] = box_snr(pol, xyxy, lab["kind"], return_signal=True)
        if min_snr is not None:
            thr = np.where(lab["kind"] == 1, min_snr[1], min_snr[0])
            lab = _select(lab, ~(lab["snr"] < thr))
        if resolve and len(lab["kind"]) > 1:
            ok, _ = resolve_overlaps(lab, ratio=ratio, shape=shape, q_max=q_max, iou_ss=iou_ss, iou_rr=iou_rr, iomin_sr=iomin_sr)
            lab = _select(lab, ok)
    return lab