"""Realistic GIWAXS image generator on top of pygidSIM.

Usage (from a notebook in this folder):

    from pipeline import make_sample, he_log_view, PEAK_MIN, PEAK_MAX
    r = make_sample(cif_path, "both", rng, params)      # modes: "oriented", "rings", "both"

Pipeline: simulate peaks from the CIF -> select peaks -> render (pseudo-Voigt spots, arcs, rings)
-> absolute counts -> background -> noise -> detector geometry -> missing wedge.
"""
import numpy as np
from pygidsim import GIWAXSFromCif
from pygidsim.directions import get_unique_directions

PEAK_MIN, PEAK_MAX = 1e4, 2e7        # counts of the brightest peak per image (log-uniform draw)


A_L = 2 ** (2 / 3) - 1


def render_giwaxs_sharp(q, I, q_xy_range, q_z_range,
                        pix=0.003,             # pixel size in Å^-1 (real detectors: ~0.002 to 0.005)
                        inst_fwhm=0.004,       # instrumental resolution (FWHM, Å^-1)
                        size_nm=40.0,          # crystallite size -> Scherrer FWHM = 2*pi*0.9/L
                        disorder=0.002,        # paracrystalline-like term: FWHM += disorder * |q|^2  (Å)
                        mosaic_fwhm_deg=1.5,   # FWHM of the orientation spread (1 to 3 deg typical)
                        eta=0.3,               # Lorentzian fraction of the pseudo-Voigt profile
                        window=12.0,           # draw each peak only within +-window*FWHM (speed)
                        ):
    """Return an image normalized to a maximum of 1, shape (nz, nx)."""
    nx = int(round((q_xy_range[1] - q_xy_range[0]) / pix))
    nz = int(round((q_z_range[1] - q_z_range[0]) / pix))
    x = q_xy_range[0] + (np.arange(nx) + 0.5) * pix
    z = q_z_range[0] + (np.arange(nz) + 0.5) * pix
    img = np.zeros((nz, nx), dtype=np.float32)
    dchi = np.deg2rad(mosaic_fwhm_deg)
    size_fwhm = 2 * np.pi * 0.9 / (size_nm * 10.0)

    for qxy, qz, inten in zip(q[0], q[1], I):
        r = np.hypot(qxy, qz)
        if r == 0:
            continue
        f_r = np.sqrt(inst_fwhm ** 2 + size_fwhm ** 2 + (disorder * r ** 2) ** 2)
        f_t = np.sqrt(f_r ** 2 + (r * dchi) ** 2)
        w_r, w_t = f_r / 2, f_t / 2              # half widths at half maximum

        # only evaluate a window around the peak
        hw = window * max(f_r, f_t)
        ix0 = max(int((qxy - hw - q_xy_range[0]) / pix), 0)
        ix1 = min(int((qxy + hw - q_xy_range[0]) / pix) + 1, nx)
        iz0 = max(int((qz - hw - q_z_range[0]) / pix), 0)
        iz1 = min(int((qz + hw - q_z_range[0]) / pix) + 1, nz)
        if ix0 >= ix1 or iz0 >= iz1:
            continue

        dx = x[None, ix0:ix1] - qxy
        dz = z[iz0:iz1, None] - qz
        d_rad = (dx * qxy + dz * qz) / r         # along q
        d_tan = (dx * qz - dz * qxy) / r         # along the arc
        rho2 = (d_rad / w_r) ** 2 + (d_tan / w_t) ** 2

        gauss = np.exp(-np.log(2) * rho2) * np.log(2) / (np.pi * w_r * w_t)
        lorentz = (1 + A_L * rho2) ** -1.5 * A_L / (2 * np.pi * w_r * w_t)
        img[iz0:iz1, ix0:ix1] += inten * (eta * lorentz + (1 - eta) * gauss)

    return img / img.max()


def select_peaks(I, n_max=None, rel_threshold=None, mode="weighted", power=1.0, rng=None):
    """Return the indices of the peaks to render.

    n_max          : keep at most this many peaks (None = no limit)
    rel_threshold  : drop peaks weaker than rel_threshold * I.max() (None = off)
    mode           : "top"      -> the n_max strongest peaks
                     "weighted" -> random subset, probability ~ I**power, so strong peaks
                                   are likely but not guaranteed and weak ones sometimes appear
    """
    idx = np.arange(len(I))
    if rel_threshold is not None:
        idx = idx[I[idx] >= rel_threshold * I.max()]
    if n_max is None or len(idx) <= n_max:
        return idx
    if mode == "top":
        return idx[np.argsort(I[idx])[::-1][:n_max]]
    rng = rng or np.random.default_rng()
    p = I[idx] ** power
    return rng.choice(idx, size=n_max, replace=False, p=p / p.sum())


# ---- COLORBOX1 fork (v3): more spots / rings, skewed to the high end, with some very sparse frames; low-q bias in the random peak choice ----
Q0_LOW = 1.5      # 1/A: selection weight exp(-|q| / Q0_LOW) on top of I**power


def count_v3(rng, lo, hi, p_sparse, sparse_hi, skew):
    """Number of peaks to keep: with probability p_sparse a very sparse frame (lo .. sparse_hi), otherwise sparse_hi + (hi - sparse_hi) * u**skew; skew < 1 puts the weight on the high end."""
    if rng.random() < p_sparse:
        return int(rng.integers(lo, sparse_hi + 1))
    return int(sparse_hi + (hi - sparse_hi) * rng.random() ** skew)


def select_peaks_q(I, q, n_max=None, rel_threshold=None, mode="weighted", power=1.0, rng=None, q0=Q0_LOW):
    """select_peaks with an extra weight exp(-|q|/q0) in the random draw (only matters when more than n_max peaks pass the threshold)."""
    idx = np.arange(len(I))
    if rel_threshold is not None:
        idx = idx[I[idx] >= rel_threshold * I.max()]
    if n_max is None or len(idx) <= n_max:
        return idx
    if mode == "top":
        return idx[np.argsort(I[idx])[::-1][:n_max]]
    rng = rng or np.random.default_rng()
    p = I[idx] ** power * np.exp(-np.abs(np.asarray(q)[idx]) / q0)
    return rng.choice(idx, size=n_max, replace=False, p=p / p.sum())


def make_background(q_xy_range, q_z_range, npix=(600, 300), amp=100.0,
                    q_decay=1.0,     # decay length of the diffuse term, Å^-1
                    halo_frac=0.3,   # halo height relative to the diffuse term at q = 0
                    halo_q=1.5,      # halo centre, Å^-1 (amorphous organic packing)
                    halo_w=0.25,     # halo width (sigma), Å^-1
                    offset_frac=0.02 # constant offset relative to amp
                    ):
    """Smooth background in counts, shape (npix[1], npix[0])."""
    x = np.linspace(*q_xy_range, npix[0])
    z = np.linspace(*q_z_range, npix[1])
    X, Z = np.meshgrid(x, z)
    R = np.hypot(X, Z)
    diffuse = np.exp(-R / q_decay)
    halo = halo_frac * np.exp(-0.5 * ((R - halo_q) / halo_w) ** 2)
    return amp * (diffuse + halo + offset_frac)


def add_detector_noise(expected, rng, gain_sigma=0.01, read_sigma=2.0, saturation=None):
    gain = rng.normal(1.0, gain_sigma, size=expected.shape)       # 1 % pixel gain spread
    lam = np.clip(expected * gain, 0, None)
    counts = rng.poisson(lam).astype(np.float64)                  # shot noise
    counts += rng.normal(0.0, read_sigma, size=counts.shape)      # readout noise
    if saturation is not None:
        counts = np.minimum(counts, saturation)
    return np.clip(np.rint(counts), 0, None).astype(np.float32)


def apply_detector_geometry(img, q_xy_range, q_z_range, rng,
                            beamstop_r=0.08,        # radius of the beam-stop disc, Å^-1
                            beamstop_bar=0.0,       # half-width of a vertical bar at q_xy = 0 (0 = off)
                            module_px=(487, 195),   # module size (columns, rows) in pixels
                            gap_px=(7, 17),         # gap width (columns, rows) in pixels; (0, 0) = no gaps
                            dead_frac=1e-4,         # fraction of dead pixels
                            hot_frac=1e-5,          # fraction of hot pixels
                            mask_value=np.nan):
    nz, nx = img.shape
    x = q_xy_range[0] + (np.arange(nx) + 0.5) * (q_xy_range[1] - q_xy_range[0]) / nx
    z = q_z_range[0] + (np.arange(nz) + 0.5) * (q_z_range[1] - q_z_range[0]) / nz
    X, Z = np.meshgrid(x, z)

    masked = np.hypot(X, Z) < beamstop_r                         # beam stop disc
    if beamstop_bar > 0:
        masked |= np.abs(X) < beamstop_bar                       # vertical bar
    if gap_px[0] > 0:                                            # vertical gaps between modules
        col = (np.arange(nx) + rng.integers(module_px[0])) % module_px[0]
        masked |= (col < gap_px[0])[None, :]
    if gap_px[1] > 0:                                            # horizontal gaps between modules
        row = (np.arange(nz) + rng.integers(module_px[1])) % module_px[1]
        masked |= (row < gap_px[1])[:, None]

    out = img.astype(np.float32).copy()
    out[rng.random(img.shape) < dead_frac] = 0                   # dead pixels
    hot = rng.random(img.shape) < hot_frac                       # hot pixels
    out[hot] = rng.uniform(0.01, 0.5, hot.sum()) * img.max()
    out[masked] = mask_value
    return out, ~masked


def wedge_mask(shape, q_xy_range, q_z_range, wavelength, margin=0.01):
    nz, nx = shape
    x = q_xy_range[0] + (np.arange(nx) + 0.5) * (q_xy_range[1] - q_xy_range[0]) / nx
    z = q_z_range[0] + (np.arange(nz) + 0.5) * (q_z_range[1] - q_z_range[0]) / nz
    k = 2 * np.pi / wavelength
    edge = k - np.sqrt(np.clip(k ** 2 - z ** 2, 0, None))        # wedge half-width at each q_z
    return np.abs(x)[None, :] < (edge[:, None] - margin)


def render_rings(q1d, I1d, q_xy_range, q_z_range, pix=0.003,
                 inst_fwhm=0.004, size_nm=40.0, disorder=0.002, eta=0.3,
                 kappa=0.0,        # angular concentration: 0 = uniform ring, larger = concentrated near the texture axis
                 chi0_deg=0.0):    # axis of the concentration, angle from the q_z axis
    """Debye-Scherrer rings from a powder pattern (q1d: |q| of each ring, I1d: its intensity).

    The radial profile of all rings is built once on a fine 1D grid and mapped to the image with
    np.interp, so the cost is independent of the number of rings. Returns an unnormalized float32 image.
    """
    nx = int(round((q_xy_range[1] - q_xy_range[0]) / pix))
    nz = int(round((q_z_range[1] - q_z_range[0]) / pix))
    x = q_xy_range[0] + (np.arange(nx) + 0.5) * pix
    z = q_z_range[0] + (np.arange(nz) + 0.5) * pix
    X, Z = np.meshgrid(x, z)
    R = np.hypot(X, Z)

    Rg = np.linspace(0, R.max() * 1.001, 20001)
    prof = np.zeros_like(Rg)
    size_fwhm = 2 * np.pi * 0.9 / (size_nm * 10.0)
    for q0, I0 in zip(q1d, I1d):
        w = np.sqrt(inst_fwhm ** 2 + size_fwhm ** 2 + (disorder * q0 ** 2) ** 2) / 2   # HWHM
        u = (Rg - q0) / w
        gauss = np.sqrt(np.log(2) / np.pi) / w * np.exp(-np.log(2) * u ** 2)           # unit area
        lorentz = 1.0 / (np.pi * w * (1 + u ** 2))                                     # unit area
        prof += I0 / (2 * np.pi * q0) * (eta * lorentz + (1 - eta) * gauss)            # spread over the circle

    ring = np.interp(R, Rg, prof)
    if kappa > 0:
        chi = np.arctan2(np.abs(X), Z)                                                 # angle from the q_z axis
        ring *= np.exp(kappa * (np.cos(2 * (chi - np.deg2rad(chi0_deg))) - 1))
    return ring.astype(np.float32)


def he_log_view(img, valid, n_bins=2048, floor=1.0):
    """Log + histogram-equalized view of a counts image.

    Counts are clipped at `floor`, log10-transformed, rescaled to [0, 1], then equalized using the
    histogram of the valid pixels only. Masked pixels are returned as NaN. Result is float32 in [0, 1].
    """
    out = np.full(img.shape, np.nan, dtype=np.float32)
    v = np.log10(np.maximum(img[valid], floor))
    lo, hi = v.min(), v.max()
    if hi <= lo:
        out[valid] = 0.0
        return out
    v = (v - lo) / (hi - lo)
    hist, edges = np.histogram(v, bins=n_bins, range=(0, 1))
    cdf = np.cumsum(hist).astype(np.float64)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])                     # 0 to 1
    out[valid] = np.interp(v, 0.5 * (edges[:-1] + edges[1:]), cdf)
    return out


def loguni(rng, lo, hi):
    """Log-uniform random number in [lo, hi]."""
    return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))


def random_orientation(rng, max_index=2):
    """Random texture axis (h, k, l) with all |indices| <= max_index (49 distinct axes for 2), drawn from `rng`.

    Uses pygidSIM's list of unique directions (inversion and collinear duplicates removed), so (0, 0, 1) is
    one of the options like any other.
    """
    dirs = get_unique_directions(max_index)
    return tuple(int(v) for v in dirs[rng.integers(len(dirs))])


def make_sample(cif_path, mode, rng, params, orientation=(0, 0, 1), strain_sigma=0.005, pix=0.003, move_fromMW=True,
                n_peaks=None, select_mode="weighted", n_rings=None, ring_kappa=0.0, max_tries=10, max_index=2):
    """Generate one realistic image from the CIF (v3 fork: peak counts and low-q weighting, see count_v3).

    mode        : "oriented" -> one oriented texture (`orientation`): spots and arcs
                  "rings"    -> powder (Debye-Scherrer) rings only: isotropic crystallites
                  "both"     -> the oriented texture plus rings
    orientation : texture axis (h, k, l), or "random" = a new axis with indices up to `max_index` on every try
                  (drawn from `rng`, so seeded runs are reproducible); the axis used is stored in meta["orientation"]
    n_peaks     : exact number of spots (None = random 2 to 200)
    n_rings     : exact number of rings (None = random 2 to 15)
    ring_kappa  : 0 = uniform rings; larger = rings concentrated near the q_z axis
    strain_sigma: std of the relative change of a, b, c (0 = unstrained lattice)
    Returns a dict with the final image (NaN = masked), mask, peak/ring labels and the parameters used.
    """
    use_spots = mode in ("oriented", "both")
    use_rings = mode in ("rings", "both")
    assert use_spots or use_rings, "mode must be 'oriented', 'rings' or 'both'"
    meta = {"mode": mode, "orientation": None}

    for _ in range(max_tries):
        # ---- texture axis (a random axis is redrawn on every try, so an axis without peaks is not fatal) ----
        if use_spots:
            orient = random_orientation(rng, max_index) if isinstance(orientation, str) else tuple(orientation)
        # ---- crystal: optional lattice strain (shifts peak positions slightly) ----
        scale = tuple(1 + rng.normal(0, strain_sigma, 3)) if strain_sigma > 0 else None
        giw = GIWAXSFromCif(cif_path, params, scale=scale).giwaxs
        meta["scale"] = scale
        ok = True

        # ---- oriented spots ----
        q_sel, I_sel, mi_sel = np.zeros((2, 0)), np.zeros(0), []
        if use_spots:
            q_all, I_all, mi_all = giw.giwaxs_sim(orientation=np.asarray(orient, dtype=float),
                                                  move_fromMW=move_fromMW, return_mi=True)
            if len(I_all) == 0:
                ok = False
            else:
                if n_peaks is not None:                          # exact number, no intensity cut
                    n_max, rel_thr = int(n_peaks), None
                else:
                    n_max, rel_thr = count_v3(rng, 2, 400, 0.15, 20, 0.7), loguni(rng, 3e-5, 1e-2)
                sel = select_peaks_q(I_all, np.hypot(q_all[0], q_all[1]), n_max=n_max, rel_threshold=rel_thr, mode=select_mode,
                                   power=float(rng.uniform(0.3, 1.0)), rng=rng)
                q_sel, I_sel, mi_sel = q_all[:, sel], I_all[sel], [mi_all[i] for i in sel]
                ok = len(I_sel) > 0

        # ---- powder rings ----
        q_ring, I_ring = np.zeros(0), np.zeros(0)
        if use_rings and ok:
            q1d, I1d = giw.giwaxs_sim(orientation=None)          # powder pattern: |q| and intensities
            if len(I1d) == 0:
                ok = False
            else:
                n_r = int(n_rings) if n_rings is not None else count_v3(rng, 2, 30, 0.2, 5, 0.7)
                keep = select_peaks_q(I1d, q1d, n_max=n_r, rel_threshold=loguni(rng, 1e-4, 1e-1),
                                    mode=select_mode, power=0.5, rng=rng)
                q_ring, I_ring = q1d[keep], I1d[keep]
                ok = len(I_ring) > 0
        if ok:
            break
    else:
        raise RuntimeError("no peaks inside the q-range after max_tries")

    # ---- peak shape (randomized, shared by spots and rings) ----
    shape = dict(pix=pix, inst_fwhm=float(rng.uniform(0.003, 0.006)), size_nm=loguni(rng, 15, 150),
                 disorder=float(rng.uniform(0.001, 0.004)), eta=float(rng.uniform(0.1, 0.6)))
    mosaic = loguni(rng, 0.5, 4.0)

    img = None
    if use_spots:
        img = render_giwaxs_sharp(q_sel, I_sel, params.q_xy_range, params.q_z_range,
                                  mosaic_fwhm_deg=mosaic, **shape)             # max = 1
    if use_rings:
        ring = render_rings(q_ring, I_ring, params.q_xy_range, params.q_z_range, kappa=ring_kappa, **shape)
        ring = ring / ring.max()                                               # max = 1
        if img is None:
            img = ring
        else:
            ring_weight = loguni(rng, 1e-3, 0.3)                               # ring height relative to the brightest spot
            img = img + ring * ring_weight
            img = img / img.max()
            meta["ring_weight"] = ring_weight

    # ---- absolute counts, background, noise ----
    peak_max = loguni(rng, PEAK_MIN, PEAK_MAX)
    bg_ratio = loguni(rng, 1e-5, 1e-3)
    bg = make_background(params.q_xy_range, params.q_z_range, npix=(img.shape[1], img.shape[0]),
                         amp=peak_max * bg_ratio, q_decay=float(rng.uniform(0.7, 1.5)),
                         halo_frac=float(rng.uniform(0.1, 0.6)), halo_q=float(rng.uniform(1.3, 1.7)),
                         halo_w=float(rng.uniform(0.15, 0.35)))
    expected = img * peak_max + bg
    noisy = add_detector_noise(expected, rng, gain_sigma=float(rng.uniform(0.005, 0.02)),
                               read_sigma=float(rng.uniform(0.0, 3.0)))

    # ---- detector geometry and missing wedge ----
    det, valid = apply_detector_geometry(noisy, params.q_xy_range, params.q_z_range, rng,
                                         beamstop_r=float(rng.uniform(0.05, 0.12)))
    valid_final = valid & ~wedge_mask(det.shape, params.q_xy_range, params.q_z_range, params.wavelength)
    img_final = np.where(valid_final, det, np.nan).astype(np.float32)

    meta["orientation"] = list(orient) if use_spots else None
    meta.update(shape=shape, mosaic_fwhm_deg=mosaic, n_peaks=len(I_sel), n_rings=len(I_ring),
                peak_max=peak_max, bg_ratio=bg_ratio)
    return dict(img=img_final, valid=valid_final, q=q_sel, I=I_sel, mi=mi_sel,
                ring_q=q_ring, ring_I=I_ring, meta=meta)
