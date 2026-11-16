"""Detector masks for the simulator: real converted masks plus the GIWAXS missing wedge.

A frame's mask has two independent causes and they come from different places.

  THE DETECTOR. Module gaps, dead channels, the beamstop, and the falloff at the corners of a
  rectangular detector. These are geometry, so they were converted once from the real mask files
  and their poni calibrations -- 13 geometries across Eiger2 CdTe 4M, Eiger 4M, Perkin-Elmer and
  Lambda 750k, at distances 0.099 to 0.665 m and 15 to 101 keV. Measured against the polar frames
  we had been using, the converted masks reproduce the detector-edge falloff to within 0.01-0.05
  in reach at every chi, and both peak at chi ~ 42-48 deg, which is the corner of a rectangular
  detector seen from the beam.

  THE SAMPLE GEOMETRY. Near the q_z axis there is a region grazing incidence cannot reach at all.
  That is not a detector property and no poni file records it -- none of the delivered calibrations
  carries an incidence angle -- so a converted mask has a valid high-chi region where a real frame
  has nothing. The real frames fall from 0.66 valid at chi 80 deg to 0.11 at 89 deg; the converted
  masks stay near 0.74 all the way up. The missing wedge is therefore generated here instead, per
  frame, from the geometry.

WHY GENERATING IT IS BETTER THAN SHIPPING IT. The wedge depends on the incidence angle, which is
chosen per experiment; making it analytic turns it into an axis we can vary rather than something
frozen into each mask. It also means any mask can be combined with any incidence angle.

THE WEDGE. With k = 2pi/lambda and incidence angle a_i, a point at (q_xy, q_z) needs an exit angle
a_f with sin(a_f) = q_z/k - sin(a_i), and the smallest in-plane component it can have is

    q_xy_min(q_z) = k |cos(a_f) - cos(a_i)|

reached when the in-plane scattering angle goes to zero. Anything with q_xy below that is
inaccessible. Validated against the real frames at 18 keV, a_i = 0.2 deg:

    chi deg        83     85     87     88     89
    real frames   0.472  0.357  0.230  0.163  0.106
    this wedge    0.522  0.385  0.247  0.172  0.096

and it is 1.000 for chi below ~70 deg, so it does nothing except near the meridian. The incidence
angle is a weak lever (0.1 vs 0.5 deg moves those numbers by a few percent); the photon energy is
the strong one, and that comes from each mask's own calibration.
"""
import json
import os

import numpy as np

HEIGHT, WIDTH = 512, 1024
WORK = os.environ.get('GIWAXS_WORK', '/mnt/lustre/work/schreiber/szb389')
MASK_DIR = f'{WORK}/datasets/detector_masks/work'
ALPHA_I_DEG = (0.10, 0.50)      # grazing incidence, the usual thin-film range

#: The masks in use. Ten of the thirteen delivered geometries; three were dropped because their
#: q_max does not straddle what the CIF bank can supply (|q| <= 4.95 A^-1, median 2.55, p90 3.65).
#: Taking each mask's own q_max is the right thing -- its gaps sit at the q they sit at because of
#: that geometry -- but it ties the mask choice to peak coverage, and these three broke it:
#:   P21 Perkin-Elmer, 101 keV, q_max 20.79 -- every reflection lands in the left 24% of the frame
#:   P08 LISA Lambda 750k x2, q_max 1.34/1.35 -- only 11% of reflections fit at all
#: ID13 at D 0.099 m (q_max 7.80) is kept deliberately: peaks reach the left 64%, off but usable,
#: and it is the only short-distance geometry in the set.
KEEP_DEFAULT = ['ID10_Eiger2CdTe4M_D626_22p5keV_2024_03',
                'ID10_Eiger2CdTe4M_D327_20keV_2024_07',
                'ID10_Eiger2CdTe4M_D401_22p5keV_2024_12',
                'ID10_Eiger2_4M_D606_22p5keV_2025_04',
                'ID10_Eiger2CdTe4M_D405_22p5keV_2025_09',
                'ID10_Eiger2CdTe4M_D479_22p5keV_2026_03',
                'P08_Eiger4M_D629_25keV_2026_07',
                'P08_Eiger4M_D563_22p5keV_2026_07',
                'ID13_Eiger4M_D099_15keV_2024_06',
                'ID13_Eiger4M_D259_15keV_2024_06']


def missing_wedge(q_max, energy_keV, alpha_i_deg, height=HEIGHT, width=WIDTH):
    """-> bool (height, width), True where grazing incidence can reach (q, chi)."""
    lam = 12.398/float(energy_keV)
    k = 2*np.pi/lam
    ai = np.radians(float(alpha_i_deg))
    q = (np.arange(width) + 0.5)/width*float(q_max)
    chi = np.radians((np.arange(height) + 0.5)/height*90.0)
    qz = q[None, :]*np.sin(chi)[:, None]
    qxy = q[None, :]*np.cos(chi)[:, None]
    s = qz/k - np.sin(ai)
    af = np.arcsin(np.clip(s, -1.0, 1.0))
    qxy_min = k*np.abs(np.cos(af) - np.cos(ai))
    return (np.abs(s) <= 1.0) & (qxy >= qxy_min)


class MaskBank:
    """The converted detector masks, drawn one per frame and combined with a fresh wedge."""

    def __init__(self, path=MASK_DIR, keep='default', alpha_i=ALPHA_I_DEG, seed=None):
        self.rng = np.random.default_rng(seed)
        self.alpha_i = alpha_i
        recs = json.load(open(os.path.join(path, 'manifest.json')))['masks']
        n_all = len(recs)
        if keep == 'default':
            keep = KEEP_DEFAULT
        if keep is not None:
            keep = set(keep)
            recs = [r for i, r in enumerate(recs) if i in keep or r['id'] in keep]
        self.masks, self.meta = [], []
        for r in recs:
            f = os.path.join(path, r['id'], 'polar_mask.npy')
            if not os.path.exists(f):
                continue
            m = np.load(f).astype(bool)
            if m.shape != (HEIGHT, WIDTH):
                continue
            self.masks.append(m)
            self.meta.append(dict(id=r['id'], detector=r.get('detector'),
                                  q_max=float(r.get('q_max-1', r.get('q_max_A-1', 4.45))),
                                  energy_keV=float(r.get('energy_keV', 12.4)),
                                  distance_m=float(r.get('distance_m', 0.0))))
        if not self.masks:
            raise RuntimeError(f'no usable polar masks under {path}')
        print(f'[masks] {len(self.masks)} of {n_all} detector geometries, '
              f'q_max {min(m["q_max"] for m in self.meta):.2f}-'
              f'{max(m["q_max"] for m in self.meta):.2f}, '
              f'incidence angle {alpha_i[0]}-{alpha_i[1]} deg', flush=True)

    def draw(self):
        """-> (mask bool (512,1024), meta dict). Detector mask AND a fresh missing wedge.

        The q_max returned is the DETECTOR'S, not a borrowed one: the mask's gaps sit at the q
        they sit at because of that geometry, so the frame's q axis has to be the same one or the
        gaps land at the wrong q relative to the peaks.
        """
        i = int(self.rng.integers(len(self.masks)))
        md = dict(self.meta[i])
        md['alpha_i_deg'] = float(self.rng.uniform(*self.alpha_i))
        w = missing_wedge(md['q_max'], md['energy_keV'], md['alpha_i_deg'])
        return (self.masks[i] & w), md
