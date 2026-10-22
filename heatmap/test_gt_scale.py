"""Check of the GT box scaling in heatmap/evaluation.py (synthetic boxes; needs the evaluation imports, so run on a compute node)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import types
for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
    _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
from heatmap import evaluation as E

mask = np.ones(E.POLAR, bool)
seg = [100., 200., 110., 240.]                       # segment: 10 px wide (x, radial), 40 px high (y, angular), centre (105, 220)
ring = [300., 0., 310., 512.]                        # ring: full chi
gt = np.array([seg, ring], np.float32)
class IC:                                            # minimal stand-in for an image container
    class polar_labels: boxes = gt; confidences = np.array([1., 1.], np.float32)
    converted_mask = mask
for coef in [None, (2, 2), (2.5, 2.5), (2.8, 1.3), (3.0, 3.0)]:
    E.set_gt_box_coef(*(coef if coef else (None, None)))
    g = E.gt_of(IC)
    print('coef', coef, 'scale', E.GT_SCALE, '| seg', g['gt'][0].round(2).tolist(), '| ring', g['gt'][1].round(2).tolist(), '| raw unchanged', bool(np.array_equal(g['gt_raw'], gt)))
# expected: (2,2) = identical to None; (2.5,2.5): seg half widths 5 -> 6.25, 20 -> 25; ring x half width 5 -> 6.25, y unchanged; (2.8,1.3): x scale 0.65, y scale 1.4
E.set_gt_box_coef(2.8, 1.3); g = E.gt_of(IC)
assert np.allclose(g['gt'][0], [105 - 5 * 0.65, 220 - 20 * 1.4, 105 + 5 * 0.65, 220 + 20 * 1.4]), g['gt'][0]
assert np.allclose(g['gt'][1], [305 - 5 * 0.65, 0, 305 + 5 * 0.65, 512]), g['gt'][1]
E.set_gt_box_coef(2, 2); assert np.array_equal(E.gt_of(IC)['gt'], gt)
print('OK')
