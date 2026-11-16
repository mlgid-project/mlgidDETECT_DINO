"""Training-time reader for the realsim_v2 shards written by generate_dataset.py (this folder).

    sim = RealSimV2('/mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_v2_data/<name>', box_coef=(3.0, 3.0), flip='random', seed=42)
    img, boxes, mask, is_ring = sim.simulate_img()      # same contract as realsim.realbkg_simulation.RealBkgSimulation.simulate_img

img     float32 [512, 1024]  raw counts -> deployed contrast (clip 5/99.5 percentile, log10, histogram equalisation; util.exp_preprocess.apply_contrast
                              with the mlgidDETECT defaults), masked pixels 0 -- the same input the validation frames get (ic.converted_polar_image)
boxes   float32 [N, 4]       xyxy pixels, rebuilt from the stored sigmas for the training box convention (see box_coef)
mask    bool    [512, 1024]  True = valid
is_ring bool    [N]          kind == 1

box_coef = (a, w) as in heatmap/train.py --box_coef: FULL box width in sigma, a = angular (chi), w = radial (|q|). The shard labels store sigmas (not boxes), so
any convention can be built here; the stored `boxes` array (ratio of the generation config) is not used.
flip    : 'random' = the augmentation of the legacy simulator (simulation.flip_image): independently with probability 0.5 flip the image vertically (chi rows) and
          horizontally (|q| columns), boxes and mask with it; 'none' = the stored layout (row 0 = chi 0 deg when the generation config says flip_chi=False).

Images are drawn without replacement from a seeded permutation of all images in the folder (so an epoch never repeats an image until all were used).
Shards are ZIP_STORED zips of .npz members: random access, one open handle per shard per process.
"""
import io, json, os, sys, zipfile, pathlib, threading, queue
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# deployed contrast (the 'log_he' channel of heatmap/evaluation.py CONTRAST_CHANNELS; at these settings apply_contrast == mlgidDETECT _contrast_correction)
DEPLOYED = {'name': 'log_he', 'clip': (5.0, 99.5), 'log': True, 'gamma': None, 'he': True, 'clahe': None}
SHAPE = (512, 1024)


def index_shards(folder):
    """[(zip path, member name)] of every image in <folder>/shards, sorted."""
    items = []
    for z in sorted((pathlib.Path(folder) / 'shards').glob('shard_*.zip')):
        with zipfile.ZipFile(z) as zf:
            items += [(str(z), n) for n in sorted(zf.namelist())]
    return items


class RealSimV2:
    def __init__(self, folder, box_coef=(3.0, 3.0), flip='random', seed=0, prefetch=0, contrast=DEPLOYED, device='cuda'):
        import polar_labels as P
        self.P, self.folder, self.device = P, str(folder), device
        self.cfg = json.loads((pathlib.Path(folder) / 'config.json').read_text())
        self.box_coef = (float(box_coef[0]), float(box_coef[1]))
        assert flip in ('random', 'none'), flip
        self.flip = flip
        self.stored_flip = bool(self.cfg.get('flip_chi', False))
        self.contrast = contrast
        self.items = index_shards(folder)
        if not self.items:
            raise RuntimeError(f'no shards in {folder}/shards')
        self.rng = np.random.default_rng(seed)
        self._order, self._pos = self.rng.permutation(len(self.items)), 0
        self._zips = {}
        self._q = None
        if prefetch:                                         # background thread: keeps `prefetch` finished frames ready (zip read, contrast and boxes release the GIL mostly)
            self._q = queue.Queue(prefetch)
            threading.Thread(target=self._fill, daemon=True).start()
        print(f'[realsim_v2] {len(self.items)} images from {folder}; box_coef (angular, radial) = {self.box_coef}; flip = {self.flip} (stored layout flip_chi = {self.stored_flip})', flush=True)

    def __len__(self):
        return len(self.items)

    def _read(self, k):
        path, name = self.items[k]
        zf = self._zips.get(path)
        if zf is None:
            zf = self._zips[path] = zipfile.ZipFile(path)
        return np.load(io.BytesIO(zf.read(name)), allow_pickle=False)

    def load(self, k):
        """Image k of the folder -> (img, boxes, mask, is_ring) as numpy arrays (see module docstring)."""
        from util.exp_preprocess import apply_contrast
        P, d = self.P, self._read(k)
        raw = d['img'].astype(np.float32)
        mask = np.unpackbits(d['mask'])[:SHAPE[0] * SHAPE[1]].reshape(SHAPE).astype(bool)
        lab = dict(kind=d['kind'], q_c=d['q_c'].astype(np.float64), chi_c=d['chi_c'].astype(np.float64),
                   sigma_q=d['sigma_q'].astype(np.float64), sigma_chi=d['sigma_chi'].astype(np.float64))
        a, w = self.box_coef
        # boxes in the layout of the stored image first, then flip together with the image
        boxes = P.boxes_from_labels(lab, ratio=(w, a), shape=SHAPE, q_max=float(json.loads(str(d['info']))['q_max']), fmt='px_xyxy', flip_chi=self.stored_flip)
        img = apply_contrast(raw, mask, self.contrast).astype(np.float32)
        img[~mask] = 0.0
        if self.flip == 'random':                            # same as simulation.flip_image: two independent coin flips
            if self.rng.random() > 0.5:                      # vertical: chi rows
                img, mask = img[::-1], mask[::-1]
                boxes = np.column_stack([boxes[:, 0], SHAPE[0] - boxes[:, 3], boxes[:, 2], SHAPE[0] - boxes[:, 1]]) if len(boxes) else boxes
            if self.rng.random() > 0.5:                      # horizontal: |q| columns
                img, mask = img[:, ::-1], mask[:, ::-1]
                boxes = np.column_stack([SHAPE[1] - boxes[:, 2], boxes[:, 1], SHAPE[1] - boxes[:, 0], boxes[:, 3]]) if len(boxes) else boxes
            img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
        return img, boxes.astype(np.float32).reshape(-1, 4), mask, (lab['kind'] == 1)

    def _next_index(self):
        if self._pos >= len(self._order):
            self._order, self._pos = self.rng.permutation(len(self.items)), 0
        k = int(self._order[self._pos]); self._pos += 1
        return k

    def _fill(self):
        while True:
            try:
                out = self.load(self._next_index())
            except Exception as e:                           # a bad member must not stop the training
                print(f'[realsim_v2] skipped an image: {type(e).__name__}: {e}', flush=True)
                continue
            if len(out[1]) == 0:
                continue
            self._q.put(out)

    def simulate_img(self):
        """One frame as torch tensors, contract of RealBkgSimulation.simulate_img: (img [H,W] float32, boxes [N,4] xyxy, mask bool, is_ring bool), all on `device`."""
        import torch
        while True:
            if self._q is not None:
                out = self._q.get()
            else:
                try:
                    out = self.load(self._next_index())
                except Exception as e:
                    print(f'[realsim_v2] skipped an image: {type(e).__name__}: {e}', flush=True)
                    continue
                if len(out[1]) == 0:
                    continue
            img, boxes, mask, ring = out
            return tuple(torch.from_numpy(v).to(self.device) for v in (img, boxes, mask, ring))
