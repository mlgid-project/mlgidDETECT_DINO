"""Cache the SELECTED donor pool so a dump/inspection script starts in seconds, not minutes.

RealBkgSimulation._load_donors decompresses the whole bank (444 x 512 x 1024 float32, ~930 MB out
of a 597 MB gzipped file), then computes a tiled-MAD noise map and a lag-1 autocorrelation for
every one of the 444, and only then ranks and keeps 189. That is the right thing inside a training
job, where it is paid once for 500 epochs, but it dominates the cost of dumping 20 frames.

This script runs that selection ONCE and writes the derived arrays for the 189 kept donors.
Nothing is recomputed on load, so the numbers are identical to the training run's by construction
-- the cache stores the outputs of the same code, not a reimplementation.

  python diagnostics/cache_realbkg_donors.py [--config ...] [--out PATH]
"""
import os, sys, json, argparse
import numpy as np
import h5py

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

DEFAULT = '/mnt/lustre/work/schreiber/szb389/datasets/realbkg_donors_selected.h5'
MMDIR   = '/mnt/lustre/work/schreiber/szb389/datasets/realbkg_donors_mm'

# Plain .npy next to the HDF5, because gzip is the whole cost. The h5 cache is 877 MB compressed
# and takes ~4 minutes to inflate into the ~1.3 GB the pool occupies; the same arrays as raw .npy
# open through mmap_mode='r' in milliseconds and page in only the frames a script touches. The
# HDF5 stays the archival copy -- the memmap is derived from it and can be rebuilt any time.


def write_memmap(h5path=DEFAULT, mmdir=MMDIR):
    """Explode the HDF5 cache into plain .npy so it can be memory-mapped."""
    os.makedirs(mmdir, exist_ok=True)
    with h5py.File(h5path, 'r') as f:
        for k, dt in [('bkg', np.float32), ('mask', bool), ('noise', np.float32),
                      ('coef', np.float32), ('qmax', np.float32)]:
            np.save(os.path.join(mmdir, k + '.npy'), f[k][()].astype(dt))
        with open(os.path.join(mmdir, 'meta.json'), 'w') as g:
            g.write(json.dumps([json.loads(s) for s in f['meta'][()].astype(str)]))
    n = sum(os.path.getsize(os.path.join(mmdir, x)) for x in os.listdir(mmdir))
    print(f'wrote {mmdir}  ({n/1e6:.0f} MB, uncompressed)')
    return mmdir


def load_into(sim, path=DEFAULT, mmdir=MMDIR):
    """Attach a cached pool to a RealBkgSimulation instance. Returns True if a cache was used.

    Prefers the memmap directory: arrays are opened lazily, so a script that touches a handful of
    donors never pays for the rest. Falls back to the HDF5 cache, then to nothing.
    """
    if mmdir and os.path.exists(os.path.join(mmdir, 'bkg.npy')):
        ld = lambda k: np.load(os.path.join(mmdir, k + '.npy'), mmap_mode='r')
        sim.bkg, sim.mask = ld('bkg'), ld('mask')
        sim.noise, sim.coef = ld('noise'), ld('coef')
        sim.qmax = np.load(os.path.join(mmdir, 'qmax.npy'))
        sim.meta = json.load(open(os.path.join(mmdir, 'meta.json')))
        print(f'[cache] {len(sim.bkg)} donors memmapped from {mmdir}', flush=True)
        return True
    if not os.path.exists(path):
        return False
    with h5py.File(path, 'r') as f:
        sim.bkg   = f['bkg'][()].astype(np.float32)
        sim.mask  = f['mask'][()].astype(bool)
        sim.noise = f['noise'][()].astype(np.float32)
        sim.coef  = f['coef'][()].astype(np.float32)
        sim.qmax  = f['qmax'][()].astype(np.float32)
        sim.meta  = [json.loads(s) for s in f['meta'][()].astype(str)]
    print(f'[cache] {len(sim.bkg)} donors from {path}', flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/DINO/DINO_4scale_swin_realbkg.py')
    ap.add_argument('--out', default=DEFAULT)
    args = ap.parse_args()

    from util.slconfig import SLConfig
    from simulation import SimulationConfig
    from realbkg_simulation import RealBkgSimulation

    cfg = SLConfig.fromfile(os.path.join(_REPO, args.config))
    sc = SimulationConfig()
    coefs = getattr(cfg, 'box_coef_override', None) or (2.80, 1.30)
    sc.a_coef, sc.w_coef = float(coefs[0]), float(coefs[1])

    sim = RealBkgSimulation(
        bank_path=cfg.physics_bank_path, donor_path=cfg.realbkg_donor_path,
        stats_path=cfg.realbkg_stats_path, sim_config=sc, device='cpu',
        n_oriented=tuple(getattr(cfg, 'realbkg_n_oriented', (1, 3))),
        p_ring=float(getattr(cfg, 'realbkg_p_ring', 0.15)))

    with h5py.File(args.out, 'w') as f:
        f.create_dataset('bkg',   data=sim.bkg.astype(np.float32),   compression='gzip', compression_opts=1)
        f.create_dataset('mask',  data=sim.mask,                     compression='gzip', compression_opts=1)
        f.create_dataset('noise', data=sim.noise.astype(np.float32), compression='gzip', compression_opts=1)
        f.create_dataset('coef',  data=sim.coef.astype(np.float32),  compression='gzip', compression_opts=1)
        f.create_dataset('qmax',  data=np.asarray(sim.qmax, np.float32))
        f.create_dataset('meta',  data=np.array([json.dumps(m) for m in sim.meta],
                                                dtype=h5py.string_dtype()))
        f.attrs['source_bank'] = cfg.realbkg_donor_path
        f.attrs['note'] = 'SELECTED pool only; produced by RealBkgSimulation._load_donors itself.'
    print(f'wrote {args.out}  ({os.path.getsize(args.out)/1e6:.0f} MB)  {len(sim.bkg)} donors')
    write_memmap(args.out, MMDIR)


if __name__ == '__main__':
    main()
