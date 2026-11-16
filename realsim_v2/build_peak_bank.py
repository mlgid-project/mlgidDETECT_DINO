"""Peak bank in the format of the old realsim (realsim/physics_simulation.py) from the stored peak tables of the realsim_v2 shards.

    python realsim_v2/build_peak_bank.py OUT.npz [N_SHARDS|all] [SHARDS_DIR[,SHARDS_DIR2...]] [MAX_IMAGES]          # default: all shards of v2_60k

One entry per structure and kind: 'oriented' = the shard's spots, 'powder' = its rings (entry_cif = MaterialId, so a structure that was 'both' has two entries).
Also the stored peak widths sigma_q (1/A), sigma_chi (rad, NaN for rings) and the image's pseudo-Voigt eta per peak (used by realsim `v2_widths=True`).
Values are the stored ones: |q| in 1/A (move_fromMW=True positions of the generation), intensity as stored; chi in degrees FROM THE q_xy AXIS (bank convention) =
90 - degrees(chi_c), because the shard labels measure chi from the q_z axis; powder entries have chi = -1 as in the old bank.
Used by realsim/realbkg_simulation.py with `structure_frames=True` (one structure per frame) and GIWAXS_BANK=OUT.npz (realsim/factory.py).
"""
import sys, zipfile, io, json, glob, numpy as np
SH = '/mnt/DATA/mlgidDETECT_DINO_HEATMAP/realsim_v2_data/v2_60k/shards'


def build(out, n_shards=None, sh=SH, max_images=None):
    """sh = shard folder or comma-separated folders (taken in this order; a structure that is already in the bank is skipped, so each MaterialId has at most one oriented and one powder entry); max_images = stop when the bank holds that many images."""
    files = [f for d in str(sh).split(',') for f in sorted(glob.glob(f'{d}/shard_*.zip'))][:n_shards]
    seen, n_img = set(), 0
    q, chi, I, sq, sc, eta, start, count, kind, cif = [], [], [], [], [], [], [], [], [], []
    pos = 0
    for f in files:
        z = zipfile.ZipFile(f)
        for m in sorted(z.namelist()):
            if max_images is not None and n_img >= max_images:
                break
            d = np.load(io.BytesIO(z.read(m)))
            info = json.loads(str(d['info']))
            if info['MaterialId'] in seen:
                continue
            seen.add(info['MaterialId']); n_img += 1
            k = d['kind']
            for kk, name in ((0, 'oriented'), (1, 'powder')):
                sel = k == kk
                if sel.sum() < 1:
                    continue
                q.append(d['q_c'][sel].astype(np.float32)); I.append(d['intensity'][sel].astype(np.float32))
                sq.append(d['sigma_q'][sel].astype(np.float32)); sc.append(d['sigma_chi'][sel].astype(np.float32))     # 1/A and rad (NaN for rings)
                eta.append(np.full(sel.sum(), info['meta']['shape']['eta'], np.float32))                              # pseudo-Voigt mixing of the image (shared by its spots and rings)
                chi.append((90.0 - np.degrees(d['chi_c'][sel])).astype(np.float32) if kk == 0 else np.full(sel.sum(), -1.0, np.float32))
                start.append(pos); count.append(int(sel.sum())); kind.append(name); cif.append(info['MaterialId'])
                pos += int(sel.sum())
        print(f, len(start), 'entries', flush=True)
    np.savez(out, q=np.concatenate(q), chi=np.concatenate(chi), intensity=np.concatenate(I), sigma_q=np.concatenate(sq), sigma_chi=np.concatenate(sc), eta=np.concatenate(eta), entry_start=np.array(start, np.int64),
             entry_count=np.array(count, np.int64), entry_kind=np.array(kind), entry_cif=np.array(cif))
    print('images', n_img, 'entries', len(start), 'oriented', kind.count('oriented'), 'powder', kind.count('powder'), 'peaks', pos)


if __name__ == '__main__':
    build(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != 'all' else None, sys.argv[3] if len(sys.argv) > 3 else SH, int(sys.argv[4]) if len(sys.argv) > 4 else None)
