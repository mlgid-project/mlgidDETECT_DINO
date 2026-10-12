"""Build the real-background simulator (realsim/realbkg_simulation.py) with the settings of the last DINO realbkg run (config r6).

Effective settings = DINO_4scale_swin_realbkg.py (base) <- ..._r3.py <- ..._r6.py on branch multi-channel-analysis, merged by hand below.
Anything can be overridden for sim development without touching code:
    make_realbkg_sim(overrides=dict(p_ring=0.4, contrast_min=2.5))      (heatmap/train.py: --realsim_opt p_ring=0.4 contrast_min=2.5)

Data layout (env GIWAXS_WORK, default the cluster path; on colorbox1 point it at the folder with the same sub-layout, see realsim/README.md):
    $GIWAXS_WORK/datasets/cif_library_organic/bank/bank_organic_hkl.npz     hkl bank (5.4 GB)
    $GIWAXS_WORK/datasets/sim_real_stats.npz                                peak statistics measured on real frames
    $GIWAXS_WORK/datasets/sim_background_bank4.h5                           donor bank (NOT read in mosaic mode; path only passed through)
    $GIWAXS_WORK/datasets/realbkg_donors_mm/{mask.npy,qmax.npy}             mosaic mask bank, q_max values
    $GIWAXS_WORK/datasets/detector_masks/work/{manifest.json,<id>/polar_mask.npy}
    $GIWAXS_WORK/tmp_diag/sim2/{donors_final.json,donor_cache_repaired.npz} the 90 clean donor frames (repaired cache)
"""
import os

# settings of run r6, as keyword arguments of RealBkgSimulation
R6 = dict(
    n_oriented=(1, 3), p_ring=0.60, n_powder=(1, 2),
    unified_labels=True, contrast_min=2.0, snr_min=6.0, seg_iou_max=0.30, ring_iou_max=0.10,
    max_peaks=200, spots_cap=(2, 70), rings_cap=(1, 13),
    mosaic=True, mosaic_pool=48, mosaic_refresh=64, mosaic_seed=None,
    intensity_decades=(3.0, 7.3010), amplitude_mode='pygid', mask_bank=True, mask_keep='default',
    ring_box_from_mask=True, seg_wide_frac=0.50, seg_wide_sigma=((8.1, 0.45), (2.9, 0.45)),
    bkg_v2=True, surrogate_frac=0.5, v2_canvas=(2048, 4096), v2_refresh=25, v2_n_pc=6)


def work_dir():
    return os.environ.get('GIWAXS_WORK', '/mnt/lustre/work/schreiber/szb389')


def make_realbkg_sim(device='cuda', box_coef=(2.80, 1.30), overrides=None, bank_path=None, stats_path=None):
    from realsim.simulation_mc import SimulationConfig
    from realsim.realbkg_simulation import RealBkgSimulation
    W = work_dir()
    cfg = SimulationConfig()
    cfg.a_coef, cfg.w_coef = float(box_coef[0]), float(box_coef[1])
    kw = dict(R6)
    kw.update(overrides or {})
    unknown = set(kw) - set(RealBkgSimulation.__init__.__code__.co_varnames)
    if unknown:
        raise ValueError(f'unknown RealBkgSimulation option(s): {sorted(unknown)}')
    sim = RealBkgSimulation(
        bank_path=bank_path or f'{W}/datasets/cif_library_organic/bank/bank_organic_hkl.npz',
        donor_path=f'{W}/datasets/sim_background_bank4.h5',
        stats_path=stats_path or f'{W}/datasets/sim_real_stats.npz',
        sim_config=cfg, device=device, **kw)
    print(f'[realsim] real-background sim ON (r6 settings, overrides {overrides or {}}), box convention a_coef,w_coef = {box_coef}', flush=True)
    return sim
