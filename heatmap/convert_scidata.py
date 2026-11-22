"""Convert the raw Pilatus TIFFs of the Das et al. 2026 GIWAXS benchmark (Sci. Data, Zenodo 10.5281/zenodo.22657132) to pygid q-space NeXus, one merged h5 per beamtime run, for the mlgidBASE detect / fit / match pipeline.

  ~/miniconda3/envs/mlgidgui/bin/python heatmap/convert_scidata.py --root <giwaxs_scidata_raw> --out <dir> [--runs 1 3] [--limit N]

Frames: <root>/matched_frames.json (raw base frame <-> curated class, from the name matching of 2026-11-20), extracted TIFFs in <root>/frames/<archive>/.
Geometry (beamline_params.json of the dataset, validated on one PM6:L8-BO frame: radial maxima at q = 0.31 and 1.69 A^-1):
  run 1 (2024): wavelength 0.826561 A (15.0 keV per the beamline live log; beamline_params.json lists 1.07812 A = the 11.5 keV AgBeh calibration shot, which is wrong for the sample frames, checked on the alpha-FAPbI3 rings), distance 650.912 mm, beam centre x 1258.96, y 1623.31 FROM THE BOTTOM of the 1679-row image
  run 3 (2026): wavelength 0.826561 A, distance 652.67 mm, beam centre x 1088.39, y 1546.58 FROM THE BOTTOM
  pixel 172 um, Pilatus3 2M 1679 x 1475; PONI built here with poni1 = (1679 - y) * pixel (the beamstop sits near the top of the raw image), poni2 = x * pixel, no rotations; flips: fliplr True, flipud False; ai = filename angle (0p19 = 0.19 deg).
Output per run: <out>/run<k>.h5 with entry_0000 ... (q_xy / q_z 0 - 3.5 A^-1, dq 0.002, as the corpus), sample/name = curated class, title = raw file name, process/scidata = provenance; <out>/run<k>_index.json = entry -> raw file, class, angle."""
import argparse, json, os, re, shutil, sys, tempfile
from pathlib import Path
import h5py, numpy as np

RUNS = {1: dict(archive='GIWAXS_2024', wl=0.826561e-10,   # NOT the 1.07812 A of beamline_params.json: that is the 11.5 keV of the AgBeh calibration shot; 1360 of 1369 logged frames of the 2024 archive were taken at 15.0 keV (livelogfile.json Si111_monochromator_energy)
     dist=0.650912, cx=1258.96, cy_bottom=1623.31), 3: dict(archive='GIWAXS_2026', wl=0.826561e-10, dist=0.65267, cx=1088.39, cy_bottom=1546.58)}
PX, H, W = 1.72e-4, 1679, 1475

p = argparse.ArgumentParser()
p.add_argument('--root', required=True); p.add_argument('--out', required=True); p.add_argument('--runs', type=int, nargs='+', default=[1, 3]); p.add_argument('--limit', type=int, default=0)
a = p.parse_args()
import mlgidgui.conversion as _C
from mlgidgui.conversion import execute
_orig = _C._method_kwargs_for
_C._method_kwargs_for = lambda cfg, scan: {**_orig(cfg, scan), 'frame_num': None}          # pygid 0.2.20: single-frame loads must be converted with frame_num=None
from mlgidgui.conversion_config import ConversionConfig, RawScan, CONV_DET2Q_GID, GEOM_GID, OUTPUT_SEPARATE_FILES
from PIL import Image
root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
rows = json.load(open(root / 'matched_frames.json'))
for run in a.runs:
    P = RUNS[run]; sel = [r for r in rows if r['archive'].startswith(P['archive'])]
    if a.limit:
        sel = sel[:a.limit]
    work = Path(tempfile.mkdtemp(prefix=f'scidata_run{run}_', dir=str(out)))
    poni = work / 'run.poni'
    poni.write_text(f"poni_version: 2\nDetector: Detector\nDetector_config: {{\"pixel1\": {PX}, \"pixel2\": {PX}, \"max_shape\": [{H}, {W}]}}\nDistance: {P['dist']}\nPoni1: {(H - P['cy_bottom']) * PX}\nPoni2: {P['cx'] * PX}\n"
                    f"Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: {P['wl']}\n")
    done, index = [], {}
    for n, r in enumerate(sel):
        tifs = list((root / 'frames').rglob(r['file']))
        if not tifs:
            continue
        ang = re.search(r'_(\d+p\d+)_\d+\.tif$', r['file']); ai = float(ang.group(1).replace('p', '.'))
        img = np.asarray(Image.open(tifs[0])).astype(np.float32)
        if img.shape != (H, W):
            print('skip (shape)', r['file'], img.shape); continue
        rawf = work / f'raw{n:05d}.h5'
        with h5py.File(rawf, 'w') as h:
            h.create_dataset('entry_0000/measurement/data', data=img[None])
        cfg = ConversionConfig(geometry=GEOM_GID, conv_type=CONV_DET2Q_GID, vert_positive=True, hor_positive=True, dq=0.002, q_xy_range=(0.0, 3.5), q_z_range=(0.0, 3.5), poni_path=poni, ai=ai,
                               expmeta_overrides=dict(fliplr=True, flipud=False, transp=False), output_mode=OUTPUT_SEPARATE_FILES, output_dir=work / f'o{n:05d}', output_filename='f.h5', overwrite_file=True)
        try:
            res = execute([RawScan(file_path=rawf, entry='entry_0000/measurement/data', frame_num=[0])], cfg)
        except Exception as e:
            print('FAILED', r['file'], repr(e)[:200]); rawf.unlink(); continue
        rawf.unlink()
        done.append((n, r, ai, res[0]))
        if len(done) % 100 == 0:
            print(f'run {run}: {len(done)} frames converted', flush=True)
    with h5py.File(out / f'run{run}.h5', 'w') as fo:
        for k, (n, r, ai, f) in enumerate(done):
            name = f'entry_{k:04d}'
            with h5py.File(f, 'r') as fi:
                fi.copy(fi[list(fi.keys())[0]], fo, name=name)
            g = fo[name]
            for key in ('title',):
                if key in g:
                    del g[key]
            g['title'] = r['file']
            gs = g.require_group('sample');
            if 'name' in gs:
                del gs['name']
            gs['name'] = r['classes'][0]
            ps = g.require_group('process/scidata')
            for kk, v in dict(raw_file=r['file'], archive=r['archive'], curated_key=r['key'], cls=r['classes'][0], angle_of_incidence_deg=ai, run=run, distance_m=P['dist'], wavelength_m=P['wl'], beam_centre_x_px=P['cx'],
                              beam_centre_y_from_bottom_px=P['cy_bottom'], flips='fliplr True, flipud False', source='Das et al. 2026 Sci. Data, Zenodo 10.5281/zenodo.22657132').items():
                ps[kk] = v
            index[name] = dict(raw=r['file'], cls=r['classes'][0], ai=ai)
    json.dump(index, open(out / f'run{run}_index.json', 'w'))
    shutil.rmtree(work, ignore_errors=True)
    print(f'run {run}: {len(done)} entries written to {out}/run{run}.h5', flush=True)
