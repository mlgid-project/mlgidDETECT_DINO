"""Convert the frames of selection.json (select_label_set.py) from the RAW detector files to q-space NeXus with pygid, the way mlgidGUI does it, into ONE h5 file (entry_0000 ... ) that mlgidGUI opens for labelling.

  ~/miniconda3/envs/mlgidgui/bin/python heatmap/convert_label_set.py --selection <selection.json> --out <file.h5> [--workdir <dir>] [--moneta /mnt/moneta_folder]

Per frame (the settings are those of the beamtime's own result files, PROCESSED_DATA/<run>_result.h5, pygid 0.2.10, checked against them):
  geometry   the beamtime's calibration (PONI, LaB6): PROCESSED_DATA/LaB6_new.poni (2024_07_ESRF_ID10_Dima; distance 327.166 mm, 20 keV = wavelength 0.6199 A, Eiger2 CdTe 4M), fliplr + flipud, no transpose
  angle      angle of incidence = the motor 'om' of that frame in the scan's master file (an incidence scan: om changes from frame to frame), degrees
  q map      GID q_xy / q_z, both 0 - 3.5 A^-1, dq 0.002 (1750 x 1750), as in the result files
  metadata   instrument / process groups written by pygid; the sample group is copied from the result file of the same sample; start_time / end_time from the master file; process/label_set holds the
             provenance (raw file, frame, scan, region, cluster size, corpus h5 + frame)
The frames are converted one by one (one angle each), then merged into the output file in the order of the selection (n = 1 is entry_0000)."""
import argparse, glob, json, os, re, shutil, sys, tempfile
from pathlib import Path
import h5py, numpy as np

p = argparse.ArgumentParser()
p.add_argument('--selection', required=True); p.add_argument('--out', required=True); p.add_argument('--workdir', default=None)
p.add_argument('--moneta', default='/mnt/moneta_folder'); p.add_argument('--limit', type=int, default=0)
a = p.parse_args()
import mlgidgui.conversion as _C
from mlgidgui.conversion import execute
_orig_kw = _C._method_kwargs_for
# pygid loads only the requested raw frame (Conversion(frame_num=[i])) and then indexes the loaded one-frame stack with the SAME frame number -> IndexError for i > 0 (pygid 0.2.20, also through the GUI's
# execute). The conversion method must be called with frame_num=None (= every loaded frame); the stored data/frame_num still records the raw frame index.
_C._method_kwargs_for = lambda cfg, scan: {**_orig_kw(cfg, scan), 'frame_num': None}
from mlgidgui.conversion_config import ConversionConfig, RawScan, CONV_DET2Q_GID, GEOM_GID, OUTPUT_SEPARATE_FILES
sel = json.load(open(a.selection))
if a.limit:
    sel = sel[:a.limit]
work = Path(a.workdir or tempfile.mkdtemp(prefix='label_set_')); work.mkdir(parents=True, exist_ok=True)
BT = {'2024_07_ESRF_ID10_Dima': 'id10/20240711'}
tmp = []
for r in sel:
    folder = Path(a.moneta + r['folder']); raw = folder / 'eiger4m_0000.h5'
    frame = int(r['raw_name'].split('#')[1])
    parts = folder.parts; run = parts[-2]; sample = parts[-3]; scan_no = int(re.search(r'scan(\d+)', parts[-1]).group(1)); rawroot = folder.parents[2]
    master = glob.glob(str(rawroot / sample / f'sc*_{sample}.h5'))
    assert master, f'no master file for {sample}'
    with h5py.File(master[0], 'r') as m:
        g = m[f'{run}_{scan_no}.1']
        om = float(np.asarray(g['measurement/om'])[frame]); t0 = g['start_time'][()].decode(); t1 = g['end_time'][()].decode(); title = g['title'][()].decode()
        energy = float(np.asarray(g['measurement/energy'])[frame]) if 'energy' in g['measurement'] else None
    beam = BT[r['beamtime']]
    poni = Path(a.moneta + f'/srv/groupshare/OMBD/xray/beamtimes/{r["beamtime"]}/{beam}/PROCESSED_DATA/LaB6_new.poni')
    cfg = ConversionConfig(geometry=GEOM_GID, conv_type=CONV_DET2Q_GID, vert_positive=True, hor_positive=True, dq=0.002, q_xy_range=(0.0, 3.5), q_z_range=(0.0, 3.5),
                           poni_path=poni, ai=om, expmeta_overrides=dict(fliplr=True, flipud=True, transp=False), output_mode=OUTPUT_SEPARATE_FILES, output_dir=work / f'f{r["n"]:02d}',
                           output_filename=f'frame{r["n"]:02d}.h5', overwrite_file=True)
    print(f'[{r["n"]}] {sample}/{run}/scan{scan_no:04d} frame {frame}: ai(om) {om:.4f} deg, energy {energy} keV, scan "{title}"', flush=True)
    out = execute([RawScan(file_path=raw, entry='entry_0000/measurement/data', frame_num=[frame])], cfg)
    tmp.append((r, out[0], dict(om=om, start=t0, end=t1, title=title, sample=sample, run=run, scan=scan_no, master=master[0], frame=frame, raw=str(raw), beam=beam)))
    print('   ->', out[0], flush=True)
# merge
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with h5py.File(a.out, 'w') as fo:
    for r, f, meta in tmp:
        with h5py.File(f, 'r') as fi:
            ent = list(fi.keys())[0]
            name = f'entry_{r["n"] - 1:04d}'
            fi.copy(fi[ent], fo, name=name)
            g = fo[name]
        # sample group from the result file of the same sample (same beamtime): the metadata that exists for it
        res = Path(a.moneta + f'/srv/groupshare/OMBD/xray/beamtimes/{r["beamtime"]}/{meta["beam"]}/PROCESSED_DATA/{meta["run"]}_result.h5')
        src = 'pygid default (no result file for this sample)'
        if res.exists():
            with h5py.File(res, 'r') as fr:
                e = fr['entry']
                if 'sample' in e:
                    if 'sample' in g:
                        del g['sample']
                    fr.copy(e['sample'], g, name='sample'); src = str(res)
        if src.startswith('pygid default'):                    # no result file for this sample: only the name is known (the folder name), nothing is invented
            gs = g.require_group('sample'); gs['name'] = meta['sample']; src = 'none: no result file for this sample, only the folder name written as sample/name'
        for k, v in (('start_time', meta['start']), ('end_time', meta['end'])):
            if k in g:
                del g[k]
            g[k] = v
        if 'title' in g:
            del g['title']
        g['title'] = f'R3 #{r["n"]} {meta["sample"]} scan{meta["scan"]:04d} frame {meta["frame"]}'
        ls = g.require_group('process/label_set')
        for k, v in dict(raw_file=meta['raw'], raw_frame=meta['frame'], scan_title=meta['title'], master_file=meta['master'], sample_metadata_from=src, region=r['region'], far_cluster_size=r['cluster'],
                         corpus_h5=r['corpus_h5'], corpus_frame=r['corpus_idx'], poni=f'{a.moneta}/srv/groupshare/OMBD/xray/beamtimes/{r["beamtime"]}/{meta["beam"]}/PROCESSED_DATA/LaB6_new.poni',
                         angle_of_incidence_source='measurement/om of the scan master file').items():
            ls[k] = v
print('written', a.out, len(tmp), 'entries')
