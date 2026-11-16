"""Batch generation of realistic GIWAXS training images (polar 512 x 1024, raw counts + mask + box labels).

    python generate_dataset.py --out /mnt/DATA/giwaxs_sim/test100 --n 100 --shard-size 25
    python generate_dataset.py --out /mnt/DATA/giwaxs_sim/full --n 500000        # resumes: finished shards are skipped

One structure per image: image `i` uses structure `perm[i]` of the GNoME rows with NSites in [2, 200] (fixed permutation)
and the random generator default_rng([BASE_SEED, i]), so any image can be regenerated with `generate_image(i, cfg)`.

Output: <out>/config.json, <out>/shards/shard_00000.zip (each member <idx>.npz, ZIP_STORED, random access),
<out>/failures/shard_00000.jsonl. A shard is written as .tmp and renamed when complete.

Each npz holds: img float32 (512, 1024) raw counts with masked pixels = 0, mask bit-packed (np.unpackbits(mask)[:512*1024]
reshaped to (512, 1024); 1 = valid), kind/src/q_c/chi_c/sigma_q/sigma_chi/intensity/snr/signal (per label), mi int16 (N, 3)
(zeros for rings), boxes float32 (N, 4) in `box_fmt` (pixels of img as stored), info (JSON string: MaterialId, idx, seed, mode, meta).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                      # one core per worker, no oversubscription

import argparse, io, json, sys, time, traceback, zipfile
import multiprocessing as mp
import pathlib
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

GNOME = pathlib.Path("/mnt/DATA/gnome")
DEFAULT_CFG = dict(
    nsites=[2, 200], base_seed=20261102, perm_seed=7, shard_size=250,
    q_range=[0.0, 5.0], energy=18000, max_index=2, polar_shape=[512, 1024],
    mode_probs={"oriented": 0.25, "both": 0.5, "rings": 0.25},
    box_ratio=[3.0, 3.0], box_fmt="px_cxcywh", flip_chi=False,
    iou_ss=0.1, iou_rr=0.1, iomin_sr=0.0, min_snr=[1.1, 1.1],
)

_W = {}                                                 # per-process cache


def _setup(cfg):
    """Load the table, the permutation, the zip and the simulation parameters once per process."""
    if _W.get("cfg") == cfg:
        return _W
    import pandas as pd
    from pygidsim import ExpParameters
    df = pd.read_csv(GNOME / "stable_materials_summary.csv", index_col=0)
    df = df[df["NSites"].between(*cfg["nsites"])].reset_index(drop=True)
    ids = df["MaterialId"].to_numpy()
    perm = np.random.default_rng(cfg["perm_seed"]).permutation(len(ids))
    q0, q1 = cfg["q_range"]
    _W.clear()
    _W.update(cfg=cfg, ids=ids, perm=perm, zip=zipfile.ZipFile(GNOME / "by_id.zip"),
              params=ExpParameters(q_xy_range=(q0, q1), q_z_range=(q0, q1), en=cfg["energy"]))
    return _W


def n_structures(cfg):
    return len(_setup(cfg)["ids"])


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


def generate_image(idx, cfg):
    """Deterministic: (idx, cfg) -> dict of arrays + info. Raises on failure."""
    import polar_labels as P
    from pipeline import make_sample
    W = _setup(cfg)
    mid = str(W["ids"][W["perm"][idx]])
    rng = np.random.default_rng([cfg["base_seed"], idx])
    modes = list(cfg["mode_probs"])
    mode = str(rng.choice(modes, p=[cfg["mode_probs"][m] for m in modes]))
    t0 = time.time()
    with tempfile.TemporaryDirectory(dir="/dev/shm" if os.path.isdir("/dev/shm") else None) as tmp:
        cif = pathlib.Path(tmp) / f"{mid}.CIF"
        cif.write_bytes(W["zip"].read(f"by_id/{mid}.CIF"))
        res = make_sample(str(cif), mode, rng, W["params"], orientation="random", max_index=cfg["max_index"])
    pol, q_max = P.to_polar(res, W["params"], shape=tuple(cfg["polar_shape"]))
    ratio, flip = tuple(cfg["box_ratio"]), cfg["flip_chi"]
    lab = P.box_labels(res, W["params"], shape=tuple(cfg["polar_shape"]), q_max=q_max, pol=pol, ratio=ratio,
                       min_snr=cfg["min_snr"], iou_ss=cfg["iou_ss"], iou_rr=cfg["iou_rr"], iomin_sr=cfg["iomin_sr"])
    boxes = P.boxes_from_labels(lab, ratio=ratio, shape=tuple(cfg["polar_shape"]), q_max=q_max,
                                fmt=cfg["box_fmt"], flip_chi=flip)
    img, mask = P.to_raw_output(pol, flip_chi=flip)
    n = len(lab["kind"])
    mi = np.zeros((n, 3), dtype=np.int16)
    for k, m in enumerate(lab["mi"]):
        if m is not None:
            mi[k] = np.rint(np.asarray(m).reshape(-1, 3)[0])
    info = dict(MaterialId=mid, idx=int(idx), seed=[cfg["base_seed"], int(idx)], mode=mode, q_max=q_max,
                n_spots=int((lab["kind"] == 0).sum()), n_rings=int((lab["kind"] == 1).sum()),
                seconds=round(time.time() - t0, 2), meta=_jsonable(res["meta"]))
    arrays = dict(img=img, mask=np.packbits(mask), kind=lab["kind"], src=lab["src"].astype(np.int32),
                  q_c=lab["q_c"].astype(np.float32), chi_c=lab["chi_c"].astype(np.float32),
                  sigma_q=lab["sigma_q"].astype(np.float32), sigma_chi=lab["sigma_chi"].astype(np.float32),
                  intensity=lab["intensity"].astype(np.float32), snr=lab["snr"].astype(np.float32),
                  signal=lab["signal"].astype(np.float32), mi=mi, boxes=boxes.astype(np.float32),
                  info=np.array(json.dumps(info)))
    return arrays


def run_shard(args):
    shard, n_total, out, cfg = args
    out = pathlib.Path(out)
    final = out / "shards" / f"shard_{shard:05d}.zip"
    if final.exists():
        return shard, 0, 0, 0.0
    tmp = final.with_suffix(".zip.tmp")
    fail_path = out / "failures" / f"shard_{shard:05d}.jsonl"
    ok = bad = 0
    t0 = time.time()
    fails = []
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for idx in range(shard * cfg["shard_size"], min((shard + 1) * cfg["shard_size"], n_total)):
            try:
                arrays = generate_image(idx, cfg)
                buf = io.BytesIO()
                np.savez_compressed(buf, **arrays)
                zf.writestr(f"{idx:08d}.npz", buf.getvalue())
                ok += 1
            except Exception as e:
                bad += 1
                fails.append(dict(idx=idx, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-600:]))
    if fails:
        fail_path.write_text("".join(json.dumps(f) + "\n" for f in fails))
    os.replace(tmp, final)
    return shard, ok, bad, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True, help="number of images (indices 0..n-1)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=DEFAULT_CFG["shard_size"])
    a = ap.parse_args()
    cfg = dict(DEFAULT_CFG, shard_size=a.shard_size)
    out = pathlib.Path(a.out)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    (out / "failures").mkdir(exist_ok=True)
    cfg_path = out / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old != cfg:
            sys.exit(f"{cfg_path} differs from the requested configuration; use a new --out directory")
    else:
        cfg_path.write_text(json.dumps(cfg, indent=1))
    n_str = n_structures(cfg)
    if a.n > n_str:
        sys.exit(f"--n {a.n} exceeds the {n_str} structures (one structure per image)")
    for f in (out / "shards").glob("*.tmp"):                    # leftovers of an interrupted run
        f.unlink()
    n_shards = -(-a.n // cfg["shard_size"])
    todo = [(s, a.n, str(out), cfg) for s in range(n_shards) if not (out / "shards" / f"shard_{s:05d}.zip").exists()]
    print(f"{n_str} structures, {a.n} images, {n_shards} shards, {n_shards - len(todo)} done, {len(todo)} to do", flush=True)
    t0, done_ok, done_bad = time.time(), 0, 0
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=40) as pool:
        for k, (s, ok, bad, sec) in enumerate(pool.imap_unordered(run_shard, todo), 1):
            done_ok += ok; done_bad += bad
            el = time.time() - t0
            print(f"shard {s:5d}: {ok} ok, {bad} failed, {sec:.0f} s | {k}/{len(todo)} shards, {done_ok} images, "
                  f"{done_bad} failed, elapsed {el / 60:.1f} min, eta {el / k * (len(todo) - k) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
