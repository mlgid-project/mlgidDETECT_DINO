# Command reference — backbone curation, SSL pretraining, detector test

All the useful commands from this work, in one place. Cluster: **galvani** (SLURM),
GPU partition `a100-galvani` (max 72h = `3-00:00:00`). Env: **`DINO_GIWAXS`**.

---

## 0. Paths (paste this block into your shell first)
```bash
REPO=/mnt/lustre/home/schreiber/szb389/mlgidDETECT_DINO
PY=/home/schreiber/szb389/.conda/envs/DINO_GIWAXS/bin/python
CUR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation

SSL_RUN=$CUR/ssl_runs/simmim1                       # SSL training run dir
BB_DIR=$SSL_RUN/backbone_export                      # exported detector backbone weights
DET_RUN=$CUR/detector_runs/dino_ssl1                 # detector-with-SSL run dir
BASELINE=/mnt/lustre/work/schreiber/szb389/train_output/ringseg_2class_20260603-142434
H5_41=/mnt/lustre/work/schreiber/szb389/datasets/41.h5
H5_ORG=/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5
export PYTHONPATH=$REPO
```

---

## 1. SLURM / cluster
```bash
squeue -u $USER                                      # my jobs (ST: R=run, PD=pending, CG=completing)
squeue -u $USER -o "%.10i %.18j %.2t %.12M %.12L %R" # readable: id name state elapsed left node
scancel <JOBID>                                      # cancel a job
sinfo -o "%P %G %l" | grep -i a100                   # a100 partition + time limit
ssh <node>; nvidia-smi                               # GPU usage on the node (e.g. galvani-cn208)
```

---

## 2. SSL backbone pretraining (SimMIM)
```bash
# start (and resume — idempotent: auto-continues from last.pth, just resubmit each 72h session)
sbatch $REPO/backbone_curation/ssl/run_simmim.sbatch

# monitor
tail -f $SSL_RUN/log.csv                             # one line/epoch: epoch,train,val,lr,sec
tail -f $REPO/backbone_curation/ssl/slurm-*.out      # live per-iteration loss
cat   $SSL_RUN/log.csv                               # full curve
```
Checkpoints: `$SSL_RUN/best.pth` (lowest val) and `$SSL_RUN/last.pth` (every epoch).

### Round 2 (simmim2 — RECIPE_v2: mask 0.70, drop_path 0.25, --aug v2, 150 ep)
```bash
sbatch $REPO/backbone_curation/ssl/run_simmim2.sbatch          # writes ssl_runs/simmim2; simmim1 untouched
SSL2=$CUR/ssl_runs/simmim2
tail -f $SSL2/log.csv                                          # NOTE: val L1 reads HIGHER than simmim1
tail -f $REPO/backbone_curation/ssl/slurm-simmim2-*.out        #       by construction (harder mask) — judge
```                                                            #       only by downstream detector AP.

---

## 3. Export the SSL backbone for the detector
```bash
mkdir -p $BB_DIR
cp $SSL_RUN/best.pth $SSL_RUN/best_snapshot.pth      # freeze (training is still writing best.pth)
$PY $REPO/backbone_curation/ssl/export_backbone.py \
    --ckpt $SSL_RUN/best_snapshot.pth \
    --out  $BB_DIR/swin_large_patch4_window12_384_22k.pth
# expect: "load test: missing=0 unexpected=0 -> OK". File name/location already match the SSL config.
```
Re-run this on the final `best.pth` when SSL hits epoch 200, then resubmit the detector (§4).

---

## 4. Detector training with the SSL backbone
```bash
sbatch $REPO/backbone_curation/ssl/run_detector_ssl.sbatch   # own A100, runs parallel to SSL; auto-resumes
tail -f $REPO/backbone_curation/ssl/dino_ssl-*.out           # live log
grep "ap_total" $REPO/backbone_curation/ssl/dino_ssl-*.out   # per-epoch AP (41 + organic, every 2 epochs)
```
Uses `config/DINO/DINO_4scale_swin_ssl.py` (= base config + SSL backbone init). Output → `$DET_RUN`.

---

## 5. Eval & comparison vs baseline
```bash
# side-by-side ap_total curve (baseline vs SSL) at matched epochs — works the moment dino_ssl logs AP
$PY $REPO/backbone_curation/compare_ap.py $BASELINE $DET_RUN

# explicit final eval on each labeled set (prints AP DataFrames + ap_total)
cd $REPO
$PY main.py --eval --resume $DET_RUN -c config/DINO/DINO_4scale_swin_ssl.py \
    --eval_file $H5_41  --output_dir $DET_RUN/eval_out
$PY main.py --eval --resume $DET_RUN -c config/DINO/DINO_4scale_swin_ssl.py \
    --eval_file $H5_ORG --output_dir $DET_RUN/eval_out
```
**Baseline to beat (from-scratch):** 41 ap ~0.76 (best 0.768@ep338) · organic ap ~0.55 (best 0.554@ep360).
Win signals: faster early climb (visible by ep30–50), higher plateau, esp. a lift on **organic**.

```bash
# snapshot training state -> figures/training_state.png (SSL loss + detector AP vs baseline)
$PY $REPO/backbone_curation/plot_state.py            # reads live logs; re-run anytime for a fresh snapshot

# download the figure to your local machine (run LOCALLY, fill in your usual ssh user@host)
scp <user>@<galvani-login-host>:$REPO/backbone_curation/figures/training_state.png ~/Downloads/
```

---

## 6. Curation pipeline (already run — for reproduce/reference)
```bash
cd $REPO/backbone_curation
$PY extract_shards.py        # fingerprint all 27 tars  -> shard_parts/
$PY extract_eval.py          # fingerprint 41 + organic -> eval_fp.npz
$PY extract_profiles.py      # I(q)/I(chi) profiles     -> profile_parts/
$PY detect_leaks.py          # 2D candidate match (figures/leak_panels.png)
$PY iq_match.py              # q-calibrated I(q) leak match + validation
$PY combined_confirm.py      # I(q) AND I(chi) leak confirm  (verdict: NO leak)
$PY dedup.py                 # redundancy analysis (phash sweep)
$PY finalize_manifest.py     # keep-list 68,375 -> 12,991 -> manifest.tsv, keep_keys.txt
$PY corpus_builder.py        # build backbone_ssl_corpus.h5 (unprocessed)
```
Key outputs in `$CUR`: `backbone_ssl_corpus.h5` (12991×512×1024), `manifest.tsv`, `keep_keys.txt`.

---

## 7. Git
```bash
cd $REPO
git status -sb                                       # branch + sync state
git log --oneline -5
git branch -vv                                       # current: backbone-ssl -> origin/backbone-ssl
git add <files> && git commit -m "..." && git push   # only when there's repo code to commit
```

---

## Typical loop while jobs run
```bash
squeue -u $USER -o "%.10i %.18j %.2t %.12M %.12L %R"  # are they alive / time left?
tail -3 $SSL_RUN/log.csv                              # SSL val trend
grep "ap_total" $REPO/backbone_curation/ssl/dino_ssl-*.out | tail   # detector AP so far
# if a job's TIME_LEFT runs out before it's done -> just resubmit the same sbatch (auto-resume)
```
