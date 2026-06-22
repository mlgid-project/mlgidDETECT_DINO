# Deploying the ssl1 + baseline ensemble

The best detector is an **ensemble of two checkpoints** (round-1 SSL `ssl1` + from-scratch
`baseline`) — they're complementary (baseline wins 41, ssl1 wins organic), and detection-level
fusion (pool each model's top-225, then class-aware NMS) WINS both eval sets:
**organic 0.605** (vs best single 0.586) · **41 0.780** (vs 0.768), every stratum up incl. faint
peaks. Verified in-repo by `ensemble_eval.py`. This doc covers exporting the two models to ONNX
and wiring the ensemble into mlgidDETECT.

## ONNX artifacts (validated — run in onnxruntime 1.19, CPU/CUDA)
```
/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/onnx/dino_ssl1.onnx
/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation/onnx/dino_baseline.onnx
```
input `images (1,1,512,1024)` float32 (= mlgidDETECT `converted_polar_image`), outputs
`pred_logits (1,900,2)`, `pred_boxes (1,900,4)`. (~1.15 GB each; artifacts live in the dataset
dir, not git.)

## Re-exporting (use `export_onnx_ensemble.py`, NOT `export_onnx.py`)
`export_onnx.py` traces on CPU, but this repo's MSDeformAttn custom op is CUDA-only and has no
ONNX symbolic → "Not implemented on the CPU". `backbone_curation/export_onnx_ensemble.py` rebinds
MSDeformAttn to its pure-PyTorch core (grid_sample, ONNX-traceable) for the export and sets
`export=True` (drops swin gradient checkpointing):
```bash
CUR=/mnt/lustre/work/schreiber/szb389/datasets/DINO_BACKBONE_curation
PYTHONPATH=$REPO python backbone_curation/export_onnx_ensemble.py \
    --checkpoint $CUR/detector_runs/dino_ssl1/checkpoint.pth --output $CUR/onnx/dino_ssl1.onnx
PYTHONPATH=$REPO python backbone_curation/export_onnx_ensemble.py \
    --checkpoint <baseline run>/checkpoint.pth          --output $CUR/onnx/dino_baseline.onnx
```

## mlgidDETECT integration (separate repo — done by hand on a new branch)
mlgidDETECT's `standard_postprocessing` uses the SAME `onnx_to_xyxy` + `filter_boxes` as this
repo, so the ensemble mirrors `ensemble_eval.py`: run two ONNX sessions, pool top-k, one NMS.

1. **`mlgiddetect/inference/inference.py:10`** — let `Inference.__init__` take an optional
   `model_path=None` (fall back to `path_utils.get_model_path(config)` when None) so two sessions
   can point at the two ONNX files.

2. **New `ensemble_postprocessing(img_container, raw_a, raw_b)`** (in
   `mlgiddetect/postprocessing/postprocessing.py`):
   - `onnx_to_xyxy` each raw result into the container; capture `(boxes, scores, pred_labels)`.
   - concat both pools into `img_container.boxes/scores/pred_labels`.
   - `filter_boxes(config, img_container)` (class-aware NMS over the union).
   - then the same `boxes_polar_to_reciprocal -> boxes_reciprocal_q_to_xy -> polar_to_cartesian`
     transforms `standard_postprocessing` runs.

3. **`mlgiddetect/evaluation/on_dataset.py:44`** — build two `Inference(config, model_path=...)`,
   and per image: `ensemble_postprocessing(img_container, inf_a.infer(img), inf_b.infer(img))`.
   Set `config.POSTPROCESSING_SCORE=0.1`, `config.POSTPROCESSING_CLASSAWARE_NMS=True`,
   `MODEL_TYPE='dino'` (ring/seg IoU defaults already in Config).

4. **Verify**: `eval_on_dataset` on organic + 41 → target **organic ≈ 0.605, 41 ≈ 0.780**, both
   above either single model. (Small deltas vs this repo possible: mlgidDETECT matches in q-space,
   the DINO repo in polar-pixel space — the ensemble should still beat each single model on both.)

For production inference, apply the same two-session + `ensemble_postprocessing` swap wherever
`Inference`/`standard_postprocessing` is called.
