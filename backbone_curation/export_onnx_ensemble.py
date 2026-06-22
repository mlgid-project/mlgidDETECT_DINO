"""
Export a trained DINO detector checkpoint to ONNX — fixed for this repo's CPU-export gotcha.

The deployed export path (../export_onnx.py) traces the model on CPU, but MSDeformAttn's custom
op (MSDeformAttnFunction.apply -> CUDA kernel) is "Not implemented on the CPU" and has no ONNX
symbolic. This script rebinds MSDeformAttn to its pure-PyTorch core (ms_deform_attn_core_pytorch,
grid_sample-based -> ONNX-traceable, CPU-ok) for the export only, and sets export=True so the
swin backbone drops gradient checkpointing. Output ONNX is device-agnostic (runs on CPU or CUDA
under onnxruntime), input = (1, in_chans, 512, 1024), outputs = [pred_logits, pred_boxes].

Usage:
  PYTHONPATH=<repo> python backbone_curation/export_onnx_ensemble.py \
      --checkpoint <run>/checkpoint.pth --output <dir>/model.onnx
"""
import argparse, os, sys
from pathlib import Path
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# --- export-only patch: route MSDeformAttn through its pure-pytorch core ---
from models.dino.ops.functions import ms_deform_attn_func as _mf
import models.dino.ops.modules.ms_deform_attn as _msda


class _CoreShim:
    @staticmethod
    def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        # core ignores level_start_index / im2col_step (CUDA-kernel-only args)
        return _mf.ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights)


_msda.MSDeformAttnFunction = _CoreShim  # rebind the name used inside MSDeformAttn.forward

from export_onnx import build_model_main, export_model_to_onnx, _to_namespace  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--channels', type=int, default=1)
    p.add_argument('--height', type=int, default=512)
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--opset', type=int, default=16)
    a = p.parse_args()

    ck = torch.load(a.checkpoint, map_location='cpu')
    margs = _to_namespace(ck['args'])
    setattr(margs, 'export', True)   # disable swin gradient checkpointing for a clean trace
    model, _, _ = build_model_main(margs)
    model.load_state_dict(ck['model'])
    export_model_to_onnx(model, (1, a.channels, a.height, a.width),
                         Path(a.output), torch.device('cpu'), opset=a.opset)


if __name__ == '__main__':
    main()
