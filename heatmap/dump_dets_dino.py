"""Cache a DINO detector checkpoint's detections (top-225 + class-aware NMS, score floor 0.1: E.dino_dets) in the dump_dets.py layout, so that
heatmap/compare_figs.py can draw it next to heatmap models. The compiled MSDeformAttn CUDA ops are NOT needed: MSDeformAttn is routed through
its pure-PyTorch core (as backbone_curation/export_onnx_ensemble.py does for the export).

  python heatmap/dump_dets_dino.py --ckpt <checkpoint.pth> --out <file>.pkl --box_coef 2.80,1.30 [--sets organic 41] [--device cuda]
--box_coef a_coef,w_coef = box convention the DINO model was trained with: the stored GT (+-1 sigma) is scaled to it (GT scale (x,y) = (w_coef/2, a_coef/2), rings in x only),
the same as HM_GT_BOX_COEF in evaluation.py. Frame layout as dump_dets.py: img [512,1024] f16, mask, gt (scaled), gtconf, boxes (polar px), scores, cls
(DINO has no usable ring/segment head here: cls = 1 if the box spans >= 70% of the valid chi range, else 0 = E.is_ring_geom)."""
import os, sys, argparse, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch


def patch_msdeform():
    """route MSDeformAttn through ms_deform_attn_core_pytorch (grid_sample), before any model is built."""
    import types
    sys.modules.setdefault('MultiScaleDeformableAttention', types.ModuleType('MultiScaleDeformableAttention'))      # compiled ops absent: stub so the import works, never called
    from models.dino.ops.functions import ms_deform_attn_func as _mf
    import models.dino.ops.modules.ms_deform_attn as _msda

    class _CoreShim:
        @staticmethod
        def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
            return _mf.ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
    _msda.MSDeformAttnFunction = _CoreShim


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
    p.add_argument('--box_coef', required=True); p.add_argument('--sets', nargs='*', default=['organic', '41'])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    patch_msdeform()
    a_coef, w_coef = (float(v) for v in a.box_coef.split(','))
    os.environ['HM_GT_BOX_COEF'] = a.box_coef
    from heatmap import evaluation as E
    print('GT box scale (x,y):', E.set_gt_box_coef(a_coef, w_coef), flush=True)
    import util.misc as utils
    from models.registry import MODULE_BUILD_FUNCS      # (main.py itself pulls in addict etc.)
    import models.dino      # registers the 'dino' build function
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    args = ck['args'] if not isinstance(ck['args'], dict) else argparse.Namespace(**ck['args'])
    args.device = a.device
    if hasattr(args, 'backbone_dir'):      # the weights come from the checkpoint: build_backbone only loads pretrained swin weights when `backbone_dir in args`
        delattr(args, 'backbone_dir')
    for k in ('pretrain_model_path', 'finetune_ignore'):
        if hasattr(args, k):
            setattr(args, k, None)
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    model, _, _ = MODULE_BUILD_FUNCS.get(args.modelname)(args)
    print('load:', model.load_state_dict(utils.clean_state_dict(ck.get('model', ck)), strict=False), 'epoch', ck.get('epoch'), flush=True)
    model = model.to(a.device).eval()
    nch = getattr(args, 'num_channels', 1)
    sets = {}
    for ds in a.sets:
        fr = []
        for cfg, ic in E.iter_frames(E.DATASETS[ds]):
            with torch.no_grad():
                o = model(E.frame_inputs(ic, a.device, nch))
            b, s = E.dino_dets(cfg, o)
            g = E.gt_of(ic)
            c = E.is_ring_geom(b, g['mask']).astype(np.int64) if len(b) else np.zeros(0, np.int64)
            fr.append(dict(img=np.asarray(ic.converted_polar_image[0, 0]).astype(np.float16), mask=g['mask'], gt=g['gt'], gtconf=g['gtconf'], boxes=b, scores=s, cls=c))
            print(f'{ds} frame {len(fr) - 1}: {len(b)} detections', flush=True)
        sets[ds] = fr
    pickle.dump(dict(run=os.path.basename(os.path.dirname(os.path.abspath(a.ckpt))), epoch=ck.get('epoch'), ckpt=a.ckpt, sets=sets), open(a.out, 'wb'))
    print('wrote', a.out, {k: len(v) for k, v in sets.items()})


if __name__ == '__main__':
    main()
