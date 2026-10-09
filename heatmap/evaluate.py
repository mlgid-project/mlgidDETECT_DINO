"""Score ssl1 and/or heatmap checkpoints with ONE code path.
  python heatmap/evaluate.py ssl1=dino:<ckpt.pth> hm=heatmap:<ckpt.pth> [...]
Heatmap checkpoints are scored twice: native (peak picking, no NMS) and `+nms` (shared filter_boxes)."""
import os, sys, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if not any(sp.partition('=')[2].startswith('dino:') for sp in sys.argv[1:]):
    # heatmap-only scoring needs no compiled DINO ops: skip the models package inits (as train.py does).
    # A run that scores a DINO checkpoint (cluster, ops built) keeps the real package.
    import types
    for _n, _s in (('models', 'models'), ('models.dino', 'models/dino')):
        _m = types.ModuleType(_n); _m.__path__ = [os.path.join(ROOT, _s)]; sys.modules[_n] = _m
import numpy as np
import torch
from heatmap import evaluation as E

DEV = 'cuda'


def load_dino(ckpt):
    from main import build_model_main
    import util.misc as utils
    with open(os.path.join(os.path.dirname(ckpt), 'config_args_all.json')) as f:
        args = argparse.Namespace(**json.load(f))
    model, _, _ = build_model_main(args)
    ck = torch.load(ckpt, map_location='cpu')
    print('  load:', model.load_state_dict(utils.clean_state_dict(ck.get('model', ck)), strict=False))
    return model.to(DEV).eval(), args


def load_heatmap(ckpt):
    from models.heatmap_head import HeatmapNet
    ck = torch.load(ckpt, map_location='cpu')
    a = ck['hm_args']
    model = HeatmapNet(backbone_ckpt=None, freeze_backbone=a['freeze_backbone'], out_stride=a['out_stride'], amp_backbone=bool(a.get('amp_backbone', False) or os.environ.get('HM_AMP') == '1'), chan_mode=a.get('chan_mode', 'he'), zero_invalid=a.get('zero_invalid', False))
    print('  load:', model.load_state_dict(ck['model'], strict=False).unexpected_keys[:3])
    # backbone weights are re-read from their source file (frozen => identical to training)
    if a.get('bb') == 'random':               # backbone weights are inside the checkpoint
        return model.to(DEV).eval(), a
    bb = torch.load(os.environ.get('HM_BB_PATH', a['bb_path']), map_location='cpu')
    from util.misc import clean_state_dict
    bb = clean_state_dict(bb.get('model', bb))
    pre = a['bb_prefix']
    bb = {k[len(pre):]: v for k, v in bb.items() if k.startswith(pre) and 'head' not in k[len(pre):]}
    if a['freeze_backbone']:
        print('  backbone:', model.backbone.load_state_dict(bb, strict=False))
    return model.to(DEV).eval(), a


@torch.no_grad()
def run(name, kind, ckpt):
    from models.heatmap_head import decode
    if kind == 'dino':
        model, args = load_dino(ckpt); nch = args.num_channels
    else:
        model, a = load_heatmap(ckpt); nch = 1
    out = {}
    for ds, path in E.DATASETS.items():
        gts, d_main, d_nms = [], [], []
        for cfg, ic in E.iter_frames(path):
            img = E.frame_inputs(ic, DEV, nch, chan_mode=getattr(model, 'chan_mode', 'he'))
            o = model(img) if kind == 'dino' else model(img, E.frame_mask(ic, DEV))
            gts.append(E.gt_of(ic))
            if kind == 'dino':
                d_main.append(E.dino_dets(cfg, o))
            else:
                pi = decode(o, model.out_stride, num_select=225)[0]
                d_main.append(E.heatmap_dets(cfg, pi, use_nms=False))
                d_nms.append(E.heatmap_dets(cfg, pi, use_nms=True))
        r = E.evaluate_dets(d_main, gts, cfg)
        print(E.format_result(name if kind == 'dino' else name + ' (native)', ds, r), flush=True)
        out[(name, ds)] = r
        if d_nms:
            r2 = E.evaluate_dets(d_nms, gts, cfg)
            print(E.format_result(name + '+nms', ds, r2), flush=True)
    return out


if __name__ == '__main__':
    for spec in sys.argv[1:]:
        name, rest = spec.split('=', 1)
        kind, ckpt = rest.split(':', 1)
        print(f'\n===== {name} ({kind}) {ckpt}', flush=True)
        run(name, kind, ckpt)
