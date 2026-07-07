import json
import torch
import torch.nn as nn


def match_name_keywords(n: str, name_keywords: list):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_param_dict(args, model_without_ddp: nn.Module):
    try:
        param_dict_type = args.param_dict_type
    except:
        param_dict_type = 'default'
    assert param_dict_type in ['default', 'ddetr_in_mmdet', 'large_wd']

    # by default
    if param_dict_type == 'default':
        # Optional higher LR for the CCTM feature-enrichment module (Cross-DINO Exp B).
        # CCTM is grafted on top of a CONVERGED warm-start, so at the body's gentle
        # fine-tune rate its zero-init LayerScale (gamma) barely moves -> the module
        # can't gain traction and a null result would be a false negative. A faster
        # rate on just the cctm.* params lets it earn its way in while the backbone /
        # encoder stay anchored. lr_cctm_mult=1.0 (default/absent) -> identical to the
        # original two-group split; all non-CCTM configs are unaffected.
        lr_cctm_mult = getattr(args, 'lr_cctm_mult', 1.0)
        if lr_cctm_mult != 1.0:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters()
                            if "backbone" not in n and "cctm" not in n and p.requires_grad]},
                {"params": [p for n, p in model_without_ddp.named_parameters()
                            if "backbone" in n and p.requires_grad],
                 "lr": args.lr_backbone},
                {"params": [p for n, p in model_without_ddp.named_parameters()
                            if "cctm" in n and p.requires_grad],
                 "lr": args.lr * lr_cctm_mult},
            ]
        else:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": args.lr_backbone,
                }
            ]
        return param_dicts

    if param_dict_type == 'ddetr_in_mmdet':
        param_dicts = [
            {
                "params":
                    [p for n, p in model_without_ddp.named_parameters()
                        if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() 
                        if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
                "lr": args.lr_backbone,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() 
                        if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr * args.lr_linear_proj_mult,
            }
        ]        
        return param_dicts

    if param_dict_type == 'large_wd':
        param_dicts = [
                {
                    "params":
                        [p for n, p in model_without_ddp.named_parameters()
                            if not match_name_keywords(n, ['backbone']) and not match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, ['backbone']) and match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr_backbone,
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() 
                            if match_name_keywords(n, ['backbone']) and not match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr_backbone,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params":
                        [p for n, p in model_without_ddp.named_parameters()
                            if not match_name_keywords(n, ['backbone']) and match_name_keywords(n, ['norm', 'bias']) and p.requires_grad],
                    "lr": args.lr,
                    "weight_decay": 0.0,
                }
            ]

        # print("param_dicts: {}".format(param_dicts))

    return param_dicts