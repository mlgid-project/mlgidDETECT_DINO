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
        # Optional higher LR for freshly-grafted modules on a warm-started detector:
        # CCTM (Cross-DINO Exp B) and the Co-DINO auxiliary heads. At the body's gentle
        # 1e-5 fine-tune rate a random/zero-init module barely moves -> it can't gain
        # traction and a null result would be a false negative. A faster rate on just
        # those params lets them earn their keep while backbone/encoder stay anchored.
        # Both mults default 1.0 (absent) -> byte-identical to the original two-group
        # split; all other configs are unaffected.
        lr_cctm_mult = getattr(args, 'lr_cctm_mult', 1.0)
        lr_cohead_mult = getattr(args, 'lr_cohead_mult', 1.0)
        split_cctm = lr_cctm_mult != 1.0
        split_cohead = lr_cohead_mult != 1.0

        def _is_split(n):
            return (split_cctm and "cctm" in n) or (split_cohead and "co_heads" in n)

        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters()
                        if "backbone" not in n and not _is_split(n) and p.requires_grad]},
            {"params": [p for n, p in model_without_ddp.named_parameters()
                        if "backbone" in n and p.requires_grad],
             "lr": args.lr_backbone},
        ]
        if split_cctm:
            param_dicts.append({"params": [p for n, p in model_without_ddp.named_parameters()
                                           if "cctm" in n and p.requires_grad],
                                "lr": args.lr * lr_cctm_mult})
        if split_cohead:
            param_dicts.append({"params": [p for n, p in model_without_ddp.named_parameters()
                                           if "co_heads" in n and p.requires_grad],
                                "lr": args.lr * lr_cohead_mult})
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