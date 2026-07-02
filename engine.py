# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device

import torch
from torchvision.ops import nms

import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
        
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def make_pseudo_targets(teacher, weak_imgs, args):
    """EMA-teacher forward on weakly-augmented real frames -> DINO-format pseudo targets.

    weak_imgs: (B,1,H,W) cuda float in [0,1]. Returns a list of B target dicts matching the
    schema SimulationDataset emits (main.py): normalized cxcywh boxes, int64 labels
    (segment=0 / ring=1), all on cuda. The teacher is in eval mode and called without
    targets, so prepare_for_cdn returns all-None and no DN queries are built.
    Per-class score thresholds (pseudo_thr_seg lower than pseudo_thr_ring) chase faint
    segment-peak recall; class-aware NMS mirrors the production filter_boxes values
    (ring IoU 0.1 / segment 0.4). IoU is invariant to the per-axis normalization, so NMS
    on normalized boxes equals pixel-space NMS. Empty pseudo-targets are valid (the image
    then only contributes negatives). See docs/SEMI_DETR_INTEGRATION.md S4c.
    """
    teacher.eval()
    H, W = weak_imgs.shape[-2:]
    device = weak_imgs.device
    x = weak_imgs.repeat(1, args.num_channels, 1, 1)
    outputs = teacher(x)
    logits, boxes = outputs['pred_logits'], outputs['pred_boxes']   # (B,Q,C), (B,Q,4) cxcywh norm
    targets = []
    for b in range(x.shape[0]):
        prob = logits[b].sigmoid()
        score, cls = prob.max(-1)
        thr = torch.where(cls == 1,
                          score.new_tensor(args.pseudo_thr_ring),
                          score.new_tensor(args.pseudo_thr_seg))
        keep = score > thr
        bx, sc, lb = boxes[b][keep], score[keep], cls[keep]
        if bx.shape[0] > 1:
            xyxy = box_cxcywh_to_xyxy(bx)
            keep_idx = []
            for c, iou in ((0, 0.4), (1, 0.1)):
                cidx = (lb == c).nonzero(as_tuple=True)[0]
                if cidx.numel():
                    keep_idx.append(cidx[nms(xyxy[cidx], sc[cidx], iou)])
            keep_idx = torch.cat(keep_idx)
            bx, sc, lb = bx[keep_idx], sc[keep_idx], lb[keep_idx]
        n = bx.shape[0]
        bx = bx.float()                                              # fp32 like SimulationDataset targets (amp: teacher emits fp16)
        targets.append({
            "boxes": bx,                                             # cxcywh, normalized, cuda
            "area": (bx[:, 2] * W) * (bx[:, 3] * H),                 # pixel area, as SimulationDataset
            "labels": lb.to(torch.int64),
            "image_id": torch.tensor(-1, device=device),             # sentinel: unlabeled frame
            "iscrowd": torch.zeros((n,), dtype=torch.int64, device=device),
            "orig_size": torch.tensor([H, W], device=device),
            "size": torch.tensor([H, W], device=device),
        })
    return targets


def train_one_epoch_semi(model: torch.nn.Module, criterion: torch.nn.Module,
                         data_loader: Iterable, real_loader: Iterable,
                         optimizer: torch.optim.Optimizer,
                         device: torch.device, epoch: int, max_norm: float = 0,
                         wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    """Semi-supervised epoch (Semi-DETR MVP, docs/SEMI_DETR_INTEGRATION.md S4e):
    total = L_sup(synthetic, DN on) + lambda(epoch) * L_unsup(real pseudo-labeled, DN off).
    Before args.semi_start_epoch this reduces to the plain supervised epoch (burn-in).
    """
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    need_tgt_for_training = args.use_dn

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}] (semi)'.format(epoch)
    print_freq = 10

    semi_on = epoch >= args.semi_start_epoch
    lam = 0.0
    if semi_on:
        # one-time hard seed of the teacher at the semi boundary (utils.ModelEma.set);
        # on a resume past the boundary this does not re-fire and the restored ema_model is used
        if epoch == args.semi_start_epoch:
            ema_m.set(model)
        lam = args.unsup_loss_weight * min(1.0, (epoch - args.semi_start_epoch + 1)
                                           / max(1, args.unsup_warmup_epochs))
    # stage-wise hybrid matching (phase 3, default off): one-to-many on the unsupervised
    # branch for the first hybrid_t1_epochs of the semi phase, then one-to-one
    hybrid_on = (semi_on and getattr(args, 'hybrid_matching', False)
                 and epoch < args.semi_start_epoch + args.hybrid_t1_epochs)

    real_iter = iter(real_loader) if semi_on else None

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            # ---- supervised branch (unchanged: DN on) ----
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # ---- unsupervised branch (real frames, teacher pseudo-labels) ----
            unsup_value = 0.0
            n_pseudo_seg = n_pseudo_ring = 0
            if semi_on and lam > 0:
                try:
                    weak, strong = next(real_iter)
                except StopIteration:
                    real_iter = iter(real_loader)
                    weak, strong = next(real_iter)
                weak = weak.to(device, non_blocking=True)
                strong = strong.to(device, non_blocking=True)
                pseudo = make_pseudo_targets(ema_m.module, weak, args)
                n_pseudo_seg = sum(int((t['labels'] == 0).sum()) for t in pseudo)
                n_pseudo_ring = sum(int((t['labels'] == 1).sum()) for t in pseudo)

                strong_in = strong.repeat(1, args.num_channels, 1, 1)
                # DN off for pseudo-targets: prepare_for_cdn would iterate targets=None in
                # train mode, so take the no-DN path by zeroing dn_number for this forward
                _dn = model.dn_number
                model.dn_number = 0
                try:
                    out_uns = model(strong_in)
                finally:
                    model.dn_number = _dn
                if hybrid_on:
                    criterion.use_o2m = True
                try:
                    ld_uns = criterion(out_uns, pseudo)
                finally:
                    criterion.use_o2m = False
                l_uns = sum(ld_uns[k] * weight_dict[k] for k in ld_uns.keys() if k in weight_dict)
                losses = losses + lam * l_uns
                unsup_value = float(lam * l_uns.detach())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item() + unsup_value

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if semi_on:
            metric_logger.update(loss_unsup=unsup_value, lam=lam,
                                 pseudo_seg=float(n_pseudo_seg), pseudo_ring=float(n_pseudo_ring))

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, epoch, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            # outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}


        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:

            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": int(image_id), 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)        

    return final_res
