# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import shutil
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
from util.configuration import Config
from util.evaluation import Evaluator, get_full_conf_results, recall_precision_curve_with_intensities
from util.exp_preprocess import standard_preprocessing
from util.labeleddataset import H5GIWAXSDataset
from util.pygidloader import PyGIDDataset, detect_dataset_type
import util.misc as utils
from util.postprocessing import onnx_to_xyxy, filter_boxes

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, train_one_epoch_semi, test
from simulation import FastSimulation
import pickle
from torch import Tensor
from torchvision.utils import save_image
import torch.multiprocessing as mp
import torchvision
from torchvision.utils import save_image
from torchvision.ops import nms

def filter_non_elong(pred_boxes):
    y_extent = pred_boxes[:,3] - pred_boxes[:,1]
    x_extent = pred_boxes[:,2] - pred_boxes[:,0]
    keep = x_extent*1.15 < y_extent
    return keep

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

class SimulationDataset(torch.utils.data.Dataset):

    def __init__(self, args, transforms = None, device = 'cuda'):

        self.args = args
        self.device = 'cuda'
        self.transforms = transforms
        self.simulation = FastSimulation(device=self.device)        

    def __getitem__(self, idx):
        image = None
        while image is None:
            try:
                image, boxes, mask, is_ring = self.simulation.simulate_img()
            except:
                pass

        image = image.repeat(self.args.num_channels, 1, 1)
        num_objects = len(boxes[0:])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        h, w = image.shape[-2:]
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / torch.tensor([w, h, w, h], device=self.device)
        target = {"boxes": boxes}
        target['area'] = area

        #learned ring/segment class: segment=0, ring=1 (label = int(is_ring))
        target["labels"] = is_ring.to(dtype=torch.int64, device=self.device)
        target["image_id"] = torch.tensor(idx, device=self.device)
        target["iscrowd"] = torch.zeros((num_objects,), dtype=torch.int64, device=self.device)
        target["orig_size"] = torch.tensor(image[0].shape, device=self.device)
        target["size"] = torch.tensor(image[0].shape, device=self.device)

        return image, target

    def __len__(self):
        #number of images in epoch
        return 1000
    
def collate_fn(batch):
    # Initialize lists to hold the tensors
    samples = []
    targets = []

    # Iterate over the batch
    for item in batch:
        # Unpack the item
        image_tensor, target_dict = item

        # Append the image tensor to the samples list
        samples.append(image_tensor)

        # Append the target dict to the targets list
        targets.append(target_dict)

    # Convert the lists to tensors
    samples = torch.stack(samples)
    #targets = [{'boxes': torch.stack([t['boxes'] for t in targets])}]

    # Return a dictionary
    #return {'samples': samples, 'targets': targets}
    return samples, targets



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', default=os.path.dirname(os.path.realpath(__file__)) + '/config/DINO/DINO_4scale_swin.py', type=str, required=False)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    # Distinct dest so this does not collide with a config-defined `eval_file`
    # (the cfg/args merge forbids a key existing in both). CLI value overrides the config.
    parser.add_argument('--eval_file', dest='eval_file_cli', default=None, type=str,
                        help='path to a labeled .h5 (pygid or roi_data) for GIWAXS AP evaluation; '
                             'overrides eval_file from the config file')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def evaluate_giwaxs_ap(model, postprocessors, args, dset_path, epoch, output_dir):
    """Run GIWAXS labeled AP evaluation on a labeled .h5 file (pygid or roi_data).

    Auto-detects the dataset layout (``detect_dataset_type``): pyGID/NeXus files with
    ``data/img_gid_q`` + ``fitted_peaks`` GT route through ``PyGIDDataset``; older
    ``roi_data`` files through ``H5GIWAXSDataset``.

    Postprocessing is intentionally identical to the deployed mlgidDETECT dino path: the
    live model's raw outputs (pred_logits, pred_boxes) are pushed through the ported
    ``onnx_to_xyxy`` (top-225 + cxcywh->xyxy) and ``filter_boxes`` (single NMS at
    POSTPROCESSING_NMSIOU, then score > POSTPROCESSING_SCORE), so the metrics here reflect
    what the exported ONNX model produces in production. Returns ``ap_total``.
    """
    config = Config()
    config.EVAL_EPOCH = str(epoch)
    config.EVAL_OUTPUT_FOLDER = str(output_dir)
    config.INPUT_DATASET = dset_path
    config.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    #match mlgidDETECT's eval_on_dataset: lower the score threshold so the PR curve is fully sampled
    config.POSTPROCESSING_SCORE = 0.1
    #2-class ring/segment model: use class-aware NMS (ring=1 IoU 0.1 / segment=0 IoU 0.4)
    config.POSTPROCESSING_CLASSAWARE_NMS = True
    if detect_dataset_type(dset_path) == 'pygid':
        #pyGID/NeXus labeled file (img_gid_q + fitted_peaks GT)
        data = PyGIDDataset(config, path=dset_path, preprocess_func=standard_preprocessing, buffer_size=5, load_labels=True)
    else:
        #roi_data-style labeled file
        data = H5GIWAXSDataset(config, path=dset_path, preprocess_func=standard_preprocessing, buffer_size=5)
    evaluator = Evaluator()

    for i, giwaxs_img_container in enumerate(data.iter_images()):
        giwaxs_img = giwaxs_img_container.converted_polar_image
        giwaxs_img = torch.tensor(giwaxs_img[:, 0, :, :]).unsqueeze(0).cuda().repeat(1, args.num_channels, 1, 1)
        labels = giwaxs_img_container.polar_labels
        outputs = model(giwaxs_img)

        #mimic the ONNX session outputs [pred_logits, pred_boxes] and run the deployed postprocessing
        raw_results = [outputs['pred_logits'].detach().cpu().numpy(),
                       outputs['pred_boxes'].detach().cpu().numpy()]
        giwaxs_img_container = onnx_to_xyxy(config, giwaxs_img_container, raw_results)
        giwaxs_img_container = filter_boxes(config, giwaxs_img_container)
        pred_boxes = giwaxs_img_container.boxes
        scores = giwaxs_img_container.scores

        evaluator.get_exp_metrics(pred_boxes, scores, torch.tensor(labels.boxes), labels.confidences)

    df1, df2 = get_full_conf_results(evaluator.metrics)
    print(df1)
    print(df2)
    return df2['ap_total'].values[0]

def main(args):
    #utils.init_distributed_mode(args)
    dataset = SimulationDataset(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')


    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)                

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)        


    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        #This GIWAXS fork has no COCO val loader; evaluate against a labeled GIWAXS .h5 instead.
        dset_name = args.eval_file_cli or getattr(args, 'eval_file', None)
        if not dset_name:
            raise ValueError("No evaluation dataset given. Pass --eval_file <labeled .h5> "
                             "or set eval_file in the config file.")
        model.eval()
        eval_ap = evaluate_giwaxs_ap(model, postprocessors, args, dset_name, args.start_epoch, output_dir)
        with open(output_dir / 'exp_ap_40_polar.txt', 'a+') as f:
            f.write(str(eval_ap) + "\n")
        return

    shutil.copy(os.path.dirname(os.path.realpath(__file__)) + '/simulation.py', output_dir / 'simulation.py')

    with open(output_dir / 'settings.txt', 'a+') as f:
            f.write('\n' + str(model))
            f.write('\n' + str(args))

    #semi-supervised (Semi-DETR MVP): real unlabeled corpus loader, built ONCE (static corpus,
    #unlike SimulationDataset which is rebuilt per epoch). CPU dataset -> workers allowed.
    real_loader = None
    if getattr(args, 'use_semi', False):
        assert args.use_ema, "use_semi requires use_ema=True (the EMA teacher generates pseudo-labels)"
        from datasets.real_unlabeled import RealUnlabeledDataset, collate_unlabeled
        real_ds = RealUnlabeledDataset(args.unlabeled_h5, strong=args.strong_aug,
                                       geom_flip=args.semi_geom_flip)
        real_loader = DataLoader(real_ds, batch_size=args.unlabeled_batch_size, shuffle=True,
                                 num_workers=args.unlabeled_workers, collate_fn=collate_unlabeled,
                                 drop_last=True, pin_memory=True,
                                 persistent_workers=args.unlabeled_workers > 0)
        logger.info("semi: {} real unlabeled frames from {}; semi_start_epoch={} lambda={} "
                    "thr(ring/seg)={}/{}".format(len(real_ds), args.unlabeled_h5,
                    args.semi_start_epoch, args.unsup_loss_weight,
                    args.pseudo_thr_ring, args.pseudo_thr_seg))

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        dataset = SimulationDataset(args)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        epoch_start_time = time.time()
        if getattr(args, 'use_semi', False):
            train_stats = train_one_epoch_semi(
                model, criterion, data_loader, real_loader, optimizer, device, epoch,
                args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
        else:
            train_stats = train_one_epoch(
                model, criterion, data_loader, optimizer, device, epoch,
                args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)
                
        # eval
        with open(output_dir / 'training_stats.txt', 'a+') as f:
            f.write('epoch: ' + str(epoch) + str(train_stats) + "\n")
        with open(output_dir  / 'bbox_loss.txt', 'a+') as f:
            f.write('epoch: ' + str(epoch) + ' loss_bbox: ' + str(train_stats['loss_bbox']) + "\n")
        with open(output_dir  / 'loss_giou.txt', 'a+') as f:
            f.write('epoch: ' + str(epoch) + ' loss_giou: ' + str(train_stats['loss_giou']) + "\n")

        #GIWAXS labeled eval on each configured dataset (e.g. 41 + organic), every eval_interval
        #epochs; each is wrapped so a failure on one never aborts training or skips the others.
        eval_interval = getattr(args, 'eval_interval', 2)
        if epoch % eval_interval == 0:
            eval_targets = getattr(args, 'eval_files', None) or {}
            if not eval_targets:
                #fall back to a single dataset (CLI override or config eval_file)
                single = args.eval_file_cli or getattr(args, 'eval_file', None)
                if single:
                    eval_targets = {'eval': single}
            model.eval()
            for name, path in eval_targets.items():
                try:
                    eval_ap = evaluate_giwaxs_ap(model, postprocessors, args, path, epoch, output_dir)
                    with open(output_dir / f'exp_ap_{name}.txt', 'a+') as f:
                        f.write(f'{epoch}\t{eval_ap}\n')
                    print(f'[epoch {epoch}] {name} ap_total = {eval_ap}')
                except Exception as e:
                    print(f'[epoch {epoch}] eval on {name} ({path}) failed: {type(e).__name__}: {e}')
            #in the semi phase, also track the EMA teacher's AP (often the stronger model in
            #mean-teacher training and the natural deployment candidate)
            if (getattr(args, 'use_semi', False) and ema_m is not None
                    and epoch >= args.semi_start_epoch):
                for name, path in eval_targets.items():
                    try:
                        eval_ap = evaluate_giwaxs_ap(ema_m.module, postprocessors, args, path, epoch, output_dir)
                        with open(output_dir / f'exp_ap_{name}_teacher.txt', 'a+') as f:
                            f.write(f'{epoch}\t{eval_ap}\n')
                        print(f'[epoch {epoch}] {name} ap_total (teacher) = {eval_ap}')
                    except Exception as e:
                        print(f'[epoch {epoch}] teacher eval on {name} ({path}) failed: {type(e).__name__}: {e}')
            model.train()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    # Allow --resume to point at a run directory that contains checkpoint.pth.
    if args.resume and os.path.isdir(args.resume):
        args.resume = os.path.join(args.resume, 'checkpoint.pth')

    # Pick up an existing checkpoint from --output_dir when --resume was not given.
    if not args.resume and args.output_dir and os.path.isfile(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')

    # Honor an explicit --output_dir; otherwise derive one (next to the checkpoint,
    # else the config root_dir + a timestamp). The previous logic split on '\\' and so
    # never matched on POSIX, always falling back to root_dir and ignoring --output_dir.
    if not args.output_dir:
        if args.resume:
            args.output_dir = os.path.dirname(args.resume)
        else:
            cfg = SLConfig.fromfile(args.config_file)
            root = cfg._cfg_dict.to_dict().get('root_dir', '')
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            args.output_dir = root + 'dinodetr' + timestamp

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.export = False
    main(args)
