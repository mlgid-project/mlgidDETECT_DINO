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
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test
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

    def __init__(self, transforms = None, device = 'cuda'):

        self.device = 'cuda'
        self.transforms = transforms
        self.simulation = FastSimulation(device=self.device)        

    def __getitem__(self, idx):
        image = None
        while image is None:
            try:
                image, boxes, mask = self.simulation.simulate_img()
            except:
                pass 

        image = image.repeat(1, 1, 1)
        image = torchvision.utils.draw_bounding_boxes(image,boxes)
        num_objects = len(boxes[0:])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        h, w = image.shape[-2:]
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / torch.tensor([w, h, w, h], device=self.device)        
        target = {"boxes": boxes}
        target['area'] = area

        target["labels"] = torch.ones((num_objects,), dtype=torch.int64, device=self.device)
        target["image_id"] = torch.tensor(idx, device=self.device)
        target["iscrowd"] = torch.zeros((num_objects,), dtype=torch.int64, device=self.device)
        target["orig_size"] = torch.tensor(image[0].shape, device=self.device)
        target["size"] = torch.tensor(image[0].shape, device=self.device)

        return image, target

    def __len__(self):
        #number of images in epoch
        return 7#500#0
    
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

def main(args):
    #utils.init_distributed_mode(args)
    dataset = SimulationDataset()
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
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    shutil.copy(os.path.dirname(os.path.realpath(__file__)) + '/simulation.py', output_dir / 'simulation.py')

    with open(output_dir / 'settings.txt', 'a+') as f:
            f.write('\n' + str(model))
            f.write('\n' + str(args))

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        dataset = SimulationDataset()
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        epoch_start_time = time.time()
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
        
        evaluate(
            model, criterion, postprocessors, data_loader, dataset, device, args.output_dir, epoch,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )


        class ImageProcessing():

            def __init__(self, model, postprocessors) -> None:
                self.model = model
                self.postprocessors = postprocessors

            def infer(self, img: np.array, k: int = None) -> bool:
                img = Tensor(img).cuda()
                raw_results = self.model(img)
                postprocessed =  self.postprocessors['bbox'](raw_results, torch.Tensor([[512, 512]]).cuda())
                scores = postprocessed[0]['scores']
                boxes = postprocessed[0]['boxes']
                return boxes.cpu(), scores.cpu()
            
        img_process = ImageProcessing(model, postprocessors)

        def eval_ap_func(dset_path, epoch, output_dir):
            config = Config()
            config.EVAL_EPOCH = str(epoch)
            config.EVAL_OUTPUT_FOLDER = str(output_dir)
            config.INPUT_DATASET = dset_path
            config.PREPROCESSING_POLAR_SHAPE = [512,1024]
            config.PREPROCESSING_LINEAR_CONTRAST = True
            config.PREPROCESSING_LINEAR_PERC_977 = False
            data = H5GIWAXSDataset(config, path = dset_path, preprocess_func=standard_preprocessing , buffer_size=5)   
            evaluator = Evaluator()

            for i, giwaxs_img_container in enumerate(data.iter_images()):

                giwaxs_img = giwaxs_img_container.converted_polar_image
                giwaxs_img = torch.tensor(giwaxs_img[:,0,:,:]).unsqueeze(0).cuda().repeat(1,3,1,1)
                raw_giwaxs_img = giwaxs_img_container.raw_polar_image
                labels = giwaxs_img_container.polar_labels
                outputs = model(giwaxs_img)

                postprocessed =  postprocessors['bbox'](outputs, torch.Tensor([[512, 1024]]).cuda())
                
                scores = postprocessed[0]['scores']
                pred_boxes = postprocessed[0]['boxes']

                idx_keep = nms(pred_boxes, scores, 0.4)
                pred_boxes = pred_boxes[idx_keep]
                scores = scores[idx_keep]

                idx_elong = filter_non_elong(pred_boxes)
                scores = scores[idx_elong]
                pred_boxes = pred_boxes[idx_elong]

                evaluator.get_exp_metrics(pred_boxes, scores, torch.tensor(labels.boxes, device = 'cuda'), labels.confidences)
            
            recalls, precisions, accuracies, scores, av_precision, recalls_levels, fp_nums = recall_precision_curve_with_intensities(evaluator.metrics)
            df1, df2 = get_full_conf_results(evaluator.metrics)
            print(df1)
            print(df2)
            return df2['ap_total'].values[0]

        try:
            dset_name = '/data/constantin/datasets/41.h5'
            model.eval()
            eval_ap = eval_ap_func(dset_name, epoch, output_dir)
            with open(output_dir  / 'exp_ap_40_polar.txt', 'a+') as f:
                f.write(str(eval_ap) + "\n")
        except:
            pass

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
    if os.path.isfile(args.output_dir + '/checkpoint.pth'):
        args.resume = args.output_dir + '/checkpoint.pth'


    if os.path.isdir('\\'.join(args.resume.split('\\')[0:-1])):
        args.output_dir ='\\'.join(args.resume.split('\\')[0:-1])
    else:
        root = '/data/constantin/train_output/'
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.output_dir = root + 'dinodetr' + timestamp

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.export = False
    main(args)
