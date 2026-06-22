import numpy as np, torch
from main import get_args_parser, build_model_main
from util.slconfig import SLConfig
from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.evaluation import Evaluator, get_full_conf_results
from torch import Tensor

CKPT='/mnt/lustre/work/schreiber/szb389/train_output/ringseg_2class_20260603-142434/checkpoint.pth'
DSET='/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5'
args=get_args_parser().parse_args([])
for k,v in SLConfig.fromfile('config/DINO/DINO_4scale_swin.py')._cfg_dict.to_dict().items(): setattr(args,k,v)
args.device='cuda'; args.export=False
model,_,_=build_model_main(args); model=model.cuda().eval()
model.load_state_dict(torch.load(CKPT,map_location='cpu')['model'])

# cache predictions ONCE (pre-NMS): raw boxes/scores/labels + GT, so we only re-run NMS per sweep value
frames=[]
cfg0=Config(); cfg0.PREPROCESSING_POLAR_SHAPE=[512,1024]; cfg0.POSTPROCESSING_SCORE=0.0
cfg0.POSTPROCESSING_CLASSAWARE_NMS=False; cfg0.POSTPROCESSING_NMSIOU=1.01  # no NMS, keep all top-225
cfg0.INPUT_DATASET=DSET
ds=PyGIDDataset(cfg0,path=DSET,preprocess_func=standard_preprocessing,buffer_size=5,load_labels=True)
with torch.no_grad():
    for ic in ds.iter_images():
        img=torch.tensor(ic.converted_polar_image[:,0,:,:]).unsqueeze(0).cuda().repeat(1,args.num_channels,1,1)
        out=model(img); raw=[out['pred_logits'].detach().cpu().numpy(), out['pred_boxes'].detach().cpu().numpy()]
        ic=onnx_to_xyxy(cfg0, ic, raw)   # top-225, no filtering yet
        frames.append((ic.boxes.clone(), ic.scores.clone(), ic.pred_labels.clone(),
                       Tensor(np.array(ic.polar_labels.boxes)), np.array(ic.polar_labels.confidences)))
ds.close() if hasattr(ds,'close') else None

from torchvision.ops import nms
def run(seg_iou, ring_iou=0.1, score=0.1):
    ev=Evaluator()
    for boxes,scores,labels,gt,conf in frames:
        keep=[]
        for cls,iou in [(1,ring_iou),(0,seg_iou)]:
            idx=(labels==cls).nonzero(as_tuple=True)[0]
            if idx.numel(): keep.append(idx[nms(boxes[idx],scores[idx],iou)])
        ki=torch.cat(keep) if keep else torch.empty(0,dtype=torch.long)
        b=boxes[ki]; s=scores[ki]; tk=s>score; b=b[tk]; s=s[tk]
        ev.get_exp_metrics(b, s, gt, conf)
    df1,df2=get_full_conf_results(ev.metrics)
    return df2['ap_total'].values[0], df1
print("seg-NMS-IoU sweep on organic (ring IoU fixed 0.1):  ap_total | recall@bestacc | fp@bestacc")
for seg in [0.4,0.3,0.2,0.15,0.1]:
    ap,df1=run(seg)
    r=df1['recall_total'].iloc[0]; fp=df1['fp'].iloc[0]
    print(f"  seg_iou={seg:.2f}: ap_total={ap:.4f}   recall={r:.3f}   fp_frac={fp:.3f}")
