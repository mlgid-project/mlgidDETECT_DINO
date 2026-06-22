import numpy as np, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.patches as mp
from main import get_args_parser, build_model_main
from util.slconfig import SLConfig
from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.matchers import get_matcher

CKPT='/mnt/lustre/work/schreiber/szb389/train_output/ringseg_2class_20260603-142434/checkpoint.pth'
DSET='/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5'
ST=0.3
args=get_args_parser().parse_args([])
for k,v in SLConfig.fromfile('config/DINO/DINO_4scale_swin.py')._cfg_dict.to_dict().items(): setattr(args,k,v)
args.device='cuda'; args.export=False
model,_,_=build_model_main(args); model=model.cuda().eval()
model.load_state_dict(torch.load(CKPT,map_location='cpu')['model'])
cfg=Config(); cfg.PREPROCESSING_POLAR_SHAPE=[512,1024]; cfg.POSTPROCESSING_SCORE=0.1
cfg.POSTPROCESSING_CLASSAWARE_NMS=True; cfg.INPUT_DATASET=DSET
ds=PyGIDDataset(cfg,path=DSET,preprocess_func=standard_preprocessing,buffer_size=5,load_labels=True)
matcher=get_matcher('q',min_iou=0.1)
imgs=[]
with torch.no_grad():
    for ic in ds.iter_images():
        a=ic.converted_polar_image[0,0].copy()
        img=torch.tensor(ic.converted_polar_image[:,0,:,:]).unsqueeze(0).cuda().repeat(1,args.num_channels,1,1)
        out=model(img); raw=[out['pred_logits'].detach().cpu().numpy(),out['pred_boxes'].detach().cpu().numpy()]
        ic=filter_boxes(cfg,onnx_to_xyxy(cfg,ic,raw))
        pred=ic.boxes; sc=ic.scores; keep=sc>ST; pred=pred[keep]; sc=sc[keep]
        gt=torch.tensor(np.array(ic.polar_labels.boxes),dtype=torch.float32) if len(ic.polar_labels.boxes) else torch.zeros((0,4))
        row=np.array([],int); col=np.array([],int)
        if len(gt) and len(pred):
            try: _,row,col=matcher(gt,pred)
            except IndexError: pass
        cset=set(col.tolist())
        tp=[pred[j].numpy() for j in range(len(pred)) if j in cset]
        fp=[pred[j].numpy() for j in range(len(pred)) if j not in cset]
        imgs.append((a, gt.numpy(), np.array(tp), np.array(fp)))
        if len(imgs)>=4: break
ds.close() if hasattr(ds,'close') else None
fig,axes=plt.subplots(2,2,figsize=(16,8))
for ax,(a,gt,tp,fp) in zip(axes.ravel(),imgs):
    ax.imshow(a,origin='lower',aspect='auto',cmap='gray')
    for b in gt: ax.add_patch(mp.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,ec='lime',lw=0.7))
    for b in tp: ax.add_patch(mp.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,ec='deepskyblue',lw=0.7,ls='--'))
    for b in fp: ax.add_patch(mp.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,ec='red',lw=0.9))
    ax.set_title(f'GT={len(gt)} (green)  TP={len(tp)} (blue dash)  FP={len(fp)} (red)'); ax.set_xlabel('q'); ax.set_ylabel('angle')
fig.suptitle('Are the red FPs real unlabeled peaks? (GT=green, matched=blue, unmatched/FP=red)')
fig.tight_layout(); out=__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'viz_fp.png'); fig.savefig(out,dpi=95,bbox_inches='tight'); print('saved',out)
