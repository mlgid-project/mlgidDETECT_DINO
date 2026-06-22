import numpy as np, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
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

def qdist_to_nearest_gt_q(qval, gt_qs):
    if len(gt_qs)==0: return np.inf
    return float(np.min(np.abs(gt_qs - qval)))

tp_iq=[]; fp_iq=[]            # angle-integrated I(q) percentile at the detection q
fp_qd=[]; tp_qd=[]           # q-distance to nearest GT peak (ring proxy)
with torch.no_grad():
    for ic in ds.iter_images():
        img_np=ic.converted_polar_image[0,0]            # (512,1024) HE image, masked=0
        valid=img_np>1e-6
        # angle-integrated intensity profile I(q): mean over valid angles per q-column
        denom=valid.sum(0); denom[denom==0]=1
        Iq=(img_np*valid).sum(0)/denom                  # (1024,)
        Iq_pct=np.argsort(np.argsort(Iq))/len(Iq)       # percentile rank of each q-column's I(q)
        img=torch.tensor(ic.converted_polar_image[:,0,:,:]).unsqueeze(0).cuda().repeat(1,args.num_channels,1,1)
        out=model(img); raw=[out['pred_logits'].detach().cpu().numpy(), out['pred_boxes'].detach().cpu().numpy()]
        ic=filter_boxes(cfg, onnx_to_xyxy(cfg, ic, raw))
        pred=ic.boxes; sc=ic.scores; keep=sc>ST; pred=pred[keep]
        L=ic.polar_labels
        gt=torch.tensor(np.array(L.boxes),dtype=torch.float32) if len(L.boxes) else torch.zeros((0,4))
        gt_qs=((gt[:,0]+gt[:,2])/2).numpy() if len(gt) else np.array([])
        row=np.array([],int); col=np.array([],int)
        if len(gt) and len(pred):
            try: _,row,col=matcher(gt,pred)
            except IndexError: pass
        cset=set(col.tolist())
        for j in range(len(pred)):
            q=float((pred[j,0]+pred[j,2])/2); qi=int(np.clip(q,0,1023))
            if j in cset: tp_iq.append(Iq_pct[qi]); tp_qd.append(qdist_to_nearest_gt_q(q,gt_qs))
            else:         fp_iq.append(Iq_pct[qi]); fp_qd.append(qdist_to_nearest_gt_q(q,gt_qs))
ds.close() if hasattr(ds,'close') else None
tp_iq=np.array(tp_iq); fp_iq=np.array(fp_iq); fp_qd=np.array(fp_qd); tp_qd=np.array(tp_qd)
print(f"TP={len(tp_iq)}  FP={len(fp_iq)}")
print("\n-- I(q) percentile at detection q (1.0 = on the strongest ring column) --")
print(f"  TP: mean={tp_iq.mean():.2f} p50={np.percentile(tp_iq,50):.2f}")
print(f"  FP: mean={fp_iq.mean():.2f} p50={np.percentile(fp_iq,50):.2f}")
print("\n-- q-distance to nearest GT peak (ring proxy), px --")
print(f"  TP: median={np.median(tp_qd):.1f}  (matched, ~0 by construction)")
print(f"  FP: median={np.median(fp_qd):.1f}  frac within 8px of a GT-q={np.mean(fp_qd<8):.2f}  frac >20px (off-ring)={np.mean(fp_qd>20):.2f}")
# If FPs sit at LOW I(q) percentile vs TPs, an I(q)/ring gate separates them.
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].hist(tp_iq,bins=20,range=(0,1),alpha=0.6,density=True,label='TP',color='tab:green')
ax[0].hist(fp_iq,bins=20,range=(0,1),alpha=0.6,density=True,label='FP',color='tab:red')
ax[0].set_title('I(q) percentile at detection q'); ax[0].set_xlabel('I(q) percentile'); ax[0].legend()
ax[1].hist(np.clip(fp_qd,0,60),bins=20,range=(0,60),color='tab:red'); ax[1].set_title('FP: q-dist to nearest GT peak (px)'); ax[1].set_xlabel('q distance')
fig.tight_layout(); out=__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'diagnose_rings.png'); fig.savefig(out,dpi=100,bbox_inches='tight'); print('saved',out)
